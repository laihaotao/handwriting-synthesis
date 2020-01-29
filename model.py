import numpy as np
import torch
import torch.nn as nn

class HandwritingPrediction(nn.Module):
    def __init__(self, hidden_size, mix_components, feature_dim):
        super(HandwritingPrediction, self).__init__()
        self.tanh = nn.Tanh()
        self.lstm1 = nn.LSTM(input_size=feature_dim,
                             hidden_size=hidden_size,
                             batch_first=True)

        # cancate the output from 1st layer with input x
        lstm2_input_size = feature_dim + hidden_size
        self.lstm2 = nn.LSTM(input_size=lstm2_input_size,
                             hidden_size=hidden_size,
                             batch_first=True)

        # formula (17) from the paper
        # use 20 mixture components, since a point (m, n) have two coordinates so it is a 2D guassian
        # which contains 40 means and 40 standard diviations (20 for m and 20 for n)
        # the total will be 20 weights + 20 correlations + 40 means + 40 std + 1 end of stroke = 121
        mdn_in_size = 2 * hidden_size
        mdn_out_size = 1 + ((1 + 1 + 2 + 2) * mix_components)
        self.linear = nn.Linear(mdn_in_size, mdn_out_size)

    def forward(self, x, prev1, prev2):
        # print('x shape: {}'.format(x.shape))

        h1, (h1_n, c1_n) = self.lstm1(x, prev1)
        x2 = torch.cat([h1, x], dim=-1)  # skip connection
        h2, (h2_n, c2_n) = self.lstm2(x2, prev2)
        h = torch.cat([h1, h2], dim=-1)  # skip connection

        # mixture of guassian computation, in the paper, fomular(15) ~ (22)
        params = self.linear(h)
        mdn_params = params.narrow(-1, 0, params.size()[-1] - 1)
        pre_weights, mu1, mu2, log_sigma1, log_sigma2, pre_rho = mdn_params.chunk(
            6, dim=-1)
        end = torch.sigmoid(params.narrow(-1, params.size()[-1] - 1, 1))
        weights = torch.softmax(pre_weights,
                            dim=-1)  # softmax make sure the sum will be 1
        rho = self.tanh(pre_rho)
        sigma1, sigma2 = torch.exp(log_sigma1), torch.exp(log_sigma2)
        return end, weights, mu1, mu2, sigma1, sigma2, \
               rho, (h1_n, c1_n), (h2_n, c2_n)

    def init_hidden(self):
        pass

# one LSTM layer with Attention mechanism attacted
class LSTM_A(nn.Module):
    def __init__(self,
                 device,
                 sent_max_len,
                 lstm1_in_size,
                 lstm1_out_size,
                 win_in_size,
                 win_out_size,
                 vars_per_fuct):
        super(LSTM_A, self).__init__()
        self.device = device
        self.sent_max_len = sent_max_len
        self.vars_per_fuct = vars_per_fuct
        self.lstm1 = nn.LSTMCell(lstm1_in_size, lstm1_out_size)
        self.softwindow = nn.Linear(win_in_size, win_out_size)

    def forward(self, strks, onehots, sents_m, w_prev, k_prev, prev):
        timesteps = strks.size()[1]
        sent_real_len = sents_m.sum(dim=1)
        w_prev = w_prev.squeeze(1)
        # print('w_prev shape:', w_prev.shape)
        # print('k_prev shape:', k_prev.shape)
        h1_list, wt_list = [], []
        for t in range(timesteps):
            # each stroke_t corresponding to a char of sentence_t
            # concat the stroke feature and sentence feature
            input_t = torch.cat((strks.narrow(1, t, 1).squeeze(1), w_prev), dim=1)

            # print('input_t shape:', input_t.shape)

            prev = self.lstm1(input_t, prev)
            h1_t = prev[0]

            nn.utils.clip_grad_value_(h1_t, 10)
            h1_list.append(h1_t)

            # attention mechanisim
            p = self.softwindow(h1_t)
            a, b, k = p.chunk(self.vars_per_fuct, dim=1)
            a, b, k = a.exp(), b.exp(), k_prev + k.exp()

            # print('a shape:', a.shape)
            # print('k shape:', k.shape)

            # compute the "dist" between current pos and all positions
            u = torch.from_numpy(np.array(range(self.sent_max_len + 1))
                ).type(torch.FloatTensor).to(self.device)
            # print('u shape:', u.shape)

            # phi = torch.sum(a * torch.exp(-1 * b * (k - u)**2), dim=1)
            # phi = torch.unsqueeze(phi * sents_m, dim=2)
            # print('phi shape:', phi.shape)
            # exit(0)

            gravity = -1 * b.unsqueeze(2) * (k.unsqueeze(2).repeat(
                1, 1, self.sent_max_len + 1) - u)**2

            # print(sent_real_len.shape)
            # print('a shape:', a.unsqueeze(2).shape)
            # print('gravity shape:', gravity.shape)

            phi = (a.unsqueeze(2) * gravity.exp()).sum(dim=1) \
                * (self.sent_max_len / sent_real_len.unsqueeze(1))

            # print('phi shape:', phi.shape)
            # print('onehots shape:', onehots.shape)
            # exit(0)

            # w_t
            # w_t = torch.sum(phi * onehots, dim=1, keepdim=True)
            w_t = torch.sum(
                phi.narrow(-1, 0, self.sent_max_len).unsqueeze(2) *onehots, dim=1)
            # print(w_t.shape)
            wt_list.append(w_t)

            # update parameter for next iteration
            k_prev = k
            w_prev = w_t

        # collection the hidden state from LSTM1 for LSTM2
        hid1 = torch.stack(h1_list, dim=1)  # (batch, timesteps, hidden_size)
        win_vec = torch.stack(wt_list, dim=1).squeeze(2)  # (batch, timesteps, len(alphabet))

        return hid1, prev, win_vec, w_prev, k_prev, phi


class HandwritingSynthesis(nn.Module):
    def __init__(self,
                 device,
                 sent_max_len,
                 batch_size,
                 hidden_size,
                 K,
                 mix_components,
                 vars_per_fuct=3):
        super(HandwritingSynthesis, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.K = K
        self.num_layers = 3

        # strk_dim = 3; sent_dim = 60
        strk_dim, sent_dim = (3, 60)
        lstm1_in_size = strk_dim + sent_dim
        win_in_size  = hidden_size
        win_out_size = self.K * vars_per_fuct  # 3K
        mdn_out_size = 1 + ((1 + 1 + 2 + 2) * mix_components)

        self.lstm1 = LSTM_A(device, sent_max_len, lstm1_in_size, hidden_size,
                            win_in_size, win_out_size, vars_per_fuct)
        self.lstm2 = nn.LSTM(
            lstm1_in_size + hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(
            lstm1_in_size + hidden_size, hidden_size, batch_first=True)
        self.mdn = nn.Linear(hidden_size * self.num_layers, mdn_out_size)
        self.tanh = nn.Tanh()

    def forward(self,
                strks,
                strks_m,
                onehots,
                sents_m,
                w_prev,
                k_prev,
                prev1,
                prev2,
                prev3,
                bias=0.):
        # strks shape:     (batch size, timesteps, strk_dim)
        # sents_m shape:   (batch size, sent_len)
        # onehots shape:   (batch size, sent_len, sent_dim)
        timesteps = strks.size()[1]
        sent_len = sents_m.size()[1]

        # print('w_prev shape: ', w_prev.shape)  # torch.Size([50, 1, 60])

        # LSTM 1 ( with attention mechanisim )
        hid1, prev1, win_vec, w_prev, k_prev, phi_prev = self.lstm1(
            strks, onehots, sents_m, w_prev, k_prev, prev1)

        # LSTM 2
        # kip connection, LSTM1's output and window vector
        lstm2_input = torch.cat((strks, hid1, win_vec), dim=2)
        hid2, prev2 = self.lstm2(lstm2_input, prev2)

        # LSTM 3
        lstm3_input = torch.cat((strks, hid2, win_vec), dim=2)
        hid3, prev3 = self.lstm3(lstm3_input, prev3)

        nn.utils.clip_grad_value_(hid2, 10)
        nn.utils.clip_grad_value_(hid3, 10)

        # mixture guassian network
        lstm_output = torch.cat((hid1, hid2, hid3), dim=2)
        params = self.mdn(lstm_output)
        nn.utils.clip_grad_value_(params, 100)
        # print('mdn params shape:', params.shape)

        mdn_params = params.narrow(-1, 0, params.size()[-1] - 1)
        pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = mdn_params.chunk(6, dim=-1)
        eos = torch.sigmoid(params.narrow(-1, params.size()[-1] - 1, 1))
        weights = torch.softmax(pi_hat * (1 + bias), dim=-1)
        rho = self.tanh(rho_hat)
        sigma1, sigma2 = sigma1_hat.exp(), sigma2_hat.exp()

        return (eos, weights, mu1, mu2, sigma1, sigma2, rho), \
               (w_prev, k_prev, prev1, prev2, prev3, phi_prev)
