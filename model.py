import torch
import torch.nn as nn
from torch.autograd import Variable

cuda = torch.cuda.is_available()


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

class HandwritingSynthesis(nn.Module):
    def __init__(self,
                 device,
                 batch_size,
                 hidden_size,
                 gaussian_funct_num,
                 mix_components,
                 feature_dim,
                 vars_per_funct=3):
        super(HandwritingSynthesis, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.K = gaussian_funct_num
        self.vars_per_funct = vars_per_funct
        self.h_init, self.c_init = [], []
        self.k_prev = None

        # stroke_dim = 3; text_dim = 60
        self.stroke_dim, self.text_dim = feature_dim

        self.input_transform = nn.Linear(self.stroke_dim, hidden_size)
        self.skip_input_transform1 = nn.Linear(self.stroke_dim, hidden_size)
        self.skip_input_transform2 = nn.Linear(self.stroke_dim, hidden_size)
        self.skip_output_trainsform1 = nn.Linear(hidden_size, hidden_size)
        self.skip_output_trainsform2 = nn.Linear(hidden_size, hidden_size)

        self.w_transform1 = nn.Linear(self.text_dim, hidden_size)
        self.w_transform2 = nn.Linear(self.text_dim, hidden_size)
        self.w_transform3 = nn.Linear(self.text_dim, hidden_size)

        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.tanh = nn.Tanh()

        win_out_size = self.K * vars_per_funct  # 3K
        self.softwindow = nn.Linear(hidden_size, win_out_size)

        mdn_out_size = 1 + ((1 + 1 + 2 + 2) * mix_components)
        self.mdn = nn.Linear(hidden_size, mdn_out_size)

        self.init_hidden()

    def forward(self, strokes, onehots, masks, text_lens, w_prev):
        # strokes shape:  (batch size, timesteps, features)
        # onehots shape:  (batch size, len(line), len(alphabet))
        timesteps = strokes.size()[1]
        ''' working
        # first LSTM layer
        lstm_input = self.input_transform(strokes) + \
            self.w_transform1(w_prev).unsqueeze(dim=1)

        lstm1_output, hidden1 = self.lstm1(lstm_input, self.hidden1)
        self.hidden1 = (hidden1[0].detach(), hidden1[1].detach())

        # attention mechanism
        p = self.softwindow(lstm1_output)  # (batch size, timestep, 3K)
        p = p.permute(1, 0, 2)             # (timestep, batch size, 3K)

        a, b, k = torch.chunk(p, self.vars_per_funct, dim=-1)
        a, b, k = a.exp(), b.exp(), k.exp()
        k = torch.cumsum(k, dim=-1)

        U = onehot.size()[1]
        phi_t_u_list = []
        for u in range(U):
            phi = torch.sum(a * torch.exp(-1 * b * (k - u)**2), dim=-1)
            phi_t_u_list.append(phi)
            # todo:


        phi_t_u = torch.stack(phi_t_u_list, dim=-1).permute(1, 0, 2)
        w = torch.matmul(phi_t_u, onehot) # (batch size, timesteps, U)
        self.attention_map = phi_t_u # (batch size, timesteps, len(alphabet))
        # print('phi_t_u shape: {}'.format(phi_t_u.shape))
        # print('onehot shape: {}'.format(onehot.shape))
        # exit(0)
        '''

        U = onehots.size()[1]
        hid_1, w_list = [], []
        for time in range(timesteps):
            stroke_at_t_time = strokes.narrow(1, time, 1)
            mask_at_t_time = masks.narrow(1, time, 1)

            # first LSTM layer
            input_at_t_time = self.input_transform(stroke_at_t_time) + \
                self.w_transform1(w_prev).unsqueeze(dim=1)
            lstm1_output, hidden1 = self.lstm1(input_at_t_time, self.hidden1)
            hid_1.append(lstm1_output)
            self.hidden1 = (hidden1[0].detach(), hidden1[1].detach())

            # attention mechanism
            p = self.softwindow(lstm1_output)  # (batch size, timestep, 3K)
            p = p.permute(0, 2, 1)             # (batch size, 3K, 1)

            a, b, k = torch.chunk(p, self.vars_per_funct, dim=1)
            a, b, k = a.exp(), b.exp(), k.exp()
            k = k + self.k_prev

            u = torch.arange(U, dtype=torch.float32, device=self.device)
            phi = torch.sum(a * torch.exp(-1 * b * (k - u)**2), dim=1)
            # if phi[0, -1] > torch.max(phi[0, :-1]):
            #     self.EOS = True
            phi = (phi * mask_at_t_time).unsqueeze(2)
            wt = torch.sum(phi * onehots, dim=1, keepdim=True)
            w_list.append(wt)

            self.k_prev = k.detach()
            w_prev = wt.squeeze().detach()


        # (batch size, timestep, len(alphabet))
        w = torch.stack(w_list, dim=-1).squeeze().permute(0, 2, 1)

        # second LSTM layer
        lstm2_input = lstm1_output + \
            self.skip_input_transform1(strokes) + self.w_transform2(w)
        lstm2_output, hidden2 = self.lstm2(lstm2_input, self.hidden2)
        self.hidden2 = (hidden2[0].detach(), hidden2[1].detach())

        # third LSTM layer
        lstm3_input = lstm2_output + \
            self.skip_input_transform2(strokes) + self.w_transform3(w)
        lstm3_output, hidden3 = self.lstm3(lstm3_input, self.hidden3)
        self.hidden3 = (hidden3[0].detach(), hidden3[1].detach())

        # final output from LSTM
        lstm_output = lstm3_output + \
             self.skip_output_trainsform2(lstm2_output) + \
             self.skip_output_trainsform1(lstm1_output)
        nn.utils.clip_grad_value_(lstm1_output, 10)
        nn.utils.clip_grad_value_(lstm2_output, 10)
        nn.utils.clip_grad_value_(lstm3_output, 10)
        nn.utils.clip_grad_value_(lstm_output, 10)

        # feed into mixture density network
        params = self.mdn(lstm_output)
        nn.utils.clip_grad_value_(params, 100)

        mdn_params = params.narrow(-1, 0, params.size()[-1] - 1)
        pre_weights, mu1, mu2, \
            log_sigma1, log_sigma2, \
                pre_rho = mdn_params.chunk(6, dim=-1)
        end = torch.sigmoid(params.narrow(-1, params.size()[-1] - 1, 1))
        weights = torch.softmax(pre_weights, dim=-1)
        rho = self.tanh(pre_rho)
        sigma1, sigma2 = torch.exp(log_sigma1), torch.exp(log_sigma2)

        return end, weights, mu1, mu2, sigma1, sigma2, rho


    def init_hidden(self):
        # (batch size, seq, feature)
        h1, c1 = (torch.zeros(1, self.batch_size, self.hidden_size),
                  torch.zeros(1, self.batch_size, self.hidden_size))
        h2, c2 = (torch.zeros(1, self.batch_size, self.hidden_size),
                  torch.zeros(1, self.batch_size, self.hidden_size))
        h3, c3 = (torch.zeros(1, self.batch_size, self.hidden_size),
                  torch.zeros(1, self.batch_size, self.hidden_size))

        self.hidden1 = (h1.to(self.device), c1.to(self.device))
        self.hidden2 = (h2.to(self.device), c2.to(self.device))
        self.hidden3 = (h3.to(self.device), c3.to(self.device))
        self.k_prev = torch.zeros(self.batch_size, self.K, 1).to(self.device)

        # print('hidden shape: {}'.format(self.hidden1[0].shape))
