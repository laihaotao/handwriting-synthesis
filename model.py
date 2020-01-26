import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class HandwritingPrediction(nn.Module):
    def __init__(self, hidden_size, mix_components, feature_dim):
        super(HandwritingPrediction, self).__init__()
        self.tanh = nn.Tanh()
        self.lstm1 = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, batch_first=True)

        # cancate the output from 1st layer with input x
        lstm2_input_size = feature_dim + hidden_size
        self.lstm2 = nn.LSTM(input_size=lstm2_input_size, hidden_size=hidden_size, batch_first=True)

        # formula (17) from the paper
        # use 20 mixture components, since a point (m, n) have two coordinates so it is a 2D guassian
        # which contains 40 means and 40 standard diviations (20 for m and 20 for n)
        # the total will be 20 weights + 20 correlations + 40 means + 40 std + 1 end of stroke = 121
        mdng_in_size = 2 * hidden_size
        mdng_out_size = 1 + ((1 + 1 + 2 + 2) * mix_components)
        self.linear = nn.Linear(mdng_in_size, mdng_out_size)

    def forward(self, x, prev1, prev2):
        h1, (h1_n, c1_n) = self.lstm1(x, prev1)
        x2 = torch.cat([h1, x], dim=-1)     # skip connection
        h2, (h2_n, c2_n) = self.lstm2(x2, prev2)
        h = torch.cat([h1, h2], dim=-1)     # skip connection

        # mixture of guassian computation, in the paper, fomular(15) ~ (22)
        params = self.linear(h)
        mdn_params = params.narrow(-1, 0, params.size()[-1] - 1)
        pre_weights, mu1, mu2, log_sigma1, log_sigma2, pre_rho = mdn_params.chunk(6, dim=-1)
        end = F.sigmoid(params.narrow(-1, params.size()[-1] - 1, 1))
        weights = F.softmax(pre_weights, dim=-1)  # softmax make sure the sum will be 1
        rho = self.tanh(pre_rho)
        sigma1, sigma2 = torch.exp(log_sigma1), torch.exp(log_sigma2)
        return end, weights, mu1, mu2, sigma1, sigma2, rho, (h1_n, c1_n), (h2_n, c2_n)
