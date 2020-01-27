import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from utils import plot_stroke
from model import HandwritingPrediction, HandwritingSynthesis


# check gpu
cuda = torch.cuda.is_available()
print('cuda: {}'.format(cuda))
device = torch.device('cuda' if cuda else 'cpu')



def sample_prediction(mix_components, eos, weights, mu1, mu2, sigma1, sigma2, rho):
    # batch_size = 1, timestep = 1
    # the meaningful values can be accessed via data.[][][x]
    prob_eos = eos.data[0][0][0]
    sample_eos = np.random.binomial(1, prob_eos.cpu())
    sample_idx = np.random.choice(range(mix_components),
                                  p=weights.data[0][0].cpu().numpy())

    # sample new stroke point
    means = np.array(
        [mu1.data[0][0][sample_idx].cpu(), mu2.data[0][0][sample_idx].cpu()])

    sigma1 = sigma1.data[0][0][sample_idx].cpu()
    sigma2 = sigma2.data[0][0][sample_idx].cpu()
    std1_square, std2_square = sigma1**2, sigma2**2

    # covariance = rho * sigma1 * sigma2
    # for a covariance matrix:
    #   cij means the covariance betwenn element i and element j
    covariance = rho.data[0][0][sample_idx].cpu() * sigma1 * sigma2
    covariance_matrix = np.array([[std1_square, covariance],
                                  [covariance, std2_square]])
    prediction_point = np.random.multivariate_normal(
        means, covariance_matrix)  # point (p1, p2)

    # remember the prediction of current step (eos, p1, p2)
    out = np.array([sample_eos, prediction_point[0], prediction_point[1]])
    return out

'''
def sample_y(m, e, pi, mu1, mu2, sigma1, sigma2, rho):
    r1 = np.random.rand()
    cumulative_weight = 0
    y = torch.zeros([1, 3]).to(device)

    # Sample from GMM
    for i in range(m):
        cumulative_weight += pi[0, i]
        if cumulative_weight > r1:
            mu1_i = mu1[0, i]
            mu2_i = mu2[0, i]
            sigma1_i = sigma1[0, i]
            sigma2_i = sigma2[0, i]
            rho_i = rho[0, i]
            mvn_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.Tensor([mu1_i, mu2_i]),
                covariance_matrix=torch.Tensor(
                    [[sigma1_i**2, rho_i * sigma1_i * sigma2_i],
                     [rho_i * sigma1_i * sigma2_i, sigma2_i**2]]))
            y[0, 1:] = mvn_distribution.sample()
            break
    r2 = np.random.rand()
    if e > r2:
        y[0, 0] = 1
    else:
        y[0, 0] = 0

    return y

'''
def generate_unconditionally(hidden_size=400,
                             mix_components=20,
                             steps=700,
                             feature_dim=3,
                             random_state=700,
                             saved_model='unc_model.pt'):

    # load model and trained weights
    model = HandwritingPrediction(hidden_size, mix_components, feature_dim)
    model.load_state_dict(torch.load(saved_model)['model'])

    np.random.seed(random_state)
    zero_tensor = torch.zeros((1, 1, 3))
    init_states = [torch.zeros((1, 1, hidden_size))] * 4
    if cuda:
        model = model.cuda()
        zero_tensor = zero_tensor.cuda()
        init_states = [state.cuda() for state in init_states]
    x = Variable(zero_tensor)
    init_states = [Variable(state, requires_grad=False) for state in init_states]
    h1_init, c1_init, h2_init, c2_init = init_states
    prev = (h1_init, c1_init)
    prev2 = (h2_init, c2_init)

    record = [np.array([0, 0, 0])]

    for i in range(steps):
        eos, weights, mu1, mu2, sigma1, sigma2, \
            rho, prev, prev2 = model(x, prev, prev2)

        out = sample_prediction(mix_components, eos, weights,
                                mu1, mu2, sigma1, sigma2, rho)
        record.append(out)

        # convert current output as input to the next step
        x = torch.from_numpy(out).type(torch.FloatTensor)
        if cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=False)
        x = x.view((1, 1, 3))

    plot_stroke(np.array(record))



def text_to_onehot(text, char_to_code):
    onehot = np.zeros((len(text), len(char_to_code) + 1))
    for i in range(len(text)):
        ch = text[i]
        try:
            onehot[i][char_to_code[ch]] = 1
        except:
            onehot[i][-1] = 1
    return torch.from_numpy(onehot).type(torch.FloatTensor)


def generate_conditionally(text,
                           hidden_size=400,
                           mix_components=20,
                           K=10,
                           bias1=1.,
                           bias2=1.,
                           feature_dim=(3, 60),
                           random_state=700,
                           saved_model='con_model.pt'):

    char_to_code = torch.load('data/char_to_code.pt')
    model = HandwritingSynthesis(device, 1, hidden_size, K,
                                 mix_components, feature_dim)
    model.load_state_dict(torch.load(saved_model)['model'])
    model.to(device)

    np.random.seed(random_state)
    timesteps = 600

    text_len = torch.from_numpy(
        np.array([[len(text)]])).type(torch.FloatTensor
    ).to(device)
    onehots = text_to_onehot(text, char_to_code)
    onehots = onehots.unsqueeze(0).to(device)
    stroke = torch.zeros((1, 1, 3)).to(device)
    w_prev = onehots.narrow(1, 0, 1).squeeze()
    w_prev = w_prev.unsqueeze(dim=0).to(device)

    # stop, count, phis, record = False, 0, [], [np.zeros(3)]
    record = [np.array([0, 0, 0])]
    attention_maps = []
    for i in range(timesteps):
        # print(stroke.shape)
        # print(onehots.shape)
        # print(text_len.shape)
        # print(w_prev.shape)

        # input data shape:
        #   strokes:   (1, timestep + 1, 3)
        #   masks:     (batch size, timestep)
        #   onehots:   (batch size, len(text line), len(char list))
        #   text_lens: (batch size, 1)

        eos, weights, mu1, mu2, \
            sigma1, sigma2, rho = model(stroke, onehots, text_len, w_prev)

        out = sample_prediction(mix_components, eos, weights, mu1, mu2,
                                sigma1, sigma2, rho)
        record.append(out)

        # convert current output as input to the next step
        stroke = torch.from_numpy(out).type(torch.FloatTensor).to(device)
        stroke = stroke.view((1, 1, 3))

    plot_stroke(np.array(record))


generate_conditionally('Hellow world, nihao')
# generate_unconditionally()
