import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from utils import plot_stroke
from model import HandwritingPrediction


cuda = torch.cuda.is_available()

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
        eos, weights, mu1, mu2, sigma1, sigma2, rho, prev, prev2 = model(
            x, prev, prev2)

        # batch_size = 1, timestep = 1
        # the meaningful values can be accessed via data.[][][x]

        prob_eos = eos.data[0][0][0]
        sample_eos = np.random.binomial(1, prob_eos.cpu())
        sample_idx = np.random.choice(range(mix_components),
                                      p=weights.data[0][0].cpu().numpy())

        # sample new stroke point
        means = np.array([mu1.data[0][0][sample_idx].cpu(),
                          mu2.data[0][0][sample_idx].cpu()])

        sigma1 = sigma1.data[0][0][sample_idx].cpu()
        sigma2 = sigma2.data[0][0][sample_idx].cpu()
        std1_square, std2_square = sigma1 ** 2, sigma2 ** 2

        # covariance = rho * sigma1 * sigma2
        # for a covariance matrix:
        #   cij means the covariance betwenn element i and element j
        covariance = rho.data[0][0][sample_idx].cpu() * sigma1 * sigma2
        covariance_matrix = np.array([[std1_square, covariance],
                                      [covariance, std2_square]])
        prediction_point = np.random.multivariate_normal(means, covariance_matrix) # point (p1, p2)

        # remember the prediction of current step (eos, p1, p2)
        out = np.array([sample_eos, prediction_point[0], prediction_point[1]])
        record.append(out)

        # convert current output as input to the next step
        x = torch.from_numpy(out).type(torch.FloatTensor)
        if cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=False)
        x = x.view((1, 1, 3))

    plot_stroke(np.array(record))


generate_unconditionally()
