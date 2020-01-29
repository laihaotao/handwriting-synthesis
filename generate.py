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
    out = np.insert(prediction_point, 0, sample_eos)
    return out

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


def to_one_hot(tensor_data, alphabet_len):
    onehots = []
    for line in tensor_data:
        oh = np.zeros((line.shape[0], alphabet_len))
        oh[np.arange(line.shape[0]), line.int()] = 1
        onehots.append(oh)
    return torch.from_numpy(np.asarray(onehots)).type(torch.FloatTensor)


def encode_sentences(lines, max_len, char_to_code):
    sen_len, unknow_code = len(lines), len(char_to_code)
    sentences_coded = np.zeros((sen_len, max_len))
    mask = np.zeros((sen_len, max_len))

    for i, line in enumerate(lines):
        mask[i][0:len(line)] = 1
        for j, ch in enumerate(line):
            if ch in char_to_code:
                sentences_coded[i][j] = char_to_code[ch]
            else:
                sentences_coded[i][j] = unknow_code

    return sentences_coded, mask


def to_torch(np_data, dtype=torch.FloatTensor):
    return torch.from_numpy(np_data).type(dtype)


def generate_conditionally(text,
                           hidden_size=400,
                           mix_components=20,
                           K=10,
                           bias1=1.,
                           bias2=1.,
                           feature_dim=(3, 60),
                           random_state=700,
                           saved_model='con_model.pt'):
    text = text + ' '
    char_to_code = torch.load('data/char_to_code.pt')
    model = HandwritingSynthesis(device, len(text), 1, hidden_size,
                                 K, mix_components)
    model.load_state_dict(torch.load(saved_model)['model'])
    model.to(device)


    k_prev = torch.zeros(1, K).to(device)
    h1 = c1 = torch.zeros(1, hidden_size)
    h2 = c2 = torch.zeros(1, 1, hidden_size)
    h3 = c3 = torch.zeros(1, 1, hidden_size)
    h1, c1 = h1.to(device), c1.to(device)
    h2, c2 = h2.to(device), c2.to(device)
    h3, c3 = h3.to(device), c3.to(device)

    np.random.seed(random_state)
    timesteps = 400

    # prepare needed data
    # strk, strks_m, sents, sents_m, onehots, w_prev
    strk = to_torch(np.zeros((1, 1, 3))).to(device) # (batch, sed_len, feature_dim)
    strk_m = to_torch(np.ones((1, 1))).to(device)   # (batch, sed_len)
    sent, sent_m = encode_sentences([text], len(text), char_to_code)
    sent, sent_m = to_torch(sent), to_torch(sent_m)
    onehot = to_one_hot(sent, len(char_to_code) + 1).to(device)
    sent, sent_m = sent.to(device), sent_m.to(device)
    w_prev = onehot.narrow(1, 0, 1)
    prev1, prev2, prev3 = (h1, c1), (h2, c2),(h3, c2)

    print('strk shape:', strk.shape)
    print('strk_m shape:', strk_m.shape)
    print('sent shape:', sent.shape)
    print('sent_m shape:', sent_m.shape)
    print('onehot shape:', onehot.shape)

    stop = False
    count = 0
    phis = []
    records = [np.zeros(3)]
    # for _ in range(timesteps):
    while not stop:
        # print(stroke.shape)
        # print(onehots.shape)
        # print(text_len.shape)
        # print(w_prev.shape)

        # input data shape:
        #   strokes:   (1, timestep + 1, 3)
        #   masks:     (batch size, timestep)
        #   onehots:   (batch size, len(text line), len(char list))
        #   text_lens: (batch size, 1)
        output1, output2 = model(strk, strk_m, onehot, sent_m, w_prev, k_prev,
                                 prev1, prev2, prev3)
        eos, weights, mu1, mu2, sigma1, sigma2, rho = output1
        w_prev, k_prev, prev1, prev2, prev3, phi_prev = output2
        next_point = sample_prediction(mix_components, eos, weights, mu1, mu2,
                                       sigma1, sigma2, rho)
        records.append(next_point)
        # print(next_point)

        # convert current output as input to the next step
        strk = torch.from_numpy(next_point).type(torch.FloatTensor).to(device)
        strk = strk.view((1, 1, 3))


        phi_prev = phi_prev.squeeze(0)

        phis.append(phi_prev)
        phi_prev = phi_prev.data.cpu().numpy()

        # hack to prevent early exit (attention is unstable at the beginning)
        if count >= 20 and np.max(phi_prev) == phi_prev[-1]:
            stop = True
        count += 1

    phis = torch.stack(phis).data.cpu().numpy().T
    plot_stroke(np.array(records))
    attention_plot(phis)


def attention_plot(phis):
    plt.rcParams["figure.figsize"] = (12, 6)
    phis = phis / (np.sum(phis, axis=0, keepdims=True))
    plt.xlabel('handwriting generation')
    plt.ylabel('text scanning')
    plt.imshow(phis, cmap='hot', interpolation='nearest', aspect='auto')
    plt.show()


# generate_conditionally('hello world')
generate_conditionally('my name is haotao')
# generate_unconditionally()
