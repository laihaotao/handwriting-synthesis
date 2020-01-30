import numpy as np
import torch

from utils import plot_stroke, attention_plot
from model import HandwritingPrediction, HandwritingSynthesis

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


def sample_prediction(mix_components, params):
    eos, weights, mu1, mu2, sigma1, sigma2, rho = params

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
                             saved_model='backup/unc_model.pt'):
    np.random.seed(random_state)

    # load model and trained weights
    model = HandwritingPrediction(  # 3 is the stroke feature dim
        args.hidden_size, args.mix_components, 3)
    model.load_state_dict(torch.load(saved_model)['model'])
    optimizer = optim.Adam([{'params': model.parameters()}],
                           lr=args.learning_rate)
    model.to(device)

    # create initial input for the model, each layer
    # need 2 init values (we have 3 layers so 6 values)
    init_states = [torch.zeros((1, args.batch_size, args.hidden_size))] * 6
    init_states = [state.to(device) for state in init_states]
    h1, c1, h2, c2, h3, c3 = init_states
    prev1, prev2, prev3 = (h1, c1), (h2, c2), (h3, c3)
    strk = torch.zeros((1, 1, 3)).to(device)

    record = [np.array([0, 0, 0])]
    for i in range(steps):
        output, prev1, prev2, prev3 = model(strk, prev1, prev2, prev3)
        out = sample_prediction(mix_components, output)
        record.append(out)

        # convert current output as input to the next step
        strk = torch.from_numpy(out).type(torch.FloatTensor).to(device)
        strk = strk.view((1, 1, 3))

    res_strks = np.array(record)
    plot_stroke(res_strks)
    return res_strks


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
                           bias=1.,
                           feature_dim=(3, 60),
                           random_state=700,
                           saved_model='backup/con_model.pt'):
    text = text + ' ' # space here means a line end indicator
    file_path = os.path.join(dir_path, 'data', 'char_to_code.pt')
    char_to_code = torch.load(file_path)
    model = HandwritingSynthesis(
        device, len(text), 1, hidden_size, K, mix_components)
    model.load_state_dict(torch.load(saved_model)['model'])
    model.to(device)

    # prepare init data
    k_prev = torch.zeros(1, K).to(device)
    h1 = c1 = torch.zeros(1, hidden_size)
    h2 = c2 = torch.zeros(1, 1, hidden_size)
    h3 = c3 = torch.zeros(1, 1, hidden_size)
    h1, c1 = h1.to(device), c1.to(device)
    h2, c2 = h2.to(device), c2.to(device)
    h3, c3 = h3.to(device), c3.to(device)

    np.random.seed(random_state)

    # prepare needed data
    # strk, strks_m, sents, sents_m, onehots, w_prev
    strk = torch.zeros((1, 1, 3)).to(device) # (batch, sed_len, feature_dim)
    strk_m = torch.ones((1, 1)).to(device)   # (batch, sed_len)
    sent, sent_m = encode_sentences([text], len(text), char_to_code)
    sent, sent_m = to_torch(sent), to_torch(sent_m)
    onehot = to_one_hot(sent, len(char_to_code) + 1).to(device)
    sent, sent_m = sent.to(device), sent_m.to(device)
    w_prev = onehot.narrow(1, 0, 1)
    prev1, prev2, prev3 = (h1, c1), (h2, c2), (h3, c2)

    stop, count, phis = False, 0, []
    records = [np.zeros(3)]
    while not stop:
        # input data shape:
        #   strokes:   (1, timestep + 1, 3)
        #   masks:     (batch size, timestep)
        #   onehots:   (batch size, len(text line), len(char list))
        #   text_lens: (batch size, 1)
        output1, output2 = model(strk, strk_m, onehot, sent_m, w_prev, k_prev,
                                 prev1, prev2, prev3, bias)
        w_prev, k_prev, prev1, prev2, prev3, phi_prev = output2
        next_point = sample_prediction(mix_components, output1)
        records.append(next_point)
        # print(next_point)

        # convert current output as input to the next step
        strk = torch.from_numpy(next_point).type(torch.FloatTensor).to(device)
        strk = strk.view((1, 1, 3))

        phi_prev = phi_prev.squeeze(0)
        phis.append(phi_prev)
        phi_prev = phi_prev.data.cpu().numpy()

        # 20 is a hack to prevent early exit
        if count >= 20 and np.max(phi_prev) == phi_prev[-1]:
            stop = True
        count += 1

    phis = torch.stack(phis).data.cpu().numpy().T
    res_strks = np.array(records)
    # attention_plot(phis)
    return res_strks, phis
