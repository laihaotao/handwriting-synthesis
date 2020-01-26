import numpy as np
import random
import io
import torch


def get_dataloader(cuda, batch_size):
    # reference: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    train_data = _obtain_data('data/train_strokes.npy', 'data/train_mask.npy',
                             'data/train_onehot.npy',
                             'data/train_text_len.npy', cuda)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
    test_data = _obtain_data('data/test_strokes.npy', 'data/test_mask.npy',
                            'data/test_onehot.npy',
                            'data/test_text_len.npy', cuda)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=True)
    return train_loader, test_loader


def _obtain_data(stroke, mask, onehot, length, cuda):
    data = [np.load(stroke), np.load(mask), np.load(onehot), np.load(length)]
    data_len = len(data[0])

    for i in range(len(data)):
        # convert numpy array to pytorch tensor
        data[i] = torch.from_numpy(data[i]).type(torch.FloatTensor)
        if cuda:
            data[i] = data[i].cuda()

    data = [(data[0][i], data[1][i], data[2][i], data[3][i])
            for i in range(data_len)]
    return data



def run(timestep):
    # load data into memory
    with io.open('./data/sentences.txt', encoding='utf-8') as f:
        texts = f.readlines()
    strokes = np.load('./data/strokes-py3.npy', allow_pickle=True)

    stroke_lens = np.sort(np.array([len(stroke) for stroke in strokes]))
    # min_stroke_len = np.min(stroke_lens)
    max_stroke_len = np.max(stroke_lens)

    # char_seqs = [list(line) for line in texts]
    # char_seqs = np.asarray(char_seqs)

    total_sample_num = len(strokes)
    # train_sample_num = int(0.9 * total_sample_num)
    # test_sample_num = total_sample_num - train_sample_num

    # shuffle the whole dataset split to training and testing subset
    # np.random.seed(1)
    # idx_permute = np.random.permutation(total_sample_num)
    # strokes, char_seqs = strokes[idx_permute], char_seqs[idx_permute]

    train_strokes, train_texts = [], []
    test_strokes, test_texts   = [], []

    for i in range(total_sample_num):
        if len(strokes[i]) <= timestep + 1:
            train_strokes.append(strokes[i])
            train_texts.append(texts[i])
        else:
            test_strokes.append(strokes[i])
            test_texts.append(texts[i])

    # padding and create the mask -> 1: data; 0: padding
    def padding(data, timestep=800):
        max_len = timestep + 1  # add one more entry as the y value for later computation
        mask = np.zeros((len(data), timestep))
        for i in range(len(data)):
            data_len = len(data[i])
            mask[i, 0:data_len - 1] = 1
            data[i] = np.vstack(
                [data[i], np.zeros((max_len - data_len, 3))])
        return data, mask

    train_strokes, train_stroke_mask = padding(train_strokes)
    test_strokes, test_stroke_mask   = padding(test_strokes, max_stroke_len)

    # create one-hot encoding
    char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-'
    char_to_idx, idx_to_char = {}, {}
    idx = 0
    for ch in char_list:
        char_to_idx[ch], idx_to_char[idx] = idx, ch
        idx += 1

    def onehot_encode(data_lines, max_text_len, dictionary):
        res_onehot, res_mask, res_len = [], [], []
        for line in data_lines:
            onehot = np.zeros((max_text_len, len(dictionary) + 1))
            mask = np.zeros(max_text_len)

            mask[:max_text_len] = 1

            for i in range(len(line)):
                ch = line[i]
                try:
                    onehot[i][dictionary[ch]] = 1
                except:
                    onehot[i][-1] = 1

            res_onehot.append(onehot)
            res_mask.append(mask)

        res_onehot = np.stack(res_onehot)
        res_mask = np.stack(res_mask)
        res_len = np.array([[len(line)] for line in data_lines])
        return res_onehot, res_mask, res_len


    max_text_len = max([len(line) for line in texts])
    train_onehot, train_text_mask, train_text_len = onehot_encode(train_texts, max_text_len, char_to_idx)
    test_onehot, test_text_mask, test_text_len = onehot_encode(test_texts, max_text_len, char_to_idx)

    np.save('data/train_strokes', np.stack(train_strokes))
    np.save('data/train_mask', train_stroke_mask)
    np.save('data/test_strokes', np.stack(test_strokes))
    np.save('data/test_mask', test_stroke_mask)

    np.save('data/train_onehot', train_onehot)
    np.save('data/test_onehot', test_onehot)
    np.save('data/train_text_mask', train_text_mask)
    np.save('data/test_text_mask', test_text_mask)
    np.save('data/train_text_len', train_text_len)
    np.save('data/test_text_len', test_text_len)


if __name__ == "__main__":
    timestep = 800
    run(timestep)
