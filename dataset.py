import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HandwritingDataset(Dataset):
    def __init__(self, base_path='data/', is_training=True):
        self.base_path = base_path
        self.char_to_code = torch.load(base_path + 'char_to_code.pt')
        self._load_data(is_training)
        self._to_one_hot(self.sents, len(self.char_to_code) + 1)
        self._summary()

    def __getitem__(self, idx):
        return self.strks[idx], self.strks_m[idx], \
               self.sents[idx], self.sents_m[idx], \
               self.onehots[idx]

    def __len__(self):
        return self.len

    def _to_one_hot(self, data, alphabet_len):
        onehots = []
        for line in data:
            oh = np.zeros((line.shape[0], alphabet_len))
            oh[np.arange(line.shape[0]), line.int()] = 1
            onehots.append(oh)
        self.onehots = self._to_torch(np.asarray(onehots))

    def _load_data(self, is_t):
        if is_t:
            self.strks   = self._to_torch(np.load(self.base_path + 't_strokes.npy'))
            self.sents   = self._to_torch(np.load(self.base_path + 't_sentences.npy'))
            self.strks_m = self._to_torch(np.load(self.base_path + 't_stroke_mask.npy'))
            self.sents_m = self._to_torch(np.load(self.base_path + 't_sentences_mask.npy'))
        else:
            self.strks   = self._to_torch(np.load(self.base_path + 'v_strokes.npy'))
            self.sents   = self._to_torch(np.load(self.base_path + 'v_sentences.npy'))
            self.strks_m = self._to_torch(np.load(self.base_path + 'v_stroke_mask.npy'))
            self.sents_m = self._to_torch(np.load(self.base_path + 'v_sentences_mask.npy'))

        self.len = self.strks.shape[0]
        self.strk_dim = self.strks.shape[2]
        self.sent_len = self.sents.shape[1]

    def _to_torch(self, np_data, dtype=torch.FloatTensor):
        return torch.from_numpy(np_data).type(dtype)

    def _summary(self):
        n  = 56
        print('-' * n)
        print('| Dataset Info')
        print('-' * n)
        print('| dataset length:        ', self.len)
        print('| alphabet length:       ', len(self.char_to_code))
        print('| strokes shape:         ', self.strks.shape)
        print('| strokes mask shape:    ', self.strks_m.shape)
        print('| sentences shape:       ', self.sents.shape)
        print('| sentences mask shape:  ', self.sents_m.shape)
        print('| sents one_hot shape:   ', self.onehots.shape)
        print('-' * n)


# t_dataset = HandwritingDataset(is_training=True)
# loader = DataLoader(t_dataset, batch_size=1, shuffle=False, drop_last=True)
# v_dataset = HandwritingDataset(is_training=False)
# print('length of the training set: ', len(t_dataset))
# print('length of the validation set: ', len(v_dataset))

# t_dataset.to_one_hot(t_dataset.sents[:3], 60)