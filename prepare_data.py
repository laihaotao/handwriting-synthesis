import random
import io
import numpy as np
import torch
from torch.utils.data import Dataset
import os


def _build_alphabet():
    char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-'
    char_to_code, code_to_char = {}, {}
    for idx, ch in enumerate(char_list):
        char_to_code[ch], code_to_char[idx] = idx, ch
    return char_to_code, code_to_char


def process_and_save(base_path='data/', timestep=800):
    sentences_file = base_path + 'sentences.txt'
    strokes_file   = base_path + 'strokes-py3.npy'
    with io.open(sentences_file, encoding='utf-8') as f:
        sentences = f.readlines()
    strokes = np.load(strokes_file, allow_pickle=True)
    # sentences -> list
    # strokes   -> np.array

    strokes_lens = []
    t_strokes, v_strokes = [], []
    t_sentences, v_sentences = [], []

    # split training and validation set
    len_threshold = timestep + 1
    for idx, stk in enumerate(strokes):
        length = len(stk)
        if length <= len_threshold:
            t_strokes.append(stk)
            t_sentences.append(sentences[idx])
        else:
            v_strokes.append(stk)
            v_sentences.append(sentences[idx])
        strokes_lens.append(length)
    max_stroke_len = np.max(strokes_lens)

    # pad the stroke whose length is less than timstep
    # and create a corresponding mask
    def padding(dataset, seq_len):
        sample_num = len(dataset)
        mask = np.zeros((sample_num, seq_len))
        for i, stroke in enumerate(dataset):
            # exclude the last item x+1 which serve as y_true
            mask[i, 0:len(stroke) - 1] = 1
            dataset[i] = np.vstack([     # (len(stroke) + ?, 3)
                stroke, np.zeros((seq_len + 1 - len(stroke), 3))
            ])
        return np.asarray(dataset), mask

    t_strokes_padded, t_strokes_mask = padding(t_strokes, timestep)
    v_strokes_padded, v_strokes_mask = padding(v_strokes, max_stroke_len)

    char_to_code, code_to_char = _build_alphabet()
    def encode_sentences(lines, max_len):
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

    max_sentence_len = max([len(line) for line in sentences])
    t_sentences_coded, t_sentences_mask = encode_sentences(t_sentences, max_sentence_len)
    v_sentences_coded, v_sentences_mask = encode_sentences(v_sentences, max_sentence_len)

    torch.save(char_to_code, base_path + 'char_to_code.pt')
    torch.save(code_to_char, base_path + 'code_to_char.pt')

    np.save(base_path + 't_strokes', t_strokes_padded)
    np.save(base_path + 'v_strokes', v_strokes_padded)
    np.save(base_path + 't_stroke_mask', t_strokes_mask)
    np.save(base_path + 'v_stroke_mask', v_strokes_mask)
    np.save(base_path + 't_sentences', t_sentences_coded)
    np.save(base_path + 'v_sentences', v_sentences_coded)
    np.save(base_path + 't_sentences_mask', t_sentences_mask)
    np.save(base_path + 'v_sentences_mask', v_sentences_mask)


process_and_save()