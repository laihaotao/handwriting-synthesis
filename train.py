import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from model import HandwritingPrediction, HandwritingSynthesis
from dataset import HandwritingDataset
from loss import neg_log_likelihood
from utils import save_checkpoint, save_loss_figure
from logger import setup_logger




# most of default values here are took from the paper
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='prediction',
                    help='"prediction" or "synthesis"')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=8E-4,
                    help='learning rate for training')
parser.add_argument('--timesteps', type=int, default=800,
                    help='step in time direction for LSTM')
parser.add_argument('--hidden_size', type=int,default=400,
                    help='number of hidden size for a LSTM cell')
parser.add_argument('--mix_components', type=int, default=20,
                    help='number of mixture distribution')
parser.add_argument('--K', type=int, default=10,
                    help='number of gaussian functions for attention')
parser.add_argument('--model_dir', type=str, default='save',
                    help='location to save the result')
args = parser.parse_args()

# set up a logger for a training
train_id = str(time.time())
logger = setup_logger(train_id, args.model_dir)

logger.info('training args')
logger.info(args)

# check gpu
cuda = torch.cuda.is_available()
logger.info('cuda: {}'.format(cuda))
device = torch.device('cuda' if cuda else 'cpu')

# prepare dataset for training and validation
# strokes, strokes_mask, sentences, sentences_mask, onehots
t_dataset = HandwritingDataset(is_training=True)
v_dataset = HandwritingDataset(is_training=False)
t_loader = DataLoader(t_dataset, batch_size=args.batch_size,
                      shuffle=True, drop_last=True)
v_loader = DataLoader(v_dataset, batch_size=args.batch_size,
                      shuffle=False, drop_last=True)


def train_unconditional_model():
    logger.info('prediction model training ...')
    model = HandwritingPrediction(
        args.hidden_size, args.mix_components, 3) # 3 is the stroke feature dim
    optimizer = optim.Adam([{'params': model.parameters()}],
                           lr=args.learning_rate)
    model.to(device)

    # create initial input for the model, each layer
    # need 2 init values (we have 3 layers so 6 values)
    init_states = [torch.zeros((1, args.batch_size, args.hidden_size))] * 6
    init_states = [state.to(device) for state in init_states]
    h1, c1, h2, c2, h3, c3 = init_states

    t_loss, v_loss = [], []
    start_time = time.time()
    for epoch in range(args.num_epochs):
        ct_loss = 0
        for bat_idx, bat_data in enumerate(t_loader):
            for i in range(len(bat_data)):
                bat_data[i] = bat_data[i].to(device)
            strks, strks_m, sents, sents_m, onehots = bat_data

            # e.g. x2 will be the "target" for x1 when compute the loss
            x = strks.narrow(1, 0, args.timesteps)
            strks_m = strks_m.narrow(1, 0, args.timesteps)
            target = strks.narrow(1, 1, args.timesteps)

            # forward pass
            optimizer.zero_grad()
            output, _ = model(x, (h1, c1), (h2, c2), (h3, c3))
            loss = neg_log_likelihood(output, target, strks_m)
            ct_loss += loss.item()

            # backprogation
            loss.backward()
            optimizer.step()

            if bat_idx % 10 == 0:
                print('Train Epoch #{}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, bat_idx * args.batch_size, len(t_dataset),
                    100. * bat_idx / len(t_loader), loss.item()))

        avg_loss = ct_loss / (len(t_dataset) / args.batch_size)
        t_loss.append(avg_loss)
        print('====> Epoch #{}: Average train loss: {:.4f}'
              .format(epoch + 1, avg_loss))

        # note: disable the validation process, sicne my PC run out of memory
        # # validation
        # v_data = list(enumerate(v_loader))[0][1]
        # for i in range(len(v_data)):
        #     v_data[i] = v_data[i].to(device)
        # strks, strks_m, sents, sents_m, onehots = v_data

        # x = strks.narrow(1, 0, args.timesteps)
        # strks_m = strks_m.narrow(1, 0, args.timesteps)
        # target = strks.narrow(1, 1, args.timesteps)

        # output, _ = model(x, (h1, c1), (h2, c2), (h3, c3))
        # validation_loss = neg_log_likelihood(output, target, strks_m)
        # v_loss.append(validation_loss)
        # print('====> Epoch: {} Average validation loss: {:.4f}'.format(
        #     epoch + 1, validation_loss))

        # save the intermediate result
        validation_loss = 1.0
        filename = args.task + '_epoch_{}.pt'.format(epoch + 1)
        save_checkpoint(epoch, model, validation_loss,
                        optimizer, args.model_dir, filename)
        print('wall time: {}s'.format(time.time() - start_time))

    save_loss_figure(args, t_loss, v_loss)


def train_conditional_model():
    logger.info('synthesis model training ...')
    sent_max_len = t_dataset.sent_len
    model = HandwritingSynthesis(
        device, sent_max_len, args.batch_size,
        args.hidden_size, args.K, args.mix_components)
    optimizer = optim.Adam([{'params':model.parameters()}],
                           lr=args.learning_rate)

    model.to(device)
    k_prev = torch.zeros(args.batch_size, args.K).to(device)
    h1 = c1 = torch.zeros(args.batch_size, args.hidden_size)
    h2 = c2 = torch.zeros(1, args.batch_size, args.hidden_size)
    h3 = c3 = torch.zeros(1, args.batch_size, args.hidden_size)
    h1, c1 = h1.to(device), c1.to(device)
    h2, c2 = h2.to(device), c2.to(device)
    h3, c3 = h3.to(device), c3.to(device)

    t_loss, v_loss = [], []
    start_time = time.time()
    for epoch in range(args.num_epochs):
        ct_loss = 0

        # training loop
        for bat_idx, bat_data in enumerate(t_loader):

            for i in range(len(bat_data)):
                bat_data[i] = bat_data[i].to(device)
            strks, strks_m, sents, sents_m, onehots = bat_data

            # focus window weight on first text char
            w_prev = onehots.narrow(1, 0, 1)

            # e.g. x2 will be the "target" for x1 when compute the loss
            x = strks.narrow(1, 0, args.timesteps)
            strks_m = strks_m.narrow(1, 0, args.timesteps)
            target = strks.narrow(1, 1, args.timesteps)

            optimizer.zero_grad()
            output, _ = model(x, strks_m, onehots, sents_m, w_prev, k_prev,
                           (h1, c1), (h2, c2), (h3, c2))
            loss = neg_log_likelihood(output, target, strks_m)
            ct_loss += loss.item()

            # backprogation
            loss.backward()
            optimizer.step()

            if bat_idx % 10 == 0:
                logger.info('Train Epoch #{}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, bat_idx * args.batch_size, len(t_dataset),
                    100. * bat_idx / len(t_loader), loss.item()))

        avg_loss = ct_loss / (len(t_dataset) / args.batch_size)
        t_loss.append(avg_loss)
        logger.info('====> Epoch #{}: Average train loss: {:.4f}'
              .format(epoch + 1, avg_loss))

        # note: disable the validation process, sicne my PC run out of memory
        # # validation
        # v_data = list(enumerate(v_loader))[0][1]
        # for i in range(len(v_data)):
        #     v_data[i] = v_data[i].to(device)

        # strks, strks_m, sents, sents_m, onehots = v_data

        # w_prev = onehots.narrow(1, 0, 1)
        # x = strks.narrow(1, 0, args.timesteps)
        # strks_m = strks_m.narrow(1, 0, args.timesteps)
        # target = strks.narrow(1, 1, args.timesteps)

        # output, _ = model(x, strks_m, onehots, sents_m, w_prev, k_prev,
        #                 (h1, c1), (h2, c2), (h3, c2))
        # tmp_loss = neg_log_likelihood(output, target, strks_m)
        # v_loss.append(tmp_loss)
        # print('====> Epoch: {} Average validation loss: {:.4f}'.format(
        #     epoch + 1, tmp_loss))

        # save the intermediate result
        tmp_loss = 1.0
        filename = args.task + '_epoch_{}.pt'.format(epoch + 1)
        save_checkpoint(epoch, model, tmp_loss, optimizer,
                        args.model_dir, filename)
        logger.info('wall time: {}s'.format(time.time() - start_time))

    # visualize the change of the loss
    save_loss_figure(args, t_loss, v_loss)


if __name__ == "__main__":
    if args.task == 'prediction':
        train_unconditional_model()
    elif args.task == 'synthesis':
        train_conditional_model()
    else:
        print('no such task!')
