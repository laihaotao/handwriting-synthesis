import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from model import HandwritingPrediction, HandwritingSynthesis
from dataset import get_dataloader
from loss import log_likelihood
from utils import save_checkpoint

# check gpu
cuda = torch.cuda.is_available()
print('cuda: {}'.format(cuda))
device = torch.device('cuda' if cuda else 'cpu')


def save_loss_figure(t_loss, v_loss):
    f1 = plt.figure(1)
    plt.plot(range(1, args.num_epochs + 1),
             t_loss,
             color='blue',
             linestyle='solid')
    plt.plot(range(1, args.num_epochs + 1),
             v_loss,
             color='red',
             linestyle='solid')
    f1.savefig(args.task + "_loss_curves", bbox_inches='tight')

# parameter handling
parser = argparse.ArgumentParser()
parser.add_argument('--task',
                    type=str,
                    default='write_prediction',
                    help='"write_prediction" or "synthesis"')
parser.add_argument('--hidden_size',
                    type=int,
                    default=400,
                    help='size of LSTM hidden state')
parser.add_argument('--mix_components',
                    type=int,
                    default=20,
                    help='number of gaussian mixture components')
parser.add_argument('--feature_dim',
                    type=int,
                    default=3,
                    help='feature dimension for each timstep')
parser.add_argument('--batch_size',
                    type=int,
                    default=50,
                    help='minibatch size')
parser.add_argument('--timesteps',
                    type=int,
                    default=800,
                    help='LSTM sequence length')
parser.add_argument('--num_epochs',
                    type=int,
                    default=50,
                    help='number of epochs')
parser.add_argument('--model_dir',
                    type=str,
                    default='save',
                    help='directory to save model to')
parser.add_argument('--learning_rate',
                    type=float,
                    default=8E-4,
                    help='learning rate')
parser.add_argument('--decay_rate',
                    type=float,
                    default=0.99,
                    help='lr decay rate for adam optimizer per epoch')
parser.add_argument('--K',
                    type=int,
                    default=10,
                    help='number of attention clusters on text input')
args = parser.parse_args()

# strokes, mask, onehot, text_len
train_loader, test_loader = get_dataloader(cuda, args.batch_size)
train_ds_size = len(train_loader.dataset)
test_ds_size  = len(test_loader.dataset)
samples_per_batch = train_ds_size // args.batch_size

def train_conditional_model():
    # number of char in the diactionary
    feature_dim = (train_loader.dataset[0][0].size()[1],
                   train_loader.dataset[0][2].size()[1])

    model = HandwritingSynthesis(device, args.batch_size, args.hidden_size,
                                 args.K, args.mix_components, feature_dim)
    model.to(device)

    t_loss, v_loss = [], []
    best_validation_loss = 1E10
    optimizer = optim.Adam([{'params':model.parameters()}],
                           lr=args.learning_rate)

    # training
    start_time = time.time()
    for epoch in range(args.num_epochs):
        train_loss = 0
        for bat_idx, bat_data in enumerate(train_loader):
            # input data shape:
            #   strokes:   (batch size, timestep + 1, 3)
            #   masks:     (batch size, timestep)
            #   onehots:   (batch size, len(text line), len(char list))
            #   text_lens: (batch size, 1)
            strokes, masks, onehots, text_lens = bat_data

            # focus window weight on first text char
            w_prev = onehots.narrow(1, 0, 1).squeeze()

            # gather training batch
            x = strokes.narrow(1, 0, args.timesteps)
            masks = masks.narrow(1, 0, args.timesteps)

            # feed forward
            optimizer.zero_grad()
            outputs = model(x.to(device), onehots.to(device), masks.to(device),
                            text_lens.to(device), w_prev.to(device))
            eos, weights, mu1, mu2, sigma1, sigma2, rho = outputs
            y = strokes.narrow(1, 1, args.timesteps)
            loss = log_likelihood(eos, weights, mu1, mu2,
                                  sigma1, sigma2, rho, y, masks)
            train_loss += loss.item()

            # back propagation
            loss.backward()
            optimizer.step()

            if bat_idx % 10 == 0:
                print('Train Epoch #{}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, bat_idx * args.batch_size, train_ds_size,
                    100. * bat_idx / len(train_loader), loss.item()))

        avg_loss = train_loss / samples_per_batch
        print('====> Epoch #{}: Average train loss: {:.4f}'
              .format(epoch + 1, avg_loss))
        t_loss.append(avg_loss)

        # validation
        (test_data, masks, onehots, text_lens) = list(enumerate(test_loader))[0][1]
        x = test_data.narrow(1, 0, args.timesteps)
        y = test_data.narrow(1, 1, args.timesteps)
        masks = masks.narrow(1, 0, args.timesteps)
        w_prev = onehots.narrow(1, 0, 1).squeeze()

        outputs = model(x, onehots, masks, text_lens, w_prev)
        eos, weights, mu1, mu2, sigma1, sigma2, rho = outputs
        loss = log_likelihood(eos, weights, mu1, mu2,
                              sigma1, sigma2, rho, y, masks)
        validation_loss = loss.item()
        print('====> Epoch: {} Average validation loss: {:.4f}'
              .format(epoch + 1, validation_loss))
        v_loss.append(validation_loss)

        # checkpoint model and training
        filename = args.task + '_epoch_{}.pt'.format(epoch + 1)
        save_checkpoint(epoch, model, validation_loss, optimizer,
                        args.model_dir, filename)

        print('wall time: {}s'.format(time.time() - start_time))

    # visualize the change of the loss
    save_loss_figure(t_loss, v_loss)




def train_unconditional_model():
    model = HandwritingPrediction(args.hidden_size, args.mix_components,
                                  args.feature_dim)
    init_states = [torch.zeros((1, args.batch_size, args.hidden_size))] * 4
    if cuda:
        model = model.cuda()
        init_states = [state.cuda() for state in init_states]

    init_states = [Variable(state, requires_grad=False) for state in init_states]
    h1_init, c1_init, h2_init, c2_init = init_states
    t_loss, v_loss = [], []
    best_validation_loss = 1E10
    optimizer = optim.Adam([{'params': model.parameters()}], lr=args.learning_rate)

    # update training time and start training
    start_time = time.time()
    for epoch in range(args.num_epochs):
        train_loss = 0
        for bat_idx, (data, masks, onehots, text_lens) in enumerate(train_loader):
            # data (batch_size, timestep + 1, feature_dim)

            # gather training batch
            x = data.narrow(1, 0, args.timesteps)
            masks = masks.narrow(1, 0, args.timesteps)

            # forward pass
            optimizer.zero_grad()
            outputs = model(x, (h1_init, c1_init), (h2_init, c2_init))
            end, weights, mu1, mu2, sigma1, sigma2, rho, prev, prev2 = outputs

            # loss computation
            y = data.narrow(1, 1, args.timesteps)
            loss = log_likelihood(end, weights, mu1, mu2, sigma1, sigma2, rho, y, masks)
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()

            if bat_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, bat_idx * len(data), train_ds_size,
                    100. * bat_idx / len(train_loader),
                    loss.item()))

        # update training performance
        print('====> Epoch: {} Average train loss: {:.4f}'
            .format(epoch + 1, train_loss / (train_ds_size // args.batch_size)))
        t_loss.append(train_loss / (train_ds_size // args.batch_size))

        # validation
        (test_data, masks, onehots, text_lens) = list(enumerate(test_loader))[0][1]
        x = test_data.narrow(1, 0, args.timesteps)
        y = test_data.narrow(1, 1, args.timesteps)
        masks = masks.narrow(1, 0, args.timesteps)

        outputs = model(y, (h1_init, c1_init), (h2_init, c2_init))
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, prev, prev2 = outputs
        loss = log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2,
                            rho, y, masks)
        validation_loss = loss.item()
        print('====> Epoch: {} Average validation loss: {:.4f}'.format(\
            epoch+1, validation_loss))
        v_loss.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(epoch, model, validation_loss, optimizer,
                            args.model_dir, args.task + '_best.pt')

        # checkpoint model and training
        filename = args.task + '_epoch_{}.pt'.format(epoch + 1)
        save_checkpoint(epoch, model, validation_loss,
                        optimizer, args.model_dir, filename)

        print('wall time: {}s'.format(time.time() - start_time))

    save_loss_figure(t_loss, v_loss)


train_conditional_model()
# train_unconditional_model()