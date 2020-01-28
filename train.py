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
from torch.utils.data import DataLoader

from model import HandwritingPrediction, HandwritingSynthesis
from dataset import HandwritingDataset
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


def forward(model, data, optimizer):
    # strks and its mask: (batch, timestep + 1, strk_dim)
    # sents and its mask: (batch, sent_len)
    # sent onethos shape: (batch, sent_len, sent_dim)
    # where strk_dim = 3, sent_len = 65, sent_dim = 60
    strks, strks_m, sents, sents_m, onehots = data

    # focus window weight on first text char
    w_prev = onehots.narrow(1, 0, 1)

    # e.g. x2 will be the "target" for x1 when compute the loss
    x = strks.narrow(1, 0, args.timesteps)
    strks_m = strks_m.narrow(1, 0, args.timesteps)
    target = strks.narrow(1, 1, args.timesteps)

    if optimizer:
        optimizer.zero_grad()
    output = model(x, strks_m, sents, sents_m, onehots, w_prev)
    loss = log_likelihood(output, target, strks_m)
    return loss




# parameter handling
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='write_prediction')
parser.add_argument('--hidden_size', type=int,default=400)
parser.add_argument('--mix_components', type=int, default=20)
parser.add_argument('--feature_dim', type=int,default=3)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--timesteps', type=int, default=800)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--model_dir', type=str, default='save')
parser.add_argument('--learning_rate', type=float, default=8E-4)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--K', type=int, default=10)
args = parser.parse_args()

# strokes, strokes_mask, sentences, sentences_mask, onehots
t_dataset = HandwritingDataset(is_training=True)
v_dataset = HandwritingDataset(is_training=False)
t_loader = DataLoader(t_dataset, batch_size=args.batch_size,
                      shuffle=True, drop_last=True)
v_loader = DataLoader(v_dataset, batch_size=args.batch_size,
                      shuffle=True, drop_last=True)

train_ds_size = len(t_dataset)
samples_per_batch = train_ds_size // args.batch_size




def train_conditional_model():
    feature_dim = (3, 60)
    model = HandwritingSynthesis(device, args.batch_size, args.hidden_size,
                                 args.K, args.mix_components, feature_dim)
    model.to(device)
    model.float()
    t_loss, v_loss = [], []
    optimizer = optim.Adam([{'params':model.parameters()}],
                           lr=args.learning_rate)


    start_time = time.time()
    for epoch in range(args.num_epochs):
        ct_loss = 0
        # training
        for bat_idx, bat_data in enumerate(t_loader):
            # forward pass
            loss = forward(model, bat_data, optimizer)
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

        # validation
        v_data = v_loader[0]
        tmp_loss = forward(model, v_data, None)
        v_loss.append(tmp_loss)
        print('====> Epoch: {} Average validation loss: {:.4f}'.format(
            epoch + 1, validation_loss))

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
        for bat_idx, (data, masks, onehots, text_lens) in enumerate(t_loader):
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
                    100. * bat_idx / len(t_loader),
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