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

from model import HandwritingPrediction
from dataset import get_dataloader
from loss import log_likelihood
from utils import save_checkpoint

# check gpu
cuda = torch.cuda.is_available()
print('cuda: {}'.format(cuda))

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


train_loader, test_loader = get_dataloader(cuda, args.batch_size)

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
    for batch_idx, (data, masks, onehots, text_lens) in enumerate(train_loader):
        # data (batch_size, timestep + 1, feature_dim)

        # gather training batch
        step_back = data.narrow(1, 0, args.timesteps)
        x = Variable(step_back, requires_grad=False)
        masks = Variable(masks, requires_grad=False)
        masks = masks.narrow(1, 0, args.timesteps)

        # forward pass
        optimizer.zero_grad()
        outputs = model(x, (h1_init, c1_init), (h2_init, c2_init))
        end, weights, mu1, mu2, sigma1, sigma2, rho, prev, prev2 = outputs

        # loss computation
        tmp_y = data.narrow(1, 1, args.timesteps)
        y = Variable(tmp_y, requires_grad=False)
        loss = log_likelihood(end, weights, mu1, mu2, sigma1, sigma2, rho, y, masks)
        loss.backward()
        train_loss += loss.item()

        torch.nn.utils.clip_grad_norm(model.parameters(), 10)
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    # update training performance
    print('====> Epoch: {} Average train loss: {:.4f}'
         .format(epoch + 1, train_loss / (len(train_loader.dataset) // args.batch_size)))
    t_loss.append(train_loss / (len(train_loader.dataset) // args.batch_size))

    # validation
    # prepare validation sample data
    (validation_samples, masks, onehots, text_lens) = list(enumerate(test_loader))[0][1]
    step_back2 = validation_samples.narrow(1, 0, args.timesteps)
    masks = Variable(masks, requires_grad=False)
    masks = masks.narrow(1, 0, args.timesteps)

    x = Variable(step_back2, requires_grad=False)

    validation_samples = validation_samples.narrow(1, 1, args.timesteps)
    y = Variable(validation_samples, requires_grad=False)

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

    # # learning rate annealing
    # if (epoch+1)%10 == 0:
    #     optimizer = decay_learning_rate(optimizer)

    # checkpoint model and training
    filename = args.task + '_epoch_{}.pt'.format(epoch + 1)
    save_checkpoint(epoch, model, validation_loss,
                    optimizer, args.model_dir, filename)

    print('wall time: {}s'.format(time.time() - start_time))

f1 = plt.figure(1)
plt.plot(range(1, args.num_epochs+1), t_loss, color='blue', linestyle='solid')
plt.plot(range(1, args.num_epochs+1), v_loss, color='red', linestyle='solid')
f1.savefig(args.task +"_loss_curves", bbox_inches='tight')
