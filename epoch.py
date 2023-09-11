import os
import pandas as pd
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt

import math
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

kl_anneal_step = 0
best_acc = 0
best_epoch_idx = -1
early_stopping_cnt = 0

def train_epoch(args, epoch_idx, model, dataloader, optimizer, scheduler, loss_fn, writer, device):

    model = model.train()

    global kl_anneal_step
    epoch_acc = 0

    trn_loss = 0.0
    train_total = 0
    train_correct = 0
    for i, data in enumerate(dataloader):
        x, labels = data
        # grad init
        optimizer.zero_grad()
        # forward propagation
        model_output = model(x)
        # calculate acc
        _, predicted = torch.max(model_output.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        # calculate loss
        loss = loss_fn(model_output, labels)
        # back propagation 
        loss.backward()
        # weight update
        optimizer.step()
        
        # trn_loss summary
        trn_loss += loss.item()
        # del (memory issue)
        del loss
        del model_output
        if (i+1) % 100 == 0:
            print("epoch: {}/{} | batch: {} | trn loss: {:.4f} | trn acc: {:.4f}%".
                  format(epoch_idx+1, args.epoch_num, i+1,  trn_loss / i, 100 * train_correct / train_total)) 