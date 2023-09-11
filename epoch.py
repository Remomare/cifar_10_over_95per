import os
import pandas as pd
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt

import torch

from utils import plot_grad_flow

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
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'TRAIN EPOCH {epoch_idx}/{args.epoch}')):
        x, labels = batch
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

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # trn_loss summary
        trn_loss += loss.item()
        batch_acc = 100 * train_correct / train_total
        # del (memory issue)
        del loss
        del model_output

        if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
                tqdm.write(f'TRAIN: {batch_idx}/{len(dataloader)} - Loss={loss.item()}')
        if args.save_gradient_flow:
            plot_grad_flow(model.named_parameters())
        if args.use_tensorboard_logging:
            total_idx = batch_idx + (epoch_idx * len(dataloader))
            writer.add_scalar('TRAIN/Loss', loss.item() / args.batch_size, total_idx)
            writer.add_scalar('TRAIN/Batch_Accuracy', batch_acc.item(), batch_idx+(epoch_idx*len(dataloader)))
        torch.save(model.state_dict(), args.save_modle_path) 

def test_epoch(args, epoch_idx, model, dataloader, optimizer, scheduler, loss_fn, writer, device):

    model = model.train()

    global kl_anneal_step
    epoch_acc = 0

    trn_loss = 0.0
    train_total = 0
    train_correct = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'TRAIN EPOCH {epoch_idx}/{args.epoch}')):
        x, labels = batch
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

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # trn_loss summary
        trn_loss += loss.item()
        batch_acc = 100 * train_correct / train_total
        # del (memory issue)
        del loss
        del model_output
        
        if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
                tqdm.write(f'TRAIN: {batch_idx}/{len(dataloader)} - Loss={loss.item()}')
        if args.save_gradient_flow:
            plot_grad_flow(model.named_parameters())
        if args.use_tensorboard_logging:
            total_idx = batch_idx + (epoch_idx * len(dataloader))
            writer.add_scalar('TRAIN/Loss', loss.item() / args.batch_size, total_idx)
            writer.add_scalar('TRAIN/Batch_Accuracy', batch_acc.item(), batch_idx+(epoch_idx*len(dataloader)))
        torch.save(model.state_dict(), args.save_modle_path) 