import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  


def set_random_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_tb_experiment_name(args):

    exp_name = str()

    exp_name += "ModelName=%s - " % args.model_name
    if args.load_from_checkpoint is not None:
        exp_name += "Checkpoint=%s - " % args.load_from_checkpoint
    exp_name += "BS=%i_" % args.batch_size 
    if args.epoch is not None:
        exp_name += "EP=%i_" % args.epoch
    if args.lr is not None:
        exp_name += "LR=%.4f_" % args.lr
    exp_name += "SEED=%i_" % args.seed


    return exp_name

def vae_batch_accuracy(pred, target, non_pad_len):

    batch_size = pred.size(0)
    batch_correct = 0

    for idx in range(batch_size):
        target_len = non_pad_len[idx]
        max_pred = pred[idx].argmax(dim=-1)
        max_pred = max_pred[:target_len]
        each_correct = (max_pred == target[idx, :target_len]).sum().float()
        batch_correct += each_correct

    accuracy = 100 * batch_correct / (non_pad_len.sum())
    return accuracy

def batch_accuracy_test(pred, target):
 

    batch_size = pred.size(0)
    batch_correct = 0
    total_observation = 0

    for idx in range(batch_size):
        smaller_len = min(pred[idx].size(0), target[idx].size(0))
        each_correct = (pred[idx, :smaller_len] == target[idx, :smaller_len]).sum().float()

        batch_correct += each_correct
        total_observation += smaller_len

    accuracy = 100 * batch_correct / total_observation
    return accuracy


def plot_grad_flow(named_parameters):

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) 
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    
def plot_result_gragh(args, target, prediction):
    
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.plot(target, label='target')
    plt.plot(prediction, label='prediction')

    plt.title('prediction vs target')
    plt.legend()
    
    data_path = os.path.join(os.getcwd(), "figure_save")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    plt.savefig(f"{data_path}/VAE_figure_epoch{int(args.epochs)}_batch{args.batch_size}.png")