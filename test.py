import os
import argparse
from tqdm.auto import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import dataset
import data_preprocessing
import epoch
from model import EfficientNet
from utils import get_tb_experiment_name

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = {}
    dataloaders = {}

    traindataset, traindataloaders = dataset.init_dataset(args)
    if(args.dataset_preprocessing is True):
        traindataset, traindataloaders = data_preprocessing.data_aguments(args, traindataset)
    
    datasets = traindataset
    dataloaders = traindataloaders

    if (args.model_type is 'original'):
        model = EfficientNet()
    if (args.model_type is 'pre-trained'):
        weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        model = torchvision.models.efficientnet_b4(weights = weights)

    if args.load_from_checkpoint is not None:
        if os.path.exists(args.load_from_checkpoint) and os.path.isfile(args.load_from_checkpoint):
            model.load_state_dict(torch.load(args.load_from_checkpoint))
            print(f'Model loaded from {args.load_from_checkpoint}')
        else:
            print(args.load_from_checkpoint, 'Checkpoint does not exist or is not a file, train from scratch')

    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epoch / 2))
    loss_fn = nn.CrossEntropyLoss()

    print(model)
    if args.use_tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.tensorboard_log_dir, get_tb_experiment_name(args)))
        writer.add_text('model', str(model))
        writer.add_text('args', str(args))
        epoch.test_model()