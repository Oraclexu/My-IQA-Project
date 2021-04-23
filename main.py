"""
Define Hyper-parameters
Init Dataset and Model
Run
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import os
from torch.backends import cudnn

from DataLoader import FlatDirectoryImageDataset, \
    get_transform, get_data_loader, FoldersDistributedDataset
from MSG_GAN.GAN import MSG_GAN
from argument import parse_arguments
from loaders import loaders
from solver import solver

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True

torch.manual_seed(seed=3)


def main(args):
    # Initialize Dataset
    data = loaders(args)

    # Initialize gan and train
    solver(args, data)


if __name__ == '__main__':
    main(parse_arguments())
