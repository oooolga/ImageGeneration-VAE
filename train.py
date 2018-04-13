import argparse
import torch
import copy
from util import *
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--valid_prop', type=float, default=0.3, help='validation proportion')
args = parser.parse_args()

train_loader, valid_loader, test_loader = load_data(args)


for img, _ in train_loader:
    pdb.set_trace()
    pass

for img, _ in test_loader:
    #TODO: test here
    pass

