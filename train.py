import argparse
import torch
import copy
from util import *
from vae import *
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import pdb

USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--valid_prop', type=float, default=0, help='validation proportion')
parser.add_argument('--operation', type=str, default='deconvolution',
                    help='[deconvolution|nearest|bilinear]')
parser.add_argument('--importance_weight', action='store_true')
parser.add_argument('--k', type=int, default=None, help='k sample')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='momentum')
args = parser.parse_args()

def print_all_settings(args, model):

    print('Batch size:\t\t{}'.format(args.batch_size))
    print('Operation:\t\t{}'.format(args.operation))
    print('Importance weight:\t{}'.format(args.importance_weight))
    print('k-sample:\t\t{}\n'.format(args.k))
    print('Learning rate:\t\t{}\n'.format(args.lr))

    num_params = 0
    for param in model.parameters():
        num_params += np.prod([ sh for sh in param.shape])
    print('Model capacity:\t\t{}\n'.format(num_params))


def train(train_loader, model, optimizer, importance_flag, k=None):

    model.train()
    for batch_idx, (imgs, _) in enumerate(train_loader):

        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)

        optimizer.zero_grad()

        if importance_flag:
            # to do: lower bound
            reconst_loss, kl, lower_bounds = model.importance_inference(imgs, k=k)
        else:
            reconst_loss, kl, reconst = model.inference(imgs)
            lower_bound = -reconst_loss - kl
            loss = -lower_bound

        loss.backward()
        optimizer.step()

        print('loss={:.2f}'.format(loss[0].data[0]))

def test(test_loader, model):

    model.eval()
    for img, _ in test_loader:
        #TODO: test here
        pass

if __name__ == '__main__':

    model = get_model(args.operation)
    print_all_settings(args, model)
    train_loader, valid_loader, test_loader = load_data(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(train_loader, model, optimizer, args.importance_weight)
    



