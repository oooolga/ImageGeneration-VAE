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
parser.add_argument('--epochs', type=int, default=20,
                    help='total epochs')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
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
    print('Total epochs:\t\t{}'.format(args.epochs))
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
            raise NotImplementedError
        else:
            kl, log_x_cond_z, lower_bound, _ = model.inference(imgs)
            loss = -torch.mean(lower_bound)

        loss.backward()
        optimizer.step()

        if (batch_idx+1) % print_freq == 0:
            print('|\t\tbatch #:{}\tloss={:.2f}'.format(batch_idx+1,
                                                        loss[0].data[0]))

def eval(data_loader, model, importance_flag):

    model.eval()

    total_loss = 0
    num_data = 0
    for batch_idx, (imgs, _) in enumerate(data_loader):

        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)

        if importance_flag:
            # to do: lower bound
            raise NotImplementedError
        else:
            kl, log_x_cond_z, lower_bound, _ = model.inference(imgs)
            loss = -torch.sum(lower_bound)

        total_loss += loss[0].data[0]

        if (batch_idx+1) % print_freq == 0:
            print('|\t\tbatch #:{}\tloss={:.2f}'.format(batch_idx+1,
                                                        total_loss/(batch_idx+1)))

    avg_loss = total_loss/len(data_loader)
    return avg_loss


if __name__ == '__main__':

    model = get_model(args.operation)

    global batch_size, print_freq
    batch_size, print_freq = args.batch_size, args.print_freq

    print_all_settings(args, model)
    train_loader, _, test_loader = load_data(args)

    #avg_train_loss = eval(train_loader, model, args.importance_weight)
    #print('|\tEpoch:{}\tTrain loss={}'.format(0, avg_train_loss))
    #avg_test_loss = eval(test_loader, model, args.importance_weight)
    #print('|\tEpoch:{}\tTest loss={}'.format(0, avg_test_loss))


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch_i in range(1, args.epochs+1):
        print('|\tEpoch {}:'.format(epoch_i))
        print('|\t\tTrain:')
        train(train_loader, model, optimizer, args.importance_weight)

        print('|\t\tEval:')
        avg_train_loss = eval(train_loader, model, args.importance_weight)
        print('|\tTrain loss={}'.format(avg_train_loss))
        avg_test_loss = eval(test_loader, model, args.importance_weight)
        print('|\tTest loss={}'.format(avg_test_loss))
    



