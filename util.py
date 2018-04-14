import torch
import torch.nn.functional as F
import torch.optim as optim
from vae import VariationalAutoEncoder, VariationalUpsampleEncoder, USE_CUDA
import numpy as np
import torch.nn.functional as F
import numpy as np
import argparse, os
import copy, scipy
import scipy.misc

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets
from transforms import Compose, TestTransform, TrainTransform
import math
ZERO = 1e-5 # for numeric stability

def load_data(args):

    train_transform = Compose([
        TrainTransform(),
    ])
    train_dset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'),
                                      transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=4,
                                               )

    test_transform = Compose([
        TestTransform(),
    ])
    test_dset = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'),
                                     transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=4
                                              )

    print('Finished loading data...')
    return train_loader, test_loader


def get_model_optimizer(mode, z_dim, lr):
    if mode == 'deconvolution':
        model = VariationalAutoEncoder(z_dim)
    else:
        model = VariationalUpsampleEncoder(mode=mode, z_dim=z_dim)

    if USE_CUDA:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr)
    return model, optimizer


def factorization(n):
    from math import sqrt
    for i in range(int(sqrt(float(n))), 0, -1):
        if n % i == 0:
            #if i == 1: print('Who would enter a prime number of filters')
            return i, int(n / i)


def visualize(tensor, im_name='conv1_kernel.png', pad=1, im_scale=1.0,
              model_name='', rescale=True, result_path='.'):

    # map tensor wight in [0,255]
    if rescale:
        tensor *= 255.0
        tensor = torch.ceil(tensor)

    # pad kernel
    p2d = (pad, pad, pad, pad)
    padded_tensor = F.pad(tensor, p2d, 'constant', 255)

    # get the shape of output
    grid_Y, grid_X = factorization(tensor.size(0))
    Y, X = padded_tensor.size(2), padded_tensor.size(3)

    # reshape
    # (grid_Y*grid_X) x y_dim x x_dim x num_chann
    padded_tensor = padded_tensor.permute(0, 2, 3, 1)
    if USE_CUDA:
        padded_tensor = padded_tensor.cpu()
    padded_tensor = padded_tensor.view(grid_X, grid_Y*Y, X, -1)
    padded_tensor = padded_tensor.permute(0, 2, 1, 3)
    #padded_tensor = padded_tensor.view(1, grid_X*X, grid_Y*Y, -1)

    # kernel in numpy
    kernel_im = np.uint8((padded_tensor.data).numpy()).reshape(grid_X*X,
                                                                       grid_Y*Y, -1)
    kernel_im = scipy.misc.imresize(kernel_im, im_scale, 'nearest')
    print('|\tSaving {}...'.format(os.path.join(result_path, model_name+'_'+im_name)))
    plt.imsave(os.path.join(result_path, model_name+'_'+im_name), kernel_im, origin='upper')


def get_batch_loss(model, imgs, k=0):
    """
    Get a batch average loss, KL, and reconst loss
    :param model: the model
    :param imgs: [bsz, 3, 64, 64]
    :param k: if 0 use pure inference else use importance weight inference
    :return:
        loss: [1] batch average loss. For importance weight, this is weighted average
        kl: [1] a KL value to be displayed
        reconst_loss [1] a reconstruction loss to be displayed
    """
    if k == 0:
        kl, log_x_cond_z, lower_bound, _ = model.inference(imgs)
        loss = -torch.mean(lower_bound)
        kl = torch.mean(kl)
        reconst_loss = -torch.mean(log_x_cond_z)
    else:
        mc_kl, mc_log_x_cond_z, lower_bounds = model.importance_inference(imgs, k)
        kl = torch.mean(mc_kl)
        reconst_loss = -torch.mean(mc_log_x_cond_z)

        # bp_weights [bsz, k] and detach
        bp_weights = F.softmax(lower_bounds, dim=1).detach()
        # weighted average over lower bounds (log w)
        weighted_lower_bounds = torch.mean(lower_bounds * bp_weights, dim=1)
        loss = -torch.mean(weighted_lower_bounds)

    return loss, kl, reconst_loss

def _log2(x):
    return torch.log(x) / math.log(2.0)

def get_batch_bpp(model, imgs):
    """
    :param model: the vae
    :param imgs: [bsz, 3, 64, 64]
    :return: batch average bpp
    from https://arxiv.org/pdf/1705.05263.pdf sec 2.4
    """
    D = np.prod([sh for sh in imgs.shape[1:]] )
    num_samples = 100
    # average 2000 result
    importance_sample_sum = 0
    for _ in range(num_samples):
        # log p(x,z)/q(z|x)
        lower_bound = model.inference(imgs)[2]
        importance_sample_sum += torch.exp(lower_bound)

    # [bsz]
    LL = _log2(importance_sample_sum / num_samples)
    return torch.mean(LL - D * math.log2(256))


def save_checkpoint(state, model_name):
    torch.save(state, model_name)
    print('Finished saving model: {}'.format(model_name))

def load_checkpoint(model_name):
    if model_name and os.path.isfile(model_name):
        checkpoint = torch.load(model_name)
        args = checkpoint['args']
        model, optimizer = get_model_optimizer(args.operation, args.z_dim, args.lr)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Finished loading model and optimizer from {}'.format(model_name))
    else:
        print('File {} not found.'.format(model_name))
        raise FileNotFoundError
    return model, optimizer, args, checkpoint['epoch_i']

def print_all_settings(args, model):

    print('Batch size:\t\t{}'.format(args.batch_size))
    print('Total epochs:\t\t{}'.format(args.epochs))
    print('Operation:\t\t{}'.format(args.operation))
    print('k-sample:\t\t{}\n'.format(args.k))
    print('Learning rate:\t\t{}\n'.format(args.lr))

    num_params = 0
    for param in model.parameters():
        num_params += np.prod([ sh for sh in param.shape])

    print('Model capacity:\t\t{}\n'.format(num_params))
