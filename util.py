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

def load_data(args, eval_only=False):
    if not eval_only:
        train_transform = Compose([
            TrainTransform(),
        ])
        train_dset = datasets.ImageFolder(
                root=os.path.join(args.data_path, 'train'),
                transform=train_transform
                )
        train_loader = torch.utils.data.DataLoader(
                train_dset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=8,
                )
    else:
        train_loader = None

    test_transform = Compose([
        TestTransform(),
    ])
    test_dset = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'),
                                     transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=8
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
        tensor *= 256
        tensor = torch.floor(tensor)
        tensor = torch.clamp(tensor, min=0, max=255)

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
    padded_tensor = padded_tensor.contiguous()
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
        # log(1/n(w_1+w_2+...+w_n))
        # [bsz]
        lower_bounds_avg = _logsumexp(lower_bounds) - math.log(lower_bounds.size(1))
        loss = -torch.mean(lower_bounds_avg)

    return loss, kl, reconst_loss

def _log2(x):
    return torch.log(x) / math.log(2.0)

def _logsumexp(x):
    """
    doing an exp following a sum and a log.
    :param x: [batch_size, num_samples]
    :return: [batch_size]
    """
    # [batch_size, 1]
    max_logits = torch.max(x, dim=1, keepdim=True)[0]

    # [batch_size]
    sum_exp = torch.sum(torch.exp(x - max_logits), dim=1, keepdim=True)

    return (max_logits + torch.log(sum_exp)).squeeze(1)

def get_batch_bpp(model, imgs):
    """
    :param model: the vae
    :param imgs: [bsz, 3, 64, 64]
    :return: batch average bpp
    from https://arxiv.org/pdf/1705.05263.pdf sec 2.4
    """
    D = np.prod([sh for sh in imgs.shape[1:]] )

    # sample 2000 in total
    # use a batch sample of 200 for 10 times
    batch_samples = 200
    lower_bounds = []
    for batch_idx in range(int(2000 / batch_samples)):
        # [bsz, batch_samples] log w
        _, _, batch_lower_bounds = model.importance_inference(imgs, k=batch_samples)
        lower_bounds.append(batch_lower_bounds)
    # [bsz, 2000]
    lower_bounds = torch.cat(lower_bounds, dim=1)

    # average over 2000 samples
    # log(mean(exp(lower_bounds))) [bsz]
    importance_sample_avg = _logsumexp(lower_bounds) - math.log(2000)

    # change base to log2
    LL_2_base = importance_sample_avg / math.log(2)
    return -torch.mean(LL_2_base - D*math.log2(256)) / float(D)


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
