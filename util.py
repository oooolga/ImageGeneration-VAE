import torch
from vae import *

import numpy as np
import argparse, os
import copy, scipy
import scipy.misc

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def load_data(args):

    from torchvision import datasets
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'),
                                      transform=transform)
    anchor_pt = int(len(train_dset) * (1-args.valid_prop))
    valid_dset = copy.deepcopy(train_dset)

    train_dset.imgs = train_dset.imgs[:anchor_pt]
    valid_dset.imgs = valid_dset.imgs[anchor_pt:]

    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True)

    test_dset = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'),
                                     transform=transform)
    test_loader = torch.utils.data.DataLoader(train_dset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=True)

    print('Finished loading data...')
    return train_loader, valid_loader, test_loader

def get_model(mode):
    if mode == 'deconvolution':
        model = VariationalAutoEncoder()
    else:
        model = VariationalUpsampleEncoder(mode=mode)

    if torch.cuda.is_available():
        model = model.cuda()
    return model

def factorization(n):
    from math import sqrt
    for i in range(int(sqrt(float(n))), 0, -1):
        if n % i == 0:
            if i == 1: print('Who would enter a prime number of filters')
            return int(n / i), i
 
def visualize(tensor, im_name='conv1_kernel.jpg', pad=1, im_scale=1.0,
              model_name='', rescale=True, result_path='.'):

    # map tensor wight in [0,255]
    tensor *= 255

    # pad kernel
    p2d = (pad, pad, pad, pad)
    padded_tensor = F.pad(tensor, p2d, 'constant', 255)

    # get the shape of output
    grid_Y, grid_X = factorization(tensor.size(0))
    Y, X = padded_tensor.size(2), padded_tensor.size(3)

    # reshape
    # (grid_Y*grid_X) x y_dim x x_dim x num_chann
    padded_tensor = padded_tensor.permute(0, 2, 3, 1)
    padded_tensor = padded_tensor.cpu().view(grid_X, grid_Y*Y, X, -1)
    padded_tensor = padded_tensor.permute(0, 2, 1, 3)
    #padded_tensor = padded_tensor.view(1, grid_X*X, grid_Y*Y, -1)

    # kernel in numpy
    kernel_im = np.uint8((padded_tensor.data).numpy()).reshape(grid_X*X,
                                                               grid_Y*Y, -1)
    kernel_im = scipy.misc.imresize(kernel_im, im_scale, 'nearest')
    print('|\tSaving {}...'.format(os.path.join(result_path, model_name+'_'+im_name)))
    plt.imsave(os.path.join(result_path, model_name+'_'+im_name), kernel_im, origin='upper')