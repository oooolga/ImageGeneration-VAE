import torch
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


def visualize_kernel(kernel_tensor, im_name='conv1_kernel.jpg', pad=1, im_scale=1.0,
                     model_name='', rescale=True, result_path='.'):

    # map tensor wight in [0,255]
    if rescale:
        max_w = torch.max(kernel_tensor)
        min_w = torch.min(kernel_tensor)
        scale = torch.abs(max_w-min_w)
        kernel_tensor = (kernel_tensor - min_w) / scale * 255.0
        kernel_tensor = torch.ceil(kernel_tensor)

    # pad kernel
    p2d = (pad, pad, pad, pad)
    padded_kernel_tensor = F.pad(kernel_tensor, p2d, 'constant', 255)

    # get the shape of output
    grid_Y, grid_X = factorization(kernel_tensor.size(0))
    Y, X = padded_kernel_tensor.size(2), padded_kernel_tensor.size(3)

    # reshape
    # (grid_Y*grid_X) x y_dim x x_dim x num_chann
    padded_kernel_tensor = padded_kernel_tensor.permute(0, 2, 3, 1)
    padded_kernel_tensor = padded_kernel_tensor.cpu().view(grid_X, grid_Y*Y, X, -1)
    padded_kernel_tensor = padded_kernel_tensor.permute(0, 2, 1, 3)
    #padded_kernel_tensor = padded_kernel_tensor.view(1, grid_X*X, grid_Y*Y, -1)

    # kernel in numpy
    kernel_im = np.uint8((padded_kernel_tensor.data).numpy()).reshape(grid_X*X,
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
        loss: [1]
        KL: [1]
        reconst_loss [1]
    """
    if k == 0:
        kl, log_x_cond_z, lower_bound, _ = model.inference(imgs)
        return -torch.mean(lower_bound), torch.mean(kl), -torch.mean(log_x_cond_z)
    else:
        raise NotImplementedError


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