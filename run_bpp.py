import argparse
from util import get_model_optimizer, load_data, print_all_settings, get_batch_loss, visualize, bpp_per_img
from util import save_checkpoint, load_checkpoint
from vae import USE_CUDA
from torch.autograd import Variable
import time
from tqdm import tqdm
from transforms import Compose, TestTransform
from torchvision import datasets
import torch
import os

def parse_args():
    """
    :return: args. A argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--load_model', type=str, default=None,
                     help='model load path')
    parser.add_argument('--print_freq', type=int, default=1,
                    help='print frequency')
    return parser.parse_args()


def eval_bpp(data_loader, model):
    model.eval()
    total_bpp = 0

    for batch_idx, (imgs, _) in tqdm(enumerate(data_loader)):
        imgs = Variable(imgs, volatile=True).cuda() if USE_CUDA else Variable(imgs)
        batch_bpp = bpp_per_img(model, imgs, 256)
        total_bpp += batch_bpp[0].data[0]

    avg_bpp = total_bpp / (1+batch_idx)
    return avg_bpp


if __name__ == '__main__':
    args = parse_args()

    # load data and model
    model, _, _, epoch_i = load_checkpoint(args.load_model)
    test_transform = Compose([
        TestTransform(),
    ])
    test_dset = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'),
                                     transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dset,
                                              batch_size=1,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=8
                                              )
    avg_bpp = eval_bpp(test_loader, model)
    print("test avg bpp {:.4f}".format(avg_bpp))
