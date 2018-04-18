import argparse
from util import get_model_optimizer, load_data, print_all_settings, get_batch_loss, visualize, get_batch_bpp
from util import save_checkpoint, load_checkpoint
from vae import USE_CUDA
from torch.autograd import Variable
import time

def parse_args():
    """
    :return: args. A argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--load_model', type=str, default=None,
                     help='model load path')
    parser.add_argument('--model_name', type=str, default='deconvolution')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--print_freq', type=int, default=100,
                    help='print frequency')
    return parser.parse_args()


def eval_bpp(data_loader, model, args):
    model.eval()
    start = time.time()

    total_bpp = 0
    display_bpp = 0

    for batch_idx, (imgs, _) in enumerate(data_loader):
        imgs = Variable(imgs, volatile=True).cuda() if USE_CUDA else Variable(imgs)
        batch_bpp = get_batch_bpp(model, imgs)
        display_bpp += batch_bpp[0].data[0] / args.print_freq
        total_bpp += batch_bpp[0].data[0]

        if (batch_idx+1) % args.print_freq == 0:
            print('|\t\tbatch #:{}\tlbpp={:.2f}\tuse {:.2f} sec'.format(
                batch_idx+1, display_bpp, time.time()-start))
            start = time.time()
            display_bpp = 0

    avg_bpp = total_bpp / (1+batch_idx)
    return avg_bpp


if __name__ == '__main__':
    args = parse_args()

    # load data and model
    model, _, _, epoch_i = load_checkpoint(args.load_model)
    _, test_loader = load_data(args)

    avg_bpp = eval_bpp(test_loader, model, args)
    print("test avg bpp {:.4f}".format(avg_bpp))
