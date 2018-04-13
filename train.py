import argparse
from util import get_model, load_data, print_all_settings, get_batch_loss, visualize_kernel
from vae import USE_CUDA
from torch.autograd import Variable
import torch.optim as optim
import torch
import pdb
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--result_path', type=str, default='./result')
parser.add_argument('--model_name', type=str, default='deconvolution')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--sample_size', type=int, default=36, help='sample size')
parser.add_argument('--epochs', type=int, default=1,
                    help='total epochs')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--valid_prop', type=float, default=0, help='validation proportion')
parser.add_argument('--operation', type=str, default='deconvolution',
                    help='[deconvolution|nearest|bilinear]')
parser.add_argument('--k', type=int, default=0,
                    help="if 0 then use pure inference else use importance weighted inference")
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='momentum')
args = parser.parse_args()


def train(train_loader, model, optimizer, args):
    """
    train model over
    """
    model.train()
    start = time.time()
    display_loss = 0
    display_kl = 0
    display_reconst_loss = 0
    for batch_idx, (imgs, _) in enumerate(train_loader):

        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)

        optimizer.zero_grad()
        loss, kl, reconst_loss = get_batch_loss(model, imgs, args.k)
        display_loss += loss[0].data[0] / args.print_freq
        display_kl += kl[0].data[0] / args.print_freq
        display_reconst_loss += reconst_loss[0].data[0] / args.print_freq

        loss.backward()
        optimizer.step()

        if (batch_idx+1) % args.print_freq == 0:
            print('|\t\tbatch #:{}\tloss={:.2f}\tkl={:.2f}\treconst_loss={:.2f}\tuse {:.2f} sec'.format(
                batch_idx+1, display_loss, display_kl, display_reconst_loss, time.time()-start))
            start = time.time()
            display_loss = 0
            display_kl = 0
            display_reconst_loss = 0


def eval(data_loader, model, args):

    model.eval()
    start = time.time()
    total_loss = 0
    display_loss = 0
    display_kl = 0
    display_reconst_loss = 0

    for batch_idx, (imgs, _) in enumerate(data_loader):
        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)
        loss, kl, reconst_loss = get_batch_loss(model, imgs, args.k)
        display_loss += loss[0].data[0] / args.print_freq
        display_kl += kl[0].data[0] / args.print_freq
        display_reconst_loss += reconst_loss[0].data[0] / args.print_freq

        total_loss += loss[0].data[0]

        if (batch_idx+1) % args.print_freq == 0:
            print('|\t\tbatch #:{}\tloss={:.2f}\tkl={:.2f}\treconst_loss={:.2f}\tuse {:.2f} sec'.format(
                batch_idx+1, display_loss, display_kl, display_reconst_loss, time.time()-start))
            start = time.time()
            display_loss = 0
            display_kl = 0
            display_reconst_loss = 0

    avg_loss = total_loss/len(data_loader)
    return avg_loss


def sample_visualization(data_loader, model, im_name, sample_size):

    model.eval()

    for imgs, _ in data_loader:
        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)
        imgs = imgs[:sample_size]
        break

    reconst = model.reconstruction_sample(imgs)

    visualize_kernel(reconst, im_name=im_name, im_scale=1.0,
                     model_name=model_name, result_path=result_path)
    visualize_kernel(imgs, im_name=im_name[:-4]+'_org.jpg', im_scale=1.0,
                     model_name=model_name, result_path=result_path)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':

    model = get_model(args.operation)
    model.apply(weight_init)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    global batch_size, print_freq, result_path, model_name
    batch_size, print_freq, result_path = args.batch_size, args.print_freq, args.result_path
    model_name = args.model_name

    print_all_settings(args, model)
    train_loader, _, test_loader = load_data(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch_i in range(1, args.epochs+1):
        print('|\tEpoch {}:'.format(epoch_i))
        print('|\t\tTrain:')
        train(train_loader, model, optimizer, args)

        sample_visualization(train_loader, model, 'epoch_{}_train.jpg'.format(epoch_i),
                             args.sample_size)
        sample_visualization(test_loader, model, 'epoch_{}_test.jpg'.format(epoch_i),
                             args.sample_size)

        print('|\t\tEval train:')
        avg_train_loss = eval(train_loader, model, args)
        print('|\tTrain loss={}\n'.format(avg_train_loss))
        print('|\t\tEval test:')
        avg_test_loss = eval(test_loader, model, args)
        print('|\tTest loss={}\n'.format(avg_test_loss))

