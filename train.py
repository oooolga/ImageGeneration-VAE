import argparse
from util import get_model_optimizer, load_data, print_all_settings, get_batch_loss, visualize, get_batch_bpp
from util import save_checkpoint, load_checkpoint
from vae import USE_CUDA
from torch.autograd import Variable
import torch
import pdb
import time
import os
import math

def parse_args():
    """
    :return: args. A argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='./result')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None,
                     help='model load path')
    parser.add_argument('--evaluate', action='store_true', help='if turn on then only evaluate on test set')
    parser.add_argument('--model_name', type=str, default='deconvolution')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--sample_size', type=int, default=36, help='sample size')
    parser.add_argument('--epochs', type=int, default=10,
                    help='total epochs')
    parser.add_argument('--print_freq', type=int, default=100,
                    help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                    help='save frequency')
    parser.add_argument('--operation', type=str, default='deconvolution',
                    help='[deconvolution|nearest|bilinear]')
    parser.add_argument('--k', type=int, default=0,
                    help="if 0 then use pure inference else use importance weighted inference")
    parser.add_argument('--z_dim', type=int, default=100,
                    help="latent variable\'s dimension")
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
    return parser.parse_args()


def train(train_loader, model, optimizer, args):
    """
    train model over
    """
    model.train()
    start = time.time()
    display_loss = 0
    display_kl = 0
    display_reconst_loss = 0
    total_loss = 0
    for batch_idx, (imgs, _) in enumerate(train_loader):

        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)

        optimizer.zero_grad()
        loss, kl, reconst_loss = get_batch_loss(model, imgs, args.k)
        display_loss += loss[0].data[0] / args.print_freq
        display_kl += kl[0].data[0] / args.print_freq
        display_reconst_loss += reconst_loss[0].data[0] / args.print_freq
        total_loss += loss[0].data[0]

        loss.backward()
        optimizer.step()

        if (batch_idx+1) % args.print_freq == 0:
            print('|\t\tbatch #:{}\tloss={:.2f}\tkl={:.2f}\treconst_loss={:.2f}\tuse {:.2f} sec'.format(
                batch_idx+1, display_loss, display_kl, display_reconst_loss, time.time()-start))
            start = time.time()
            display_loss = 0
            display_kl = 0
            display_reconst_loss = 0
    return total_loss / (1+batch_idx)


def eval(data_loader, model, args):
    model.eval()
    start = time.time()

    total_loss = 0
    display_loss = 0
    display_kl = 0
    display_reconst_loss = 0

    for batch_idx, (imgs, _) in enumerate(data_loader):
        imgs = Variable(imgs, volatile=True).cuda() if USE_CUDA else Variable(imgs)
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

    avg_loss = total_loss / (1+batch_idx)
    return avg_loss

def eval_latent_space(data_loader, model, args):
    model.eval()

    for imgs, _ in data_loader:
        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)
        imgs = imgs[:sample_size]
        break

    model.compute_change_in_z(imgs, 0)


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
            print('|\t\tbatch #:{}\tloss={:.2f}\tkl={:.2f}\treconst_loss={:.2f}\tuse {:.2f} sec'.format(
                batch_idx+1, display_loss, display_kl, display_reconst_loss, time.time()-start))
            start = time.time()
            display_loss = 0
            display_kl = 0
            display_reconst_loss = 0

    avg_loss = total_bpp / (1+batch_idx)
    return avg_loss


def sample_visualization(data_loader, model, im_name, sample_size):

    model.eval()

    for imgs, _ in data_loader:
        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)
        imgs = imgs[:sample_size]
        break

    reconst = model.reconstruction_sample(imgs)

    visualize(reconst, im_name=im_name, im_scale=1.0,
                     model_name=args.model_name, result_path=result_path)
    visualize(imgs, im_name=im_name[:-4]+'_org.png', im_scale=1.0,
                     model_name=args.model_name, result_path=result_path)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def interpolate_samples(data_loader, model, args, mode='train'):
    model.eval()

    for imgs, _ in data_loader:
        imgs = Variable(imgs).cuda() if USE_CUDA else Variable(imgs)
        imgs = imgs[:2]
        break

    reconst_interp_z, reconst_interp_im = model.get_interpolate_images(imgs)
    reconst_interp_z = torch.from_numpy(reconst_interp_z).cuda()
    reconst_interp_im = torch.from_numpy(reconst_interp_im).cuda()
    visualize(reconst_interp_z, im_name='reconstruction_z_{}.png'.format(mode), im_scale=1.0,
              model_name=args.model_name, result_path=result_path)
    visualize(reconst_interp_im, im_name='reconstruction_im_{}.png'.format(mode), im_scale=1.0,
              model_name=args.model_name, result_path=result_path)


if __name__ == '__main__':
    args = parse_args()

    # make output result
    if os.path.isdir(args.out_dir):
        print("{} exists!".format(args.out_dir))
        exit()
    os.mkdir(args.out_dir)
    result_path = os.path.join(args.out_dir, 'imgs')
    if args.model_dir:
        model_path = args.model_dir
    else:
        model_path = os.path.join(args.out_dir, 'model')
    os.makedirs(result_path)
    if os.path.isdir(model_path):
        print("{} exists!".format(model_path))
        exit()
    os.makedirs(model_path)

    # load model
    curr_lr = args.lr
    if args.load_model:
        model, optimizer, args, epoch_i = load_checkpoint(args.load_model)
    else:
        model, optimizer = get_model_optimizer(args.operation, args.z_dim, curr_lr)
        model.apply(weight_init)
    print_all_settings(args, model)

    # load data
    train_loader, test_loader = load_data(args)

    # only evaluate the model
    pdb.set_trace()
    if args.evaluate:
        print('Qualitative Evaluations:')
        eval_latent_space(train_loader, model, args)
        interpolate_samples(train_loader, model, args)
        interpolate_samples(test_loader, model, args, mode='test')
        print('Evaluate BPP:')
        test_bpp = eval_bpp(test_loader, model, args)
        print("Test bpp: {:.2f}".format(test_bpp))
        exit(0)

    # main loop
    prev_avg_test_loss = math.inf
    sample_visualization(train_loader, model, 'epoch_0_train.png',
                         args.sample_size)
    sample_visualization(test_loader, model, 'epoch_0_test.png',
                         args.sample_size)
    for epoch_i in range(1, args.epochs+1):
        print('|\tEpoch {}:'.format(epoch_i))
        print('|\t\tTrain:')
        avg_train_loss = train(train_loader, model, optimizer, args)

        sample_visualization(train_loader, model, 'epoch_{}_train.png'.format(epoch_i),
                             args.sample_size)
        sample_visualization(test_loader, model, 'epoch_{}_test.png'.format(epoch_i),
                             args.sample_size)
        print('|\tTrain loss={}\n'.format(avg_train_loss))

        print('|\t\tEval test:')
        avg_test_loss  = eval(test_loader, model, args)
        print('|\tTest loss={}\n'.format(avg_test_loss))

        # learning rate decrease if not much improvment
        if (prev_avg_test_loss - avg_test_loss) / prev_avg_test_loss < 0.01:
            curr_lr /= 2
            print("decrease learning rate to {:.2f}".format(curr_lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

        if epoch_i % args.save_freq == 0:
            save_checkpoint({'epoch_i': epoch_i, 'args': args, 'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}, 
                              os.path.join(model_path, args.model_name+str(epoch_i)+'.pt'))
        prev_avg_test_loss = avg_test_loss

    interpolate_samples(train_loader, model, args)
    interpolate_samples(test_loader, model, args, mode='test')

    test_bpp = eval_bpp(test_loader, model, args)
    print("Test bpp: {:.2f}".format(test_bpp))

