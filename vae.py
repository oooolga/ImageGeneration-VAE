import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
Z_DIM = 1

class VAEBase(nn.Module):
    """
    a base class for vae
    """
    def __init__(self):
        super(VAEBase, self).__init__()
        # encoder are the same build it
        # A symmetric architecture of DCGAN
        d = 128
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 16*d, 4, 1, 0)

        self.fc_mu = nn.Linear(d*16, Z_DIM)
        self.fc_logvar = nn.Linear(d*16, Z_DIM)

        # loss
        self.bce = nn.BCELoss(size_average=False)

    def _encode(self, img):
        """
        https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_CelebA_DCGAN.py
        :param img: [bsz, w, h, c]
        :return: mu [bsz, Z_DIM], logvar [bsz, Z_DIM]
        """
        tmp = F.leaky_relu(self.conv1_bn(self.conv1(img)))
        tmp = F.leaky_relu(self.conv2_bn(self.conv2(tmp)))
        tmp = F.leaky_relu(self.conv3_bn(self.conv3(tmp)))
        tmp = F.leaky_relu(self.conv4_bn(self.conv4(tmp)))
        tmp = self.conv5(tmp)
        tmp = tmp.view(tmp.size(0), -1)

        mu = self.fc_mu(tmp)
        logvar = self.fc_logvar(tmp)
        return mu, logvar


    def _decode(self, mean, logvar):
        """
        Decode from given mean and logvar of z.
        :param mean: [bsz, 100]
        :param logvar: [bsz, 100]
        :return: reconst [bsz, 3, 64, 64], eps [bsz, Z_DIM]
        """
        raise NotImplementedError

    def importance_inference(self, imgs, k=50):
        """
        Perform an importance weighted lower bound estimate.
        :param imgs: the images
        :param k: number of sampling from posterior
        :return:
            lower_bounds: a list of lower bounds. [bsz, k]
            kl: the estimated KL(q(z|x) | p(z)). [bsz, 1]
            reconst_loss: the estimated reconstruction loss [bsz, 1]
        """
        mu, logvar = self._encode(imgs)
        sigma = torch.exp(0.5*logvar)

        # store log w of each sample from posterior
        lower_bounds = [] # used for backward

        # used for display average result
        kl = 0
        reconst_loss = 0
        for _ in range(k):
            reconst, eps = self._decode(mu, logvar)

            # log p(x | z)
            log_x_cond_z = -self.bce(reconst, imgs)
            reconst_loss += -log_x_cond_z / k

            # log p(z) - log q(z | x) = 0.5*(-mu^2 - sigma^2 eps^2 - 2*mu*sigma*eps + log sigma^2 + eps^2)
            # this is the sample estimate of KL loss.
            log_prior_minus_pos = 0.5*(-mu.pow(2) - sigma.pow(2)*eps.pow(2) - 2*mu*sigma*eps + logvar + eps.pow(2))
            log_prior_minus_pos = torch.sum(log_prior_minus_pos, dim=-1)

            # add the sample estimate of KL(q(z|x) | p(z))
            kl += -log_prior_minus_pos / k

            # log w
            lower_bound = log_x_cond_z + log_prior_minus_pos
            lower_bounds.append(lower_bound)

        lower_bounds = torch.stack(lower_bounds, 1)
        return reconst_loss, kl, lower_bounds

    def inference(self, imgs):
        """
        perform an enc and dec, getting the reconst loss, kl loss and reconstruction.
        Use an analytical form of KL the same as original vae paper.
        :param imgs: [bsz, 3, 64, 64]
        :return:
            reconst_loss: [bsz, 1]
            kl: [bsz, 1]
            reconst: [bsz, 3, 64, 64]
        """
        mu, logvar = self._encode(imgs)
        reconst, _ = self._decode(mu, logvar)

        # binary cross entropy loss
        reconst_loss = self.bce(reconst, imgs)

        # KL(q(z|x) | p(z))
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        return reconst_loss, kl, reconst



class VariationalAutoEncoder(VAEBase):
    """
    An VAE use dcgan decoder
    """
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        # https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_CelebA_DCGAN.py
        # decoder
        d = 128
        self.deconv1 = nn.ConvTranspose2d(Z_DIM, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)


    def _decode(self, mean, logvar):
        """
        Decode from given mean and logvar of z.
        :param mean: [bsz, Z_DIM]
        :param logvar: [bsz, Z_DIM]
        :return: [bsz, 3, 64, 64]
        """
        eps = torch.FloatTensor(mean.shape)
        eps.normal_()
        eps = Variable(eps)
        if USE_CUDA:
            eps = eps.cuda()

        z = mean + eps * torch.exp(0.5*logvar)
        z = z.view(z.size(0), Z_DIM, 1, 1)

        tmp = F.leaky_relu(self.deconv1_bn(self.deconv1(z)))
        tmp = F.leaky_relu(self.deconv2_bn(self.deconv2(tmp)))
        tmp = F.leaky_relu(self.deconv3_bn(self.deconv3(tmp)))
        tmp = F.leaky_relu(self.deconv4_bn(self.deconv4(tmp)))
        reconst = F.sigmoid(self.deconv5(tmp))

        return reconst, eps



class VariationalUpsampleEncoder(VAEBase):
    def __init__(self, mode='bilinear'):
        super(VariationalUpsampleEncoder, self).__init__()

        d = 128
        self.up1 = nn.Upsample(scale_factor=8, mode=mode)
        self.deconv1 = nn.Conv2d(Z_DIM, d*8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d*8)

        self.up2 = nn.Upsample(scale_factor=4, mode=mode)
        self.deconv2 = nn.Conv2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)

        self.up3 = nn.Upsample(scale_factor=4, mode=mode)
        self.deconv3 = nn.Conv2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)

        self.up4 = nn.Upsample(scale_factor=4, mode=mode)
        self.deconv4 = nn.Conv2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)

        self.up5 = nn.Upsample(scale_factor=4, mode=mode)
        self.deconv5 = nn.Conv2d(d, 3, 4, 2, 1)

    def _decode(self, mean, logvar):
        """
        Decode from given mean and logvar of z.
        :param mean: [bsz, Z_DIM]
        :param logvar: [bsz, Z_DIM]
        :return: [bsz, 3, 64, 64]
        """
        eps = torch.FloatTensor(mean.shape)
        eps.normal_()
        eps = Variable(eps)
        if USE_CUDA:
            eps = eps.cuda()

        z = mean + eps * torch.exp(0.5*logvar)
        z = z.view(z.size(0), Z_DIM, 1, 1)

        import ipdb
        # [1024, 4, 4]
        tmp = self.deconv1_bn(self.deconv1(self.up1(z)))
        tmp = F.leaky_relu(tmp)

        # [512, 8, 8]
        tmp = self.deconv2_bn(self.deconv2(self.up2(tmp)))
        tmp = F.leaky_relu(tmp)

        # [256, 16, 16]
        tmp = self.deconv3_bn(self.deconv3(self.up3(tmp)))
        tmp = F.leaky_relu(tmp)

        # [128, 32, 32]
        tmp = self.deconv4_bn(self.deconv4(self.up4(tmp)))
        tmp = F.leaky_relu(tmp)

        # [3, 64, 64]
        tmp = self.deconv5(self.up5(tmp))
        return F.sigmoid(tmp), eps


if __name__ == '__main__':
    import numpy as np

    def test_vae(vae):
        num_params = 0
        for param in vae.parameters():
            num_params += np.prod([ sh for sh in param.shape])
        imgs = torch.rand([2, 3, 64, 64])
        _, _, reconst = vae.inference(Variable(imgs))
        assert reconst.size(2) == 64
        print("num_param: {}".format(num_params))

    vae_nearest = VariationalUpsampleEncoder(mode='nearest')
    test_vae(vae_nearest)
    vae_bilinear = VariationalUpsampleEncoder(mode='bilinear')
    test_vae(vae_bilinear)
    vae = VariationalAutoEncoder()
    test_vae(vae)

    # two different inference
    imgs = torch.rand([3, 3, 64, 64])
    reconst_loss, kl, reconst = vae.inference(Variable(imgs))
    lower_bound = -reconst_loss - kl
    print("reconst_loss {}, kl {}".format(reconst_loss[0].data[0], kl[0].data[0]))

    reconst_loss, kl, lower_bounds = vae.importance_inference(Variable(imgs), k=5)
    print("reconst_loss {}, kl {}".format(reconst_loss[0].data[0], kl[0].data[0]))

