import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

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

        self.fc_mu = nn.Linear(d*16, 100)
        self.fc_logvar = nn.Linear(d*16, 100)

        # loss
        self.bce = nn.BCELoss(size_average=False)

    def _encode(self, img):
        """
        https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_CelebA_DCGAN.py
        :param img: [bsz, w, h, c]
        :return: mu [bsz, 100], logvar [bsz, 100]
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
        :return: [bsz, 3, 64, 64]
        """
        raise NotImplementedError


    def inference(self, imgs):
        """
        perform an enc and dec, getting the reconst loss, kl loss and reconstruction.
        :param imgs: [bsz, 3, 64, 64]
        :return:
            reconst_loss: [bsz, 1]
            kl_loss: [bsz, 1]
            reconst: [bsz, 3, 64, 64]
        """
        mu, logvar = self._encode(imgs)
        reconst = self._decode(mu, logvar)

        # binary cross entropy loss
        reconst_loss = self.bce(reconst, imgs)

        # kl loss
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        return reconst_loss, kl_loss, reconst



class VariationalAutoEncoder(VAEBase):
    """
    An VAE use dcgan decoder
    """
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        # https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_CelebA_DCGAN.py
        # decoder
        d = 128
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
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
        :param mean: [bsz, 100]
        :param logvar: [bsz, 100]
        :return: [bsz, 3, 64, 64]
        """
        eps = torch.FloatTensor(mean.shape)
        eps.normal_()
        eps = Variable(eps)
        if USE_CUDA:
            eps = eps.cuda()

        z = mean + eps * torch.exp(0.5*logvar)
        z = z.view(z.size(0), 100, 1, 1)

        tmp = F.leaky_relu(self.deconv1_bn(self.deconv1(z)))
        tmp = F.leaky_relu(self.deconv2_bn(self.deconv2(tmp)))
        tmp = F.leaky_relu(self.deconv3_bn(self.deconv3(tmp)))
        tmp = F.leaky_relu(self.deconv4_bn(self.deconv4(tmp)))
        reconst = F.sigmoid(self.deconv5(tmp))

        return reconst



class VariationalUpsampleEncoder(VAEBase):
    def __init__(self, upsample):
        super(VariationalUpsampleEncoder, self).__init__()

        # build decoder here
        if upsample == "nearest":
            upsample_layer = nn.UpsamplingNearest2d
        elif upsample == "bilinnear":
            upsample_layer = nn.UpsamplingBilinear2d
        else:
            print("{} upsample is not implemented".format(upsample))
            raise NotImplementedError

    def _decode(self, mean, logvar):
        raise NotImplementedError



model = VariationalAutoEncoder()
imgs = torch.rand([2, 3, 64, 64])
reconst_loss, kl_loss, reconst = model.inference(Variable(imgs))

