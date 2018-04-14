import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb

USE_CUDA = torch.cuda.is_available()

class VAEBase(nn.Module):
    """
    a base class for vae
    """
    def __init__(self, z_dim=100):
        super(VAEBase, self).__init__()
        # encoder are the same build it
        # A symmetric architecture of DCGAN
        d = 128
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 16*d, 4, 1, 0)

        self.fc_mu = nn.Linear(d*16, self.z_dim)
        self.fc_logvar = nn.Linear(d*16, self.z_dim)

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

    def reconstruct(self, imgs):
        """
        Doing a reconstruction
        :param imgs: [bsz, 3, 64, 64]
        :return:
            reconst: [bsz, 3, 64, 64]
        """
        mu, logvar = self._encode(imgs)
        reconst, _ = self._decode(mu, logvar)
        return reconst

    def sample_images(self, mu=None, logvar=None, batch_size=100):
        """
        Sample images from the model. If mu and logvar is given than sample from that.
        If not sample freely with batch_size
        :param mu: the mean of z. [bsz, Z_DIM]
        :param logvar: logvar of z. [bsz, Z_DIM]
        :param batch_size: an int
        :return: reconst [bsz, 3, 64, 64]
        """

        # sample from prior
        if mu is None:
            mu = Variable(torch.zeros(batch_size, self.z_dim), volatile=True)
            logvar = Variable(torch.zeros(batch_size, self.z_dim), volatile=True)
            if USE_CUDA:
                mu = mu.cuda()
                logvar = logvar.cuda()

        return self._decode(mu, logvar)

    def reconstruction_sample(self, imgs):
        mu, logvar = self._encode(imgs)
        reconst, _ = self._decode(mu, logvar)
        return reconst

    def get_constructed_by_latent(self, z):
        raise NotImplementedError


    def importance_inference(self, imgs, k=50):
        """
        Perform an importance weighted lower bound estimate.
        return a monte carlo estimate of KL(q(z|x) | p(z)), monte carlo estimate of E_q(z|x) log p(x|z),
        and lower bound of each sample
        :param imgs: the images
        :param k: number of sampling from posterior
        :return:
            mc_KL_q_p: [bsz, 1]
            mc_log_x_cond_z: [bsz, 1]
            lower_bounds: a list of lower bounds. [bsz, k]
        """
        mu, logvar = self._encode(imgs)
        sigma = torch.exp(0.5*logvar)

        # store log w of each sample from posterior
        lower_bounds = [] # used for backward

        # used for display average result
        mc_KL_q_p = 0
        mc_log_x_cond_z = 0
        for _ in range(k):
            # perform decoding
            reconst, eps = self._decode(mu, logvar)

            # log p(x | z)
            log_x_cond_z = imgs * torch.log(reconst) + (1-imgs) * torch.log(reconst)
            # sum over pixels and each channel
            log_x_cond_z = log_x_cond_z.view(log_x_cond_z.size(0), -1)
            log_x_cond_z = torch.sum(log_x_cond_z, dim=1)
            mc_log_x_cond_z += log_x_cond_z / k

            # log p(z) - log q(z | x) = 0.5*(-mu^2 - sigma^2 eps^2 - 2*mu*sigma*eps + \
            #                   log sigma^2 + eps^2)
            # this is the sample estimate of KL.
            log_prior_minus_pos = 0.5*(-mu.pow(2) - sigma.pow(2)*eps.pow(2) - 2*mu*sigma*eps + logvar + eps.pow(2))
            log_prior_minus_pos = torch.sum(log_prior_minus_pos, dim=1)
            KL_q_p = -log_prior_minus_pos
            mc_KL_q_p += KL_q_p / k

            # log w
            lower_bound = log_x_cond_z + log_prior_minus_pos
            lower_bounds.append(lower_bound)

        lower_bounds = torch.stack(lower_bounds, 1)
        return mc_KL_q_p, mc_log_x_cond_z, lower_bounds


    def inference(self, imgs):
        """
        perform an enc and dec, getting the lower bound, log p(x|z) and KL(q(z|x)|p(z)).
        Also return reconstruction.
        Use an analytical form of KL the same as original vae paper.
        :param imgs: [bsz, 3, 64, 64]
        :return:
            KL_q_p: [bsz, 1]
            log_x_cond_z: [bsz, 1]
            lower_bound: [bsz, 1] (log_x_cond_z - KL_q_p)
            reconst: [bsz, 3, 64, 64]
        """
        mu, logvar = self._encode(imgs)
        reconst, _ = self._decode(mu, logvar)

        # log p(x | z)
        log_x_cond_z = imgs * torch.log(reconst) + (1-imgs) * torch.log(reconst)
        # sum over pixels and each channel
        log_x_cond_z = log_x_cond_z.view(log_x_cond_z.size(0), -1)
        log_x_cond_z = torch.sum(log_x_cond_z, dim=1)

        # KL(q(z|x) | p(z))
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # sum over dimension
        KL_q_p = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # lower bound
        lower_bound = log_x_cond_z - KL_q_p
        return KL_q_p, log_x_cond_z, lower_bound, reconst

    def get_interpolate_images(self, imgs):
        mean, logvar = self._encode(imgs)

         # reparametric trick
        eps = torch.FloatTensor(mean.shape)
        eps.normal_()
        eps = Variable(eps)
        if USE_CUDA:
            eps = eps.cuda()
        z = mean + eps * torch.exp(0.5*logvar)
        z = z.view(z.size(0), self.z_dim, 1, 1)

        img_size = list(imgs.size())
        img_size[0] = 0
        reconstruct_z_img = np.empty(img_size)
        reconstruct_img = np.empty(img_size)

        z_0 = z[0,:,:,:].unsqueeze(0)
        z_1 = z[1,:,:,:].unsqueeze(0)

        im_0 = imgs[0,:,:,:].unsqueeze(0)
        im_1 = imgs[1,:,:,:].unsqueeze(0)
        for alpha in np.linspace(0,1,11):
            
            z_prime = float(alpha)*z_0+float((1-alpha))*z_1
            const = self.get_constructed_by_latent(z_prime)
            const = const.cpu().data.numpy()
            reconstruct_z_img = np.concatenate((reconstruct_z_img, const))

            im_prime = float(alpha)*im_0+float((1-alpha))*im_1
            im_prime = im_prime.cpu().data.numpy()
            reconstruct_img = np.concatenate((reconstruct_img, im_prime))

        return reconstruct_z_img, reconstruct_img




class VariationalAutoEncoder(VAEBase):
    """
    An VAE use dcgan decoder
    """
    def __init__(self, z_dim):
        super(VariationalAutoEncoder, self).__init__(z_dim)
        # https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_CelebA_DCGAN.py
        # decoder
        d = 128
        self.deconv1 = nn.ConvTranspose2d(self.z_dim, d*8, 4, 1, 0)
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

        # reparametric trick
        eps = torch.FloatTensor(mean.shape)
        eps.normal_()
        eps = Variable(eps)
        if USE_CUDA:
            eps = eps.cuda()
        z = mean + eps * torch.exp(0.5*logvar)
        z = z.view(z.size(0), self.z_dim, 1, 1)

        reconst = self.get_constructed_by_latent(z)

        return reconst, eps

    def get_constructed_by_latent(self, z):
        tmp = F.leaky_relu(self.deconv1_bn(self.deconv1(z)))
        tmp = F.leaky_relu(self.deconv2_bn(self.deconv2(tmp)))
        tmp = F.leaky_relu(self.deconv3_bn(self.deconv3(tmp)))
        tmp = F.leaky_relu(self.deconv4_bn(self.deconv4(tmp)))
        tmp = self.deconv5(tmp)

        return F.sigmoid(tmp)



class VariationalUpsampleEncoder(VAEBase):
    def __init__(self, mode='bilinear', z_dim=100):
        super(VariationalUpsampleEncoder, self).__init__(z_dim)

        d = 128
        self.up1 = nn.Upsample(scale_factor=8, mode=mode)
        self.deconv1 = nn.Conv2d(self.z_dim, d*8, 4, 2, 1)
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

        # reparametric trick
        eps = torch.FloatTensor(mean.shape)
        eps.normal_()
        eps = Variable(eps)
        if USE_CUDA:
            eps = eps.cuda()
        z = mean + eps * torch.exp(0.5*logvar)
        z = z.view(z.size(0), self.z_dim, 1, 1)

        reconst = self.get_constructed_by_latent(z)
        return reconst, eps

    def get_constructed_by_latent(self, z):
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
        return F.sigmoid(tmp)




if __name__ == '__main__':
    import numpy as np

    def test_vae(vae):
        num_params = 0
        for param in vae.parameters():
            num_params += np.prod([ sh for sh in param.shape])
        imgs = torch.rand([2, 3, 64, 64])

        if USE_CUDA:
            imgs = imgs.cuda()

        log_x_cond_z, kl, lower_bound, reconst = vae.inference(Variable(imgs))
        assert reconst.size(2) == 64
        assert log_x_cond_z.size(0) == 2
        assert kl.size(0) == 2
        assert lower_bound.size(0) == 2
        print("num_param: {}".format(num_params))

    vae_models = {'nearest': VariationalUpsampleEncoder(mode='nearest'),
                  'bilinear': VariationalUpsampleEncoder(mode='bilinear'),
                  'deconvolution': VariationalAutoEncoder()}

    for model in vae_models:
        if USE_CUDA:
            vae_models[model] = vae_models[model].cuda()
        test_vae(vae_models[model])

    # two different inference
    imgs = torch.rand([3, 3, 64, 64])
    if USE_CUDA:
        imgs = imgs.cuda()

    print("monte carlo KL should be close to analytic KL")
    kl, log_x_cond_z, lower_bound, reconst = vae_models['deconvolution'].inference(Variable(imgs))
    print("reconst_loss {}, kl {}, lower bound {}".format(
        -log_x_cond_z[0].data[0], kl[0].data[0], lower_bound[0].data[0]
    ))

    mc_kl, mc_log_x_cond_z, lower_bounds = vae_models['deconvolution'].importance_inference(Variable(imgs), k=50)
    monte_carlo_lower_bound = torch.mean(lower_bounds, dim=-1) # note this is not what happen in training
    print("reconst_loss {}, kl {}, lower bound {}".format(
        -mc_log_x_cond_z[0].data[0], mc_kl[0].data[0], monte_carlo_lower_bound[0].data[0]
    ))

    # test sampling
    vae_models['deconvolution'].sample_images()

    # test reconst
    reconst = vae_models['deconvolution'].reconstruct(Variable(imgs))

