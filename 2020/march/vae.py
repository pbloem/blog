import os, tqdm, random, pickle, math, sys

import torch
import torch as T

from torch import nn
import torch.nn.functional as F
import torch.distributions as ds

import numpy as np

import torchvision

from torch.autograd import Variable
import torchvision.transforms as tfs

from argparse import ArgumentParser

from collections import defaultdict, Counter, OrderedDict

from tensorboardX import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

"""
Basic image VAE in torch, to test the impact of different reconstruction losses.

TODO:
- try on DAS
- Dump loss curves (train/test)
- Plot loss curves (train/test)

"""

def ln(x):
    return math.log(x, math.e)

GAUSS_CONST = 0.5 * ln(2.0 * math.pi)

def atanh(x):
    """
    Inverse tanh. Likely to be numerically unstable.

    :param x:
    :return:
    """
    return 0.5 * torch.log( (1+x) / (1-x))

def contains_nan(input):
    if (not isinstance(input, torch.Tensor)) and isinstance(input, Iterable):
        for i in input:
            if contains_nan(i):
                return True
        return False
    else:
        return bool(torch.isnan(input).sum() > 0)

def kl_loss(zmean, zlsig):
    b, l = zmean.size()

    kl = 0.5 * torch.sum(zlsig.exp() - zlsig + zmean.pow(2) - 1, dim=1)

    assert kl.size() == (b,)

    return kl

def sample(zmean, zlsig, eps=None):
    b, l = zmean.size()

    if eps is None:
        eps = torch.randn(b, l)
        if zmean.is_cuda:
            eps = eps.cuda()

    return zmean + eps * (zlsig * 0.5).exp()

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    axes.spines["right"].set_visible(False)
    axes.spines["top"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.spines["left"].set_visible(False)

    axes.get_xaxis().set_tick_params(which='both', top='off', bottom='off', labeltop='off', labelbottom='off')
    axes.get_yaxis().set_tick_params(which='both', left='off', right='off', labelleft='off', labelright='off')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Pass(nn.Module):
    def forward(self, input):
        return input

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view( (input.size(0),) + self.shape)

class Debug(nn.Module):
    """
    Executes a lambda function and then returns the input. Useful for debugging.
    """
    def __init__(self, lambd):
        super(Debug, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        self.lambd(x)
        return x

class Encoder(nn.Module):
    """
    Encoder for a VAE
    """

    def __init__(self, zsize=32, colors=3, bn=False):

        super().__init__()

        a, b, c = 16, 32, 128

        self.stack = nn.Sequential(
            nn.Conv2d(1, a, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(a) if bn else Pass(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(a, b, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(b, b, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(b) if bn else Pass(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(b, c, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c, c, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(c) if bn else Pass(),
            nn.MaxPool2d((2, 2)),
            # Debug(lambda x: print(x.size())),
            nn.Flatten(),
            nn.Linear(4 * 4 * c, 2 * zsize)
        )

    def forward(self, x):

        return self.stack(x)

class Decoder(nn.Module):
    """
    Decoder for a VAE
    """
    def __init__(self, zsize=32, out_channels=1, mult=1.0, bn=False):
        super().__init__()

        a, b, c = 16, 32, 128

        self.stack = nn.Sequential(
            nn.Linear(zsize, c * 4 * 4), nn.ReLU(),
            Reshape((c, 4, 4)),
            nn.BatchNorm2d(c) if bn else Pass(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(c, b, (3, 3), padding=1), nn.ReLU(),
            nn.ConvTranspose2d(b, b, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(b) if bn else Pass(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(b, a, (3, 3), padding=1), nn.ReLU(),
            nn.ConvTranspose2d(a, a, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(a) if bn else Pass(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(a, out_channels, (3, 3), padding=1)
        )

        self.mult = mult

    def forward(self, z):

        return self.mult * self.stack(z)


class Test(nn.Module):
    """
    """

    def __init__(self, out_channels=1, height=32, width=32):
        super().__init__()

        self.p = nn.Parameter(torch.randn(1, out_channels, height, width))


    def forward(self, x):

        b, c, h, w = x.size()
        oc = self.p.size(1)
        return self.p.expand(b, oc, h, w)

def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    ## Load the data
    if arg.task == 'mnist':
        transform = tfs.Compose([tfs.Pad(padding=2), tfs.ToTensor()])

        trainset = torchvision.datasets.MNIST(root=arg.data_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root=arg.data_dir, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 1, 32, 32

    elif arg.task == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=True,
                                                download=True, transform=tfs.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=False,
                                               download=True, transform=tfs.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 32, 32

    elif arg.task == 'cifar-gs':
        transform = tfs.Compose([tfs.Grayscale(), tfs.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 1, 32, 32

    elif arg.task == 'imagenet64':

        transform = tfs.Compose([tfs.ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root=arg.data_dir + os.sep + 'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data_dir + os.sep + 'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)

        C, H, W = 3, 64, 64

    else:
        raise Exception('Task {} not recognized.'.format(arg.task))

    ## Set up the model
    out_channels = C
    if (arg.rloss == 'gauss' or arg.rloss=='laplace' or arg.rloss=='signorm' or arg.rloss == 'siglaplace' or arg.rloss == 'beta') and arg.scale is None:
        out_channels = 2 * C

    print(f'out channels: {out_channels}')

    encoder = Encoder(zsize=arg.zsize, colors=C)
    decoder = Decoder(zsize=arg.zsize, out_channels=out_channels, mult=arg.mult)

    if arg.testmodel:
        decoder = Test(out_channels=out_channels, height=H, width=W)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=list(encoder.parameters()) + list(decoder.parameters()))

    if arg.esched  is not None:
        start, end = int(arg.esched[0] * arg.epochs), (arg.esched[1] * arg.epochs)
        slope = 1.0/(end-start)

    for epoch in range(arg.epochs):

        if arg.esched is not None:
            weight = (epoch - start) * slope
            weight = np.clip(weight, 0, 1)
        else:
            weight = 1.0

        for i, (input, _) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

                # Prepare the input
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()

            # Forward pass
            if not arg.testmodel:
                zs = encoder(input)

                kloss = kl_loss(zs[:, :arg.zsize], zs[:, arg.zsize:])
                z = sample(zs[:, :arg.zsize], zs[:, arg.zsize:])

                out = decoder(z)
            else:
                out = decoder(input)
                kloss = 0

            # compute -log p per dimension
            if arg.rloss == 'xent': # binary cross-entropy (not a proper log-prob)

                rloss = F.binary_cross_entropy_with_logits(out, input, reduction='none')

            elif arg.rloss == 'bdist': #   xent + correction
                rloss = F.binary_cross_entropy_with_logits(out, input, reduction='none')

                za = out.abs()
                eza = (-za).exp()

                # - np.log(za) + np.log1p(-eza + EPS) - np.log1p(eza + EPS)
                logpart = - (za + arg.eps).log() + (-eza + arg.eps).log1p() - (eza + arg.eps).log1p()

                rloss = rloss + weight * logpart

            elif arg.rloss == 'gauss': # xent + correction
                if arg.scale is None:
                    means = T.sigmoid(out[:, :c, :, :])
                    vars  = F.sigmoid(out[:, c:, :, :])

                    rloss = GAUSS_CONST + vars.log() + (1.0/(2.0 * vars.pow(2.0))) * (input - means).pow(2.0)
                else:
                    means = T.sigmoid(out[:, :c, :, :])
                    var = arg.scale

                    rloss = GAUSS_CONST + ln(var) + (1.0/(2.0 * (var * var))) * (input - means).pow(2.0)

            elif arg.rloss == 'mse':
                means = T.sigmoid(out[:, :c, :, :])
                rloss = (input - means).pow(2.0)

            elif arg.rloss == 'mae':
                means = T.sigmoid(out[:, :c, :, :])
                rloss = (input - means).abs()

            elif arg.rloss == 'laplace':  # xent + correction
                if arg.scale is None:
                    means = T.sigmoid(out[:, :c, :, :])
                    vars  = F.softplus( out[:, c:, :, :])

                    rloss = (2.0 * vars).log() + (1.0/vars) * (input - means).abs()
                else:
                    means = T.sigmoid(out[:, :c, :, :])
                    var = arg.scale

                    rloss = ln(2.0 * var) + (1.0/var) * (input - means).abs()

            elif arg.rloss == 'signorm':
                if arg.scale is None:

                    mus = out[:, :c, :, :]
                    sgs, lsgs  = T.exp(out[:, c:, :, :] * arg.varmult), out[:, c:, :, :] * arg.varmult

                else:
                    mus = out[:, :c, :, :]
                    sgs, lsgs = arg.scale, math.log(arg.scale)

                y = input

                lny = torch.log(y + arg.eps)
                ln1y = torch.log(1 - y + arg.eps)

                x = lny - ln1y

                rloss = lny + ln1y + lsgs + GAUSS_CONST + \
                        0.5 * (1.0 / (sgs * sgs + arg.eps)) * (x - mus) ** 2

            elif arg.rloss == 'siglaplace':

                if arg.scale is None:

                    mus = out[:, :c, :, :]
                    sgs, lsgs  = T.exp(out[:, c:, :, :] * arg.varmult), out[:, c:, :, :] * arg.varmult

                else:
                    mus = out[:, :c, :, :]
                    sgs, lsgs = arg.scale, math.log(arg.scale)

                y = input

                lny = torch.log(y + arg.eps)
                ln1y = torch.log(1 - y + arg.eps)

                x = lny - ln1y

                rloss = lny + ln1y + lsgs + math.log(2.0) + \
                        (x - mus).abs() / sgs

            elif arg.rloss == 'beta':


                mean  = T.sigmoid(out[:, :c, :, :])
                mult  = F.softplus(out[:, c:, :, :] + arg.beta_add) + (1.0/(mean + arg.eps)) + arg.eps

                alpha = mean * mult
                beta  = (1 - mean) * mult

                part = alpha.lgamma() + beta.lgamma() - (alpha + beta).lgamma()
                x = input

                rloss = - (alpha - 1) * (x + arg.eps).log() - (beta - 1) * (1 - x + arg.eps).log() + part

            else:
                raise Exception(f'reconstruction loss {arg.rloss} not recognized.')

            if contains_nan(rloss):
                if arg.rloss == 'beta':
                    print('part contains nan', contains_nan(part))

                    print('alpha contains nan', contains_nan(alpha))
                    print('beta  contains nan', contains_nan(beta))

                    print('log x contains nan', contains_nan((x + arg.eps).log()))
                    print('log (1-x)  contains nan', contains_nan((1 - x + arg.eps).log()))

                raise Exception('rloss contains nan')


            rloss = rloss.reshape(b, -1).sum(dim=1) # reduce
            loss  = (rloss + kloss).mean()

            opt.zero_grad()
            loss.backward()

            opt.step()

        with torch.no_grad():
            N = 5

            # Plot reconstructions

            inputs, _ = next(iter(testloader))

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            b, c, h, w = inputs.size()

            if not arg.testmodel:
                zs = encoder(inputs)
                res = decoder(zs[:, :arg.zsize])
            else:
                res = decoder(inputs)

            outputs = res[:, :c, :, :]
            means = T.sigmoid(outputs)

            samples = None

            if arg.rloss == 'signorm' and out_channels > c:
                means = res[:, :c, :, :]
                vars = res[:, c:, :, :] * arg.varmult

                dist = ds.Normal(means, vars)
                samples = T.sigmoid(dist.sample())
                means   = T.sigmoid(dist.mean)

            if arg.rloss == 'siglaplace' and out_channels > c:
                means = res[:, :c, :, :]
                vars = res[:, c:, :, :] * arg.varmult

                dist = ds.Laplace(means, vars)
                samples = T.sigmoid(dist.sample())
                means   = T.sigmoid(dist.mean)

            if arg.rloss == 'beta':

                mean  = T.sigmoid(res[:, :c, :, :])
                mult  = (res[:, c:, :, :] + arg.beta_add).exp() + (1.0/mean) + arg.eps

                alpha = mean * mult
                beta  = (1 - mean) * mult

                dist = ds.Beta(alpha, beta)
                samples = dist.sample()
                means   = dist.mean
                vars    = dist.variance

            plt.figure(figsize=(5, 4))

            for i in range(N):

                ax = plt.subplot(4, N, i+1)
                inp = inputs[i].permute(1, 2, 0).cpu().numpy()
                if c == 1:
                    inp = inp.squeeze()

                ax.imshow(inp, cmap='gray_r')

                if i == 0:
                    ax.set_title('input')
                plt.axis('off')

                ax = plt.subplot(4, N, N+i+1)

                outp = means[i].permute(1, 2, 0).cpu().numpy()
                if c == 1:
                    outp = outp.squeeze()

                ax.imshow(outp, cmap='gray_r')

                if i == 0:
                    ax.set_title('means/modes')
                plt.axis('off')

                if samples is not None: # plot samples

                    ax = plt.subplot(4, N, 2 * N + i + 1)

                    outp = samples[i].permute(1, 2, 0).detach().cpu().numpy()
                    if c == 1:
                        outp = outp.squeeze()

                    ax.imshow(outp, cmap='gray_r')

                    if i == 0:
                        ax.set_title('sampled')
                    plt.axis('off')

                if out_channels > c: # plot the variance (or other uncertainty)

                    ax = plt.subplot(4, N, 3 * N + i + 1)

                    outp = vars[i].permute(1, 2, 0).detach().cpu().numpy()
                    if c == 1:
                        outp = outp.squeeze()

                    ax.imshow(outp, cmap='copper')

                    if i == 0:
                        ax.set_title('var')
                    plt.axis('off')

            plt.tight_layout()
            plt.savefig(f'reconstruction.{arg.rloss}.{epoch:03}.png')

            if arg.zsize == 2: # latent space plot

                N = 2000
                # gather up first 200 batches into one big tensor
                numbatches = N // arg.batch_size
                images, labels = [], []
                for i, (ims, lbs) in enumerate(testloader):
                    images.append(ims)
                    labels.append(lbs)

                    if i > numbatches:
                        break

                images, labels = torch.cat(images, dim=0), torch.cat(labels, dim=0)

                imagesg = images
                if torch.cuda.is_available():
                    imagesg = imagesg.cuda()

                n, c, h, w = images.size()

                z = encoder(imagesg)
                latents = z[:, :2].data.detach().cpu()

                mn, mx = latents.min(), latents.max()
                size = 1.0 * (mx - mn) / math.sqrt(n)
                # Change 0.75 to any value between ~ 0.5 and 1.5 to make the digits smaller or bigger

                fig = plt.figure(figsize=(8, 8))

                # colormap for the images
                norm = mpl.colors.Normalize(vmin=0, vmax=9)
                cmap = mpl.cm.get_cmap('tab10')

                for i in range(n):
                    x, y = latents[i, 0:2]
                    l = labels[i]

                    im = images[i, :]
                    alpha_im = im.permute(1, 2, 0).detach().cpu().numpy()
                    color = cmap(norm(l))
                    color_im = np.asarray(color)[None, None, :3]
                    color_im = np.broadcast_to(color_im, (h, w, 3))
                    # -- To make the digits transparent we make them solid color images and use the
                    #    actual data as an alpha channel.
                    #    color_im: 3-channel color image, with solid color corresponding to class
                    #    alpha_im: 1-channel grayscale image corrsponding to input data

                    im = np.concatenate([color_im, alpha_im], axis=2)
                    plt.imshow(im, extent=(x, x + size, y, y + size))

                    plt.xlim(mn, mx)
                    plt.ylim(mn, mx)

                plt.savefig(f'latent.{arg.rloss}.{epoch:03}.png')




if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task: [mnist, cifar10].",
                        default='mnist', type=str)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=30, type=int)

    parser.add_argument("--part-sched",
                        dest="esched",
                        nargs=2,
                        help="Schedule for scaling the partition function. Linear from 0 to 1 between these two proportions of the "
                             "toal nr of epochs.",
                        default=None, type=float)

    parser.add_argument("--evaluate-every",
                        dest="eval_every",
                        help="Run an evaluation/sample every n epochs.",
                        default=1, type=int)

    parser.add_argument("--rloss",
                        dest="rloss",
                        help="reconstruction loss [xent, bdist, gaussian, laplace, beta]",
                        default='xent', type=str)

    parser.add_argument("--scale",
                        dest="scale",
                        help="Value of the scale parameter (variance for a gaussian, b for a laplace ditstribution). If "
                             "None, the network learns a separate perameter for each pixel.",
                        default=None, type=float)

    parser.add_argument("-d", "--vae-depth",
                        dest="vae_depth",
                        help="Depth of the VAE in blocks (in addition to the 3 default blocks). Each block halves the "
                             "resolution in each dimension with a 2x2 maxpooling layer.",
                        default=0, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

    parser.add_argument("-z", "--z-size",
                        dest="zsize",
                        help="Size of latent space.",
                        default=32, type=int)

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit on the number of instances seen per epoch (for debugging).",
                        default=None, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate.",
                        default=0.001, type=float)

    parser.add_argument("--eps",
                        dest="eps",
                        help="Epsilon for stability.",
                        default=10e-5, type=float)

    parser.add_argument("--beta-add",
                        dest="beta_add",
                        help="Value added to beta before exp.",
                        default=3.0, type=float)

    parser.add_argument("--varmult",
                        dest="varmult",
                        help="Multiplier for the variance logits.",
                        default=10e-5, type=float)

    parser.add_argument("--mult",
                        dest="mult",
                        help="Multiplier for the logits.",
                        default=0.01, type=float)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/pixel', type=str)

    parser.add_argument("--batch-norm",
                        dest="batch_norm",
                        help="Use batch normalization",
                        action='store_true')

    parser.add_argument("--testmodel",
                        dest="testmodel",
                        help="A model consisting of just a parameter per output dimension",
                        action='store_true')

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)