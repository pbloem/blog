import os, tqdm, random, pickle, math, sys

import torch
import torch as T

from torch import nn
import torch.nn.functional as F

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
EPS = 10e-7

def atanh(x):
    """
    Inverse tanh. Likely to be numerically unstable.

    :param x:
    :return:
    """
    return 0.5 * torch.log( (1+x) / (1-x))

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

    def __init__(self, zsize=32, colors=3):

        super().__init__()

        a, b, c = 16, 32, 128

        self.stack = nn.Sequential(
            nn.Conv2d(1, a, (3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(a, b, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(b, b, (3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(b, c, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c, c, (3, 3), padding=1), nn.ReLU(),
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
    def __init__(self, zsize=32, out_channels=1):
        super().__init__()

        a, b, c = 16, 32, 128

        self.stack = nn.Sequential(
            nn.Linear(zsize, c * 4 * 4), nn.ReLU(),
            Reshape((c, 4, 4)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(c, b, (3, 3), padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(b, a, (3, 3), padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(a, out_channels, (3, 3), padding=1)
        )

    def forward(self, z):

        return self.stack(z)

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
    out_channels = 1
    if (arg.rloss == 'gaussian' or arg.rloss=='laplace') and arg.scale is None:
        out_channels = 2

    print(f'out channels: {out_channels}')

    encoder = Encoder(zsize=arg.zsize, colors=C)
    decoder = Decoder(zsize=arg.zsize, out_channels=out_channels)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=list(encoder.parameters()) + list(decoder.parameters()))

    for epoch in range(arg.epochs):

        for i, (input, _) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

                # Prepare the input
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()

            # Forward pass
            zs = encoder(input)

            kloss = kl_loss(zs[:, :arg.zsize], zs[:, arg.zsize:])
            z = sample(zs[:, :arg.zsize], zs[:, arg.zsize:])

            out = decoder(z)

            # compute -log p per dimension
            if arg.rloss == 'xent': # binary cross-entropy (not a proper log-prob)

                rloss = F.binary_cross_entropy_with_logits(out, input, reduction='none')

            elif arg.rloss == 'bdist': #   xent + correction
                rloss = F.binary_cross_entropy_with_logits(out, input, reduction='none')

                za = out.abs()
                eza = (-za).exp()
                logpart =  - za.log() + (-eza + EPS).log1p() - (eza + EPS).log1p() # logarithm of the normalization constant

                rloss = rloss - logpart

            elif arg.rloss == 'gauss': # xent + correction
                if arg.scale is None:
                    means = T.sigmoid(out[:, :1, :, :])
                    vars  = F.softplus( out[:, 1:, :, :])

                    rloss = GAUSS_CONST + vars.log() + (1.0/(2.0 * vars.pow(2.0))) * (out - means).pow(2.0)
                else:
                    means = out[:, :1, :, :]
                    var = arg.scale

                    rloss =  GAUSS_CONST + ln(var) + (1.0/(2.0 * (var ** 2.0))) * (out - means).pow(2.0)
            elif arg.rloss == 'laplace':  # xent + correction
                if arg.scale is None:
                    means = T.sigmoid(out[:, :1, :, :])
                    vars  = F.softplus( out[:, 1:, :, :])

                    rloss = (2.0 * vars).log() + (1.0/vars) * (out - means).abs()
                else:
                    means = out[:, :1, :, :]
                    var = arg.scale

                    rloss = ln(2.0 * var) + (1.0/var) * (out - means).abs()

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

            zs = encoder(inputs)
            outputs = decoder(zs[:, :arg.zsize])[:, :c, :, :]
            outputs = T.sigmoid(outputs)

            plt.figure(figsize=(5, 2))

            for i in range(N):

                ax = plt.subplot(2, N, i+1)
                inp = inputs[i].permute(1, 2, 0).cpu().numpy()
                if c == 1:
                    inp = inp.squeeze()

                ax.imshow(inp, cmap='gray_r')

                if i == 0:
                    ax.set_title('input')
                plt.axis('off')

                ax = plt.subplot(2, N, N+i+1)

                outp = outputs[i].permute(1, 2, 0).cpu().numpy()
                if c == 1:
                    outp = outp.squeeze()

                ax.imshow(outp, cmap='gray_r')

                if i == 0:
                    ax.set_title('reconstruction')
                plt.axis('off')

            plt.tight_layout()

            plt.savefig(f'reconstruction.{arg.rloss}.{epoch:03}.png')


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
                        help="Depth of the VAE in blocks (in addition to the 3 default blocks). Each block halves the resolution in each dimension with a 2x2 maxpooling layer.",
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

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/pixel', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)