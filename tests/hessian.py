import torch

from torch import nn
import torch.nn.functional as F


import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader, sampler
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

import random, tqdm, sys, math, os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn import decomposition as dc

H = 16 # hidden layer
P = 28*28*H + H + H*10 + 10 # total nr of params
K = 25

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset, using a fixed permutation

    initial source: https://github.com/pytorch/vision/issues/168

    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self,  start, num, total, seed = 0):
        self.start = start
        self.num = num

        self.random = random.Random(seed)

        self.l = list(range(total))
        self.random.shuffle(self.l)

    def __iter__(self):

        return iter(self.l[self.start : self.start + self.num])

    def __len__(self):
        return self.num


def init():

    w1 = torch.FloatTensor(H, 28*28)
    torch.nn.init.xavier_uniform_(w1, gain=torch.nn.init.calculate_gain('relu'))

    b1 = torch.FloatTensor(H).zero_()

    w2 = torch.FloatTensor(10, H)
    torch.nn.init.xavier_uniform_(w2) # gain=1

    b2 = torch.FloatTensor(10).zero_()

    return torch.cat([w1.view(-1), b1.view(-1), w2.view(-1), b2.view(-1)], dim=0)

def forward(x, params):

    b = x.size(0)

    s = 28*28*H

    w1 = params[:s].reshape(H, 28*28)
    b1 = params[s:s+H].reshape(1, H)

    s += H
    w2 = params[s:s+(H*10)].reshape(10, H)

    s += H*10
    b2 = params[s:].reshape(1, 10)


    h = torch.einsum('ij, bj -> bi', w1, x) + b1
    h = F.relu(h)

    o = torch.einsum('ij, bj -> bi ', w2, h) + b2

    return o

def go(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    tfms = transforms.Compose([transforms.ToTensor()])

    if (arg.task == 'mnist'):

        shape = (1, 28, 28)
        num_classes = 10

        data = arg.data + os.sep + arg.task

        if arg.final:
            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=tfms)
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch_size, shuffle=True, num_workers=2)

            test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=ToTensor())
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch_size, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=tfms)

            trainloader = DataLoader(train, batch_size=arg.batch_size, sampler=ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch_size, sampler=ChunkSampler(NUM_TRAIN, NUM_VAL, total))

    else:
        raise Exception('Task {} not recognized'.format(arg.task))

    parms = nn.Parameter(init())

    opt = torch.optim.Adam(lr=arg.lr, params=[parms])
    seen = 0

    for e in range(arg.epochs):

        for i, (inputs, labels) in enumerate(tqdm.tqdm(trainloader, 0)):

            b, c, h, w = inputs.size()
            seen += b

            # if arg.cuda:
            #     inputs, labels = inputs.cuda(), labels.cuda()



            if arg.mode == 'plain':

                opt.zero_grad()

                outputs = forward(inputs.view(b, -1), parms)

                loss = F.cross_entropy(outputs, labels)
                loss.backward()

                opt.step()
            elif arg.mode == 'pca':
                inputs = inputs.view(b, -1)

                gradients = []
                for bi in range(b): # compute separate gradient for each item in batch
                    opt.zero_grad()

                    outputs = forward(inputs[bi:bi+1], parms)

                    loss = F.cross_entropy(outputs, labels[bi:bi+1])
                    loss.backward()

                    gradients.append(parms.grad.data[None, :])

                gradients = torch.cat(gradients, dim=0)

                print(gradients.mean())

                b, p = gradients.size()

                pca = dc.PCA(n_components=K, whiten=False)
                pca.fit(gradients.cpu().numpy())

                # p.dot(X, self.components_) + self.mean_
                comps = torch.from_numpy(pca.components_)
                mean  = torch.from_numpy(pca.mean_)

                assert comps.size() == (K, P)

                # reparametrize the model
                pparms = nn.Parameter(torch.FloatTensor(K,).zero_())
                opt = torch.optim.Adam(lr=arg.lr, params=[pparms])
                opt.zero_grad()

                dparms = torch.matmul(pparms[None, :], comps) + mean + parms.data
                dparms = dparms.squeeze()

                assert dparms.size() == (P, ), f'{dparms.size()}, {P}'

                outputs = forward(inputs, dparms)

                loss = F.cross_entropy(outputs, labels)
                loss.backward()

                opt.step()

                parms.data = (torch.matmul(pparms[None, :], comps) + mean).squeeze().data

            else:
                raise Exception()

            tbw.add_scalar('hessian/loss', loss.item()/b, seen)

        # Compute accuracy on test set
        with torch.no_grad():
            if e % arg.test_every == 0:

                total, correct = 0.0, 0.0
                for input, labels in testloader:

                    # if arg.cuda:
                    #     input, labels = input.cuda(), labels.cuda()

                    output = forward(input.view(input.size(0), -1), parms)
                    outcls = output.argmax(dim=1)

                    total += outcls.size(0)
                    correct += (outcls == labels).sum().item()

                acc = correct / float(total)

                print(f'\nepoch {e}: test {acc:.4}', end='')
                tbw.add_scalar('hessian/test acc', acc, e)

                total, correct = 0.0, 0.0
                for input, labels in trainloader:

                    # if arg.cuda:
                    #     input, labels = input.cuda(), labels.cuda()

                    output = forward(input.view(input.size(0), -1), parms)
                    outcls = output.argmax(dim=1)

                    total += outcls.size(0)
                    correct += (outcls == labels).sum().item()

                acc = correct / float(total)

                print(f' train {acc:.4}')
                tbw.add_scalar('hessian/train acc', acc, e)


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=50, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-t", "--task", dest="task",
                        help="Dataset (mnist)",
                        default='mnist')

    parser.add_argument("--mode", dest="mode",
                        help="plain: Adam baseline, pca: linear subspace",
                        default='plain')

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-D", "--data-dir", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many epochs between testing for accuracy.",
                        default=1, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
