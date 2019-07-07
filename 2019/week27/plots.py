import torch, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch.nn.functional as F
import utils

import tqdm

plt.rcParams["font.family"] = "Georgia"

MAX = 60
VARS = np.linspace(0.1, MAX, 100)
REPS = 40

WIDTH = 8

def basic(axes=None):
    """
    Clean axes

    :param axes:
    :return:
    """

    if axes is None:
        axes = plt.gca()

    axes.spines["right"].set_visible(False)
    axes.spines["top"].set_visible(False)
    axes.spines["bottom"].set_visible(True)
    axes.spines["left"].set_visible(True)

    axes.get_xaxis().set_tick_params(which='both', top='off', bottom='on', labelbottom='on')
    axes.get_yaxis().set_tick_params(which='both', left='on', right='off')

def sample(nindices=2*256+2*8, size=(256, 256), var=1.0):
    assert len(size) == 2

    indices = (torch.rand(nindices, 2) * torch.tensor(size)[None, :].float()).long()
    values = torch.randn(nindices) * var

    sp = torch.sparse.FloatTensor(indices.t(), values, size)
    sp = sp.coalesce()

    return sp._indices(), sp._values()

def softmax_naive(indices, values, size):
    evals = values.exp()

    sums = utils.sum(indices, evals, size)
    return evals / sums

def dense(indices, values, size):
    dense = torch.zeros(*size) + float('-inf')

    for k in range(indices.size(1)):
        i, j = indices[:, k]
        dense[i, j] = values[k]

    return dense

def undense(dense, indices, size):
    values = torch.zeros(indices.size(1))

    for k in range(indices.size(1)):
        i, j = indices[:, k]
        values[k] = dense[i, j]

    return values

def maxdiff(indices, values, size):
    evals = values.exp()
    sums = utils.sum(indices, evals, size)
    diff = (1.0 - sums).abs()

    return diff.max()

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

def contains_inf(tensor):
    return bool((tensor == float('inf')).sum() > 0)

size = (1000, 1000)

naive_nans = np.zeros((len(VARS), REPS))
pnorm_nans = np.zeros((len(VARS), REPS))
pnorf_nans = np.zeros((len(VARS), REPS))
iters_nans = np.zeros((len(VARS), REPS))
iterf_nans = np.zeros((len(VARS), REPS))

naive_rmse = np.zeros((len(VARS), REPS))
pnorm_rmse = np.zeros((len(VARS), REPS))
pnorf_rmse = np.zeros((len(VARS), REPS))
iters_rmse = np.zeros((len(VARS), REPS))
iterf_rmse = np.zeros((len(VARS), REPS))

naive_diff = np.zeros((len(VARS), REPS))
pnorm_diff = np.zeros((len(VARS), REPS))
pnorf_diff = np.zeros((len(VARS), REPS))
iters_diff = np.zeros((len(VARS), REPS))
iterf_diff = np.zeros((len(VARS), REPS))

pnorm_maxdiff = np.zeros((len(VARS), REPS))
pnorf_maxdiff = np.zeros((len(VARS), REPS))
iters_maxdiff = np.zeros((len(VARS), REPS))
iterf_maxdiff = np.zeros((len(VARS), REPS))

first_nan = [len(VARS)] * 5
first_inf = [len(VARS)] * 5

PNP = 4
PNFP = 40

ITP = 2
ITS = 20

ITFP = 1.1
ITFS = 20

for v, var in tqdm.tqdm(enumerate(VARS)):
    for r in range(REPS):

        indices, values = sample(6000, size, var=var)
        dns = dense(indices, values, size)

        gold = undense(torch.softmax(dns, dim=1), indices, size)

        naive = softmax_naive(indices, values, size)
        pnorm = utils.logsoftmax(indices, values, size, max_method='pnorm', p=PNP)
        pnorf = utils.logsoftmax(indices, values, size, max_method='pnorm', p=PNFP)
        iters = utils.logsoftmax(indices, values, size, max_method='iteration', p=ITP, its=ITS)
        iterf = utils.logsoftmax(indices, values, size, max_method='iteration', p=ITFP, its=ITFS)

        if contains_nan(naive):
            naive_nans[v, r] = 1.0
            first_nan[0] = min(first_nan[0], v)
        else:
            naive_rmse[v, r] = (naive - gold).pow(2).sum().div(values.size(0)).sqrt()
            naive_diff[v, r] = maxdiff(indices, naive.log(), size)

        if contains_nan(pnorm):
            pnorm_nans[v, r] = 1.0
            first_nan[1] = min(first_nan[1], v)
        else:
            pnorm_rmse[v, r] = (pnorm.exp() - gold).pow(2).sum().div(values.size(0)).sqrt()
            pnorm_diff[v, r] = maxdiff(indices, pnorm, size)

        if contains_nan(pnorf):
            pnorf_nans[v, r] = 1.0
            first_nan[2] = min(first_nan[2], v)
        else:
            pnorf_rmse[v, r] = (pnorf.exp() - gold).pow(2).sum().div(values.size(0)).sqrt()
            pnorf_diff[v, r] = maxdiff(indices, pnorf, size)

        if contains_nan(iters):
            iters_nans[v, r] = 1.0
            first_nan[3] = min(first_nan[3], v)
        else:
            iters_rmse[v, r] = (iters.exp() - gold).pow(2).sum().div(values.size(0)).sqrt()
            iters_diff[v, r] = maxdiff(indices, iters, size)

        if contains_nan(iterf):
            iterf_nans[v, r] = 1.0
            first_nan[4] = min(first_nan[4], v)
        else:
            iterf_rmse[v, r] = (iterf.exp() - gold).pow(2).sum().div(values.size(0)).sqrt()
            iterf_diff[v, r] = maxdiff(indices, iterf, size)

        pnorm_max = utils.rowpnorm(indices, values, size, p=PNP)
        pnorf_max = utils.rowpnorm(indices, values, size, p=PNFP)

        iters_max = utils.itmax(indices, values, size, p=ITP, its=ITS)
        iterf_max = utils.itmax(indices, values, size, p=ITFP, its=ITFS)
        maxes = undense(dns.max(dim=1, keepdim=True)[0].expand(*size), indices, size)

        if(contains_inf(pnorm_max)):
            first_inf[1] = min(first_inf[1], v)
        else:
            pnorm_maxdiff[v, r] = (pnorm_max - maxes).abs().mean()

        if (contains_inf(pnorf_max)):
            first_inf[2] = min(first_inf[2], v)
        else:
            pnorf_maxdiff[v, r] = (pnorf_max - maxes).abs().mean()

        if(contains_inf(iters_max)):
            first_inf[3] = min(first_inf[3], v)
        else:
            iters_maxdiff[v, r] = (iters_max - maxes).abs().mean()


        if(contains_inf(iterf_max)):
            first_inf[4] = min(first_inf[4], v)
        else:
            iterf_maxdiff[v, r] = (iterf_max - maxes).abs().mean()


c = ('#73a7d3', '#b13e26', '#d38473', '#677d00', '#acd373')

## Figure 1: Just the naive approach

plt.figure(figsize=(WIDTH, WIDTH/4))

ax1 = plt.gca()

ax1.bar(VARS, naive_nans.sum(axis=1) / REPS, color=c[0])

ax1.set_ylabel('prop. NaN')
ax1.set_xlabel('variance of non-sparse elements')

ax1.legend(frameon=False)

basic(ax1)

plt.tight_layout()

plt.savefig('plot-naive.svg')
plt.savefig('plot-naive.png')

## Figure 2: The naive together with the pnorm


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(WIDTH, WIDTH/2))
w = MAX/len(VARS)/3

ax1.bar(VARS - w * 1.5, naive_nans.sum(axis=1) / REPS, width=w, label='naive', color=c[0])
ax1.bar(VARS - w * .5, pnorm_nans.sum(axis=1) / REPS, width=w, label='p-norm, p={}'.format(PNP), color=c[1])
ax1.bar(VARS + w * 0.5, pnorm_nans.sum(axis=1) / REPS, width=w, label='p-norm, p={}'.format(PNFP), color=c[2])

ax1.set_ylabel('prop. NaN')

ax1.legend(frameon=False)
basic(ax1)

ax2.errorbar(x=VARS[:first_nan[0]], y=naive_diff[:first_nan[0], :].mean(axis=1), yerr=naive_diff[:first_nan[0], :].std(axis=1), label='naive', color=c[0])
ax2.errorbar(x=VARS[:first_nan[1]], y=pnorm_diff[:first_nan[1], :].mean(axis=1), yerr=pnorm_diff[:first_nan[1], :].std(axis=1), label='p-norm, p={}'.format(PNP), color=c[1])
ax2.errorbar(x=VARS[:first_nan[2]], y=pnorf_diff[:first_nan[2], :].mean(axis=1), yerr=pnorf_diff[:first_nan[2], :].std(axis=1), label='p-norm, p={}'.format(PNFP), color=c[2])

ax2.set_xlabel('variance of non-sparse elements')
ax2.set_ylabel('dev. from 1')
basic(ax2)

plt.tight_layout()

plt.savefig('plot-pnorm.svg')
plt.savefig('plot-pnorm.png')

## Figure 3: All three approaches

_, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(WIDTH, WIDTH))
w = MAX/len(VARS)/5

ax1.bar(VARS - w * 2.5, naive_nans.sum(axis=1) / REPS, width=w, label='naive', color=c[0])
ax1.bar(VARS - w * 1.5, pnorm_nans.sum(axis=1) / REPS, width=w, label='p-norm, p={}'.format(PNP), color=c[1])
ax1.bar(VARS - w * 0.5, pnorm_nans.sum(axis=1) / REPS, width=w, label='p-norm, p={}'.format(PNFP), color=c[2])
ax1.bar(VARS + w * 0.5, iters_nans.sum(axis=1) / REPS, width=w, label='iterative p=2', color=c[3])
ax1.bar(VARS + w * 1.5, iterf_nans.sum(axis=1) / REPS, width=w, label='iterative p=1.1', color=c[4])


ax1.set_ylabel('prop. NaN')

ax1.legend(frameon=False)
basic(ax1)

ax2.errorbar(x=VARS[:first_nan[0]], y=naive_diff[:first_nan[0], :].mean(axis=1), yerr=naive_diff[:first_nan[0], :].std(axis=1), label='naive', color=c[0])
ax2.errorbar(x=VARS[:first_nan[1]], y=pnorm_diff[:first_nan[1], :].mean(axis=1), yerr=pnorm_diff[:first_nan[1], :].std(axis=1), label='p-norm, p={}'.format(PNP), color=c[1])
ax2.errorbar(x=VARS[:first_nan[2]], y=pnorf_diff[:first_nan[2], :].mean(axis=1), yerr=pnorf_diff[:first_nan[2], :].std(axis=1), label='p-norm, p={}'.format(PNFP), color=c[2])
ax2.errorbar(x=VARS[:first_nan[3]], y=iters_diff[:first_nan[3], :].mean(axis=1), yerr=iters_diff[:first_nan[3], :].std(axis=1), label='iterative, p={}'.format(ITP), color=c[3])
ax2.errorbar(x=VARS[:first_nan[4]], y=iterf_diff[:first_nan[4], :].mean(axis=1), yerr=iterf_diff[:first_nan[4], :].std(axis=1), label='iterative, p={}'.format(ITFP), color=c[4])

ax2.set_ylabel('dev. from 1')
basic(ax2)

ax3.errorbar(x=VARS[:first_nan[0]], y=naive_rmse[:first_nan[0], :].mean(axis=1), yerr=naive_rmse[:first_nan[0], :].std(axis=1), label='naive', color=c[0])
ax3.errorbar(x=VARS[:first_nan[1]], y=pnorm_rmse[:first_nan[1], :].mean(axis=1), yerr=pnorm_rmse[:first_nan[1], :].std(axis=1), label='p-norm, p={}'.format(PNP), color=c[1])
ax3.errorbar(x=VARS[:first_nan[2]], y=pnorf_rmse[:first_nan[2], :].mean(axis=1), yerr=pnorf_rmse[:first_nan[2], :].std(axis=1), label='p-norm, p={}'.format(PNFP), color=c[2])
ax3.errorbar(x=VARS[:first_nan[3]], y=iters_rmse[:first_nan[3], :].mean(axis=1), yerr=iters_rmse[:first_nan[3], :].std(axis=1), label='iterative, p={}'.format(ITP), color=c[3])
ax3.errorbar(x=VARS[:first_nan[4]], y=iterf_rmse[:first_nan[4], :].mean(axis=1), yerr=iterf_rmse[:first_nan[4], :].std(axis=1), label='iterative, p={}'.format(ITFP), color=c[4])

ax3.set_ylabel('RMSE')
basic(ax3)

ax4.errorbar(x=VARS[:first_inf[1]], y=pnorm_maxdiff[:first_inf[1], :].mean(axis=1), yerr=pnorm_maxdiff[:first_inf[1], :].std(axis=1), label='p-norm, p={}'.format(PNP), color=c[1])
ax4.errorbar(x=VARS[:first_inf[2]], y=pnorf_maxdiff[:first_inf[2], :].mean(axis=1), yerr=pnorf_maxdiff[:first_inf[2], :].std(axis=1), label='p-norm, p={}'.format(PNFP), color=c[2])
ax4.errorbar(x=VARS[:first_inf[3]], y=iters_maxdiff[:first_inf[3], :].mean(axis=1), yerr=iters_maxdiff[:first_inf[3], :].std(axis=1), label='iterative, p={}'.format(ITP), color=c[3])
ax4.errorbar(x=VARS[:first_inf[4]], y=iterf_maxdiff[:first_inf[4], :].mean(axis=1), yerr=iterf_maxdiff[:first_inf[4], :].std(axis=1), label='iterative, p={}'.format(ITFP), color=c[4])

ax4.set_xlabel('variance of non-sparse elements')
ax4.set_ylabel('row max MAE')
basic(ax4)

plt.tight_layout()

plt.savefig('plot-all.svg')
plt.savefig('plot-all.png')
