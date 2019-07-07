import numpy as np
import scipy.sparse as sp

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class SpecialSpmmFunction(torch.autograd.Function):
    """
        Special function for only sparse region backpropataion layer.

        source: PyGAT, https://github.com/Diego999/pyGAT
    """

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]

        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


spmm = SpecialSpmm()

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def logsoftmax(indices, values, size, its=10, p=2, max_method='iteration', row=True, cuda=torch.cuda.is_available()):
    """
    Row or column log-softmaxes a sparse matrix (using logsumexp trick)

    :param indices:
    :param values:
    :param size:
    :param row:
    :return:
    """

    epsilon = 0.00000001
    dv = 'cuda' if cuda else 'cpu'

    # relud = F.relu(values)
    if max_method == 'pnorm':
        maxes = rowpnorm(indices, values, size, p=p, cuda=cuda)
    elif max_method == 'iteration':
        maxes = itmax(indices, values, size,its=its, p=p, cuda=cuda)
    else:
        raise Exception('Max method {} not recognized'.format(max_method))

    # print('    mm', maxes.mean())

    mvalues = torch.exp(values - maxes)

    sums = sum(indices, mvalues, size, row=row)  # row/column sums]

    return mvalues.log() - sums.log()

def rowpnorm(indices, values, size, p, row=True, cuda=torch.cuda.is_available()):
    """
    Row or column p-norms a sparse matrix
    :param indices:
    :param values:
    :param size:
    :param row:
    :return:
    """
    dv = 'cuda' if cuda else 'cpu'

    pvalues = torch.pow(values, p)
    sums = sum(indices, pvalues, size, row=row)

    return torch.pow(sums, 1.0/p)

def itmax(indices, values, size, its=10, p=2, row=True, cuda=torch.cuda.is_available()):
    """
    Iterative computation of row max
    :param indices:
    :param values:
    :param size:
    :param p:
    :param row:
    :param cuda:
    :return:
    """

    dv = 'cuda' if cuda else 'cpu'

    # create an initial vector with all values made positive
    # weights = values - values.min()
    weights = F.softplus(values)
    weights = weights/sum(indices, weights, size)

    # iterate, weights converges to a one-hot vector
    for i in range(its):
        weights = weights.pow(p)

        sums = sum(indices, weights, size, row=row)  # row/column sums
        weights = weights/sums

    return sum(indices, values * weights, size, row=row)

def sum(indices, values, size, row=True, cuda=torch.cuda.is_available()):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries

    :return:
    """
    epsilon = 0.000000001
    dv = 'cuda' if cuda else 'cpu'

    if row:
        ones = torch.ones((size[1], 1), device=dv)
    else:
        ones = torch.ones((size[0], 1), device=dv)
        # transpose the matrix
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)

    sums = spmm(indices, values, size, ones)  # row/column sums
    sums = torch.index_select(sums.squeeze(), 0, indices[0, :]).squeeze() + epsilon

    return sums

#
# def sparsemult(use_cuda):
#     return SparseMultGPU.apply if use_cuda else SparseMultCPU.apply
#
# class SparseMultCPU(torch.autograd.Function):
#
#     """
#     Sparse matrix multiplication with gradients over the value-vector
#
#     Does not work with batch dim.
#     """
#
#     @staticmethod
#     def forward(ctx, indices, values, size, vector):
#
#         # print(type(size), size, list(size), intlist(size))
#         # print(indices.size(), values.size(), torch.Size(intlist(size)))
#
#         matrix = torch.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))
#
#         ctx.indices, ctx.matrix, ctx.vector = indices, matrix, vector
#
#         return torch.mm(matrix, vector.unsqueeze(1))
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_output = grad_output.data
#
#         # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices
#
#         i_ixs = ctx.indices[0,:]
#         j_ixs = ctx.indices[1,:]
#         output_select = grad_output.view(-1)[i_ixs]
#         vector_select = ctx.vector.view(-1)[j_ixs]
#
#         grad_values = output_select *  vector_select
#
#         grad_vector = torch.mm(ctx.matrix.t(), grad_output).t()
#         return None, Variable(grad_values), None, Variable(grad_vector)
#
# class SparseMultGPU(torch.autograd.Function):
#
#     """
#     Sparse matrix multiplication with gradients over the value-vector
#
#     Does not work with batch dim.
#     """
#
#     @staticmethod
#     def forward(ctx, indices, values, size, vector):
#
#         # print(type(size), size, list(size), intlist(size))
#
#         matrix = torch.cuda.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))
#
#         ctx.indices, ctx.matrix, ctx.vector = indices, matrix, vector
#
#         return torch.mm(matrix, vector.unsqueeze(1))
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_output = grad_output.data
#
#         # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices
#
#         i_ixs = ctx.indices[0,:]
#         j_ixs = ctx.indices[1,:]
#         output_select = grad_output.view(-1)[i_ixs]
#         vector_select = ctx.vector.view(-1)[j_ixs]
#
#         grad_values = output_select *  vector_select
#
#         grad_vector = torch.mm(ctx.matrix.t(), grad_output).t()
#         return None, Variable(grad_values), None, Variable(grad_vector)
