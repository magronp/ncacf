#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import numpy as np
import torch
import os
import bottleneck as bn
from scipy import sparse
from joblib import Parallel, delayed
from numba import jit
from matplotlib import pyplot as plt


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def get_optimal_val_model_relaxed(path_current, range_lW, range_lH, n_epochs):

    path_current = path_current + 'relaxed/'

    # Load the validation score for the pretrained models
    Nw, Nh = len(range_lW), len(range_lH)
    val_ndcg = np.zeros((Nw, Nh, n_epochs))
    for iW, lW in enumerate(range_lW):
        for iH, lH in enumerate(range_lH):
            path_ndcg = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/training.npz'
            val_ndcg[iW, iH, :] = np.load(path_ndcg)['val_ndcg'][:n_epochs] * 100

    # Get the optimal hyperparameters
    ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
    lW_opt, lH_opt = range_lW[ind_opt[0]], range_lH[ind_opt[1]]
    np.savez(path_current + 'hyperparams.npz', lW=lW_opt, lH=lH_opt)
    np.savez(path_current + 'val_ndcg.npz', val_ndcg=val_ndcg, range_lW=range_lW, range_lH=range_lH)

    # Get the optimal model / training log / (WMF) and copy it to the main parent folder
    path_opt = path_current + 'lW_' + str(lW_opt)+ '/lH_' + str(lH_opt) + '/'
    files_opt = os.listdir(path_opt)
    for f in files_opt:
        shutil.copyfile(path_opt + f, path_current + f)

    return


def get_optimal_val_model_strict(path_current, range_lW, n_epochs):

    path_current = path_current + 'strict/'

    # Load the validation score for the pretrained models
    Nw = len(range_lW)
    val_ndcg = np.zeros((Nw, n_epochs))
    for iW, lW in enumerate(range_lW):
            path_ndcg = path_current + 'lW_' + str(lW) + '/training.npz'
            val_ndcg[iW, :] = np.load(path_ndcg)['val_ndcg'][:n_epochs] * 100

    # Get the optimal hyperparameters
    ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
    lW_opt = range_lW[ind_opt[0]]
    np.savez(path_current + 'hyperparams.npz', lW=lW_opt, lH=0.)
    np.savez(path_current + 'val_ndcg.npz', val_ndcg=val_ndcg, range_lW=range_lW)

    # Get the optimal model / training log / (WMF) and copy it to the main parent folder
    path_opt = path_current + 'lW_' + str(lW_opt) + '/'
    files_opt = os.listdir(path_opt)
    for f in files_opt:
        shutil.copyfile(path_opt + f, path_current + f)

    return


def plot_val_ndcg_lW_lH(path_current):

    path_ndcg = path_current + 'val_ndcg.npz'

    # Load the overall validation NDCG
    ndcg_loader = np.load(path_ndcg)
    val_ndcg, range_lW, range_lH = ndcg_loader['val_ndcg'], ndcg_loader['range_lW'], ndcg_loader['range_lH']
    Nw = len(range_lW)
    n_epochs = val_ndcg.shape[-1]

    # Plot the results
    plt.figure()
    for il, l in enumerate(range_lW):
        plt.subplot(2, Nw // 2, il + 1)
        plt.plot(np.arange(n_epochs) + 1, val_ndcg[il, :, :].T)
        plt.title(r'$\lambda_W$=' + str(l))
        if il > 2:
            plt.xlabel('Epochs')
        if il == 0 or il == 3:
            plt.ylabel('NDCG (%)')
    leg_lambda = [r'$\lambda_H$=' + str(lh) for lh in range_lH]
    plt.legend(leg_lambda)
    plt.show()

    return


def plot_val_ndcg_lW(path_current):

    path_ndcg = path_current + 'val_ndcg.npz'

    # Load the overall validation NDCG
    ndcg_loader = np.load(path_ndcg)
    val_ndcg, range_lW = ndcg_loader['val_ndcg'], ndcg_loader['range_lW']
    Nw = len(range_lW)
    n_epochs = val_ndcg.shape[-1]

    # Plot the results
    plt.figure()
    for il, l in enumerate(range_lW):
        plt.subplot(2, Nw // 2, il + 1)
        plt.plot(np.arange(n_epochs) + 1, val_ndcg[il, :].T)
        plt.title(r'$\lambda_W$=' + str(l))
        if il > 2:
            plt.xlabel('Epochs')
        if il == 0 or il == 3:
            plt.ylabel('NDCG (%)')
    plt.show()

    return


def plot_grad_flow(named_parameters):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.0001, top=0.002)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.draw()
    plt.pause(0.001)
    plt.cla()

    return


# Weighted Prediction Error for the hybrid-strict algorithm (need to compute the predicted ratings)
def wpe_hybrid_strict(h_hat, W, count_i, alpha=2.0, epsilon=1e-6):

    pred_rat = torch.matmul(h_hat, torch.transpose(W, 0, 1))
    loss = ((1 + alpha * torch.log(1 + count_i / epsilon)) * (pred_rat - count_i) ** 2).mean()

    return loss


# Weighted Prediction Error (general formulation)
def wpe_joint(count_i, pred_rat, w, h, h_con, lW, lH=0, alpha=2.0, epsilon=1e-6):

    loss = ((1 + alpha * torch.log(1 + count_i / epsilon)) * (pred_rat - count_i) ** 2).mean() + lW * (w ** 2).mean() +\
           lH * ((h - h_con) ** 2).mean()

    return loss


# Weighted Prediction Error for the in-matrix recom
def wpe_in(count_i, pred_rat, alpha=2.0, epsilon=1e-6):
    loss = ((1 + alpha * torch.log(1 + count_i / epsilon)) * (pred_rat - count_i) ** 2).mean()
    return loss


# Generate of list of user indexes for each batch
def user_idx_generator(n_users, batch_users):
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


# NDCG for out-of-matrix prediction
def my_ndcg_out(true_ratings, pred_ratings, batch_users=5000, k=None):

    # Iterate over user batches
    res = list()
    for user_idx in user_idx_generator(true_ratings.shape[0], batch_users):
        true_ratings_batch = true_ratings[user_idx]
        pred_ratings_batch = pred_ratings[user_idx, :]
        if k is None:
            ndcg_curr_batch = my_ndcg_batch(true_ratings_batch, pred_ratings_batch)
        else:
            ndcg_curr_batch = my_ndcg_k_batch(true_ratings_batch, pred_ratings_batch, k)
        res.append(ndcg_curr_batch)

    # Get the mean and std NDCG over users
    ndcg = np.hstack(res)
    # Add artifial nans where there are 0s (to avoid warnings)
    ndcg[ndcg == 0] = np.nan
    ndcg_mean = np.nanmean(ndcg)

    return ndcg_mean


# NDCG for a given batch
def my_ndcg_batch(true_ratings, pred_ratings):

    all_rank = np.argsort(np.argsort(-pred_ratings, axis=1), axis=1)

    # build the discount template
    tp = 1. / np.log2(np.arange(2, true_ratings.shape[1] + 2))
    all_disc = tp[all_rank]

    # Binarize the true ratings
    true_ratings_bin = (true_ratings > 0).tocoo()

    # Get the disc
    disc = sparse.csr_matrix((all_disc[true_ratings_bin.row, true_ratings_bin.col],
                              (true_ratings_bin.row, true_ratings_bin.col)),
                             shape=all_disc.shape)

    # DCG, ideal DCG and normalized DCG
    dcg = np.array(disc.sum(axis=1)).ravel()
    idcg = np.array([tp[:n].sum() for n in true_ratings.getnnz(axis=1)])
    ndcg = dcg / (idcg + 1e-8)

    return ndcg


# NDCG at k for a given batch
def my_ndcg_k_batch(true_ratings, pred_ratings, k=100):

    n_users_currbatch = true_ratings.shape[0]
    idx_topk_part = bn.argpartition(-pred_ratings, k, axis=1)
    topk_part = pred_ratings[np.arange(n_users_currbatch)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(n_users_currbatch)[:, np.newaxis], idx_part]

    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (true_ratings[np.arange(n_users_currbatch)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum() for n in true_ratings.getnnz(axis=1)])
    ndcg = dcg / (idcg + 1e-8)

    return ndcg


# NDCG
def my_ndcg_in(true_ratings, pred_ratings, batch_users=5000, k=None, leftout_ratings=None):

    n_users, n_songs = true_ratings.shape

    # Remove predictions on the left-out ratings ('train' for validation, and 'train+val' for testing)
    if leftout_ratings is not None:
        item_idx = np.zeros((n_users, n_songs), dtype=bool)
        item_idx[leftout_ratings.nonzero()] = True
        pred_ratings[item_idx] = -np.inf

    # Loop over user batches
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        # Take a batch
        true_ratings_batch = true_ratings[user_idx]
        pred_ratings_batch = pred_ratings[user_idx, :]
        # Call the NDCG for the current batch (depending on k)
        # If k not specified, compute the whole (standard) NDCG instead of its truncated version NDCG@k
        if k is None:
            ndcg_curr_batch = my_ndcg_in_batch(true_ratings_batch, pred_ratings_batch)
        else:
            ndcg_curr_batch = my_ndcg_in_k_batch(true_ratings_batch, pred_ratings_batch, k)
        res.append(ndcg_curr_batch)

    # Stack and get mean and std over users
    ndcg = np.hstack(res)
    # Remove 0s (artifically added to avoid warnings)
    ndcg[ndcg == 0] = np.nan
    ndcg_mean = np.nanmean(ndcg)
    ndcg_std = np.nanstd(ndcg)

    return ndcg_mean, ndcg_std


def my_ndcg_in_batch(true_ratings, pred_ratings):

    all_rank = np.argsort(np.argsort(-pred_ratings, axis=1), axis=1)

    # build the discount template
    tp = 1. / np.log2(np.arange(2, true_ratings.shape[1] + 2))
    all_disc = tp[all_rank]

    # Binarize the true ratings
    true_ratings_bin = (true_ratings > 0).tocoo()

    # Get the disc
    disc = sparse.csr_matrix((all_disc[true_ratings_bin.row, true_ratings_bin.col],
                              (true_ratings_bin.row, true_ratings_bin.col)),
                             shape=all_disc.shape)

    # DCG, ideal DCG and normalized DCG
    dcg = np.array(disc.sum(axis=1)).ravel()
    idcg = np.array([tp[:n].sum() for n in true_ratings.getnnz(axis=1)])
    ndcg = dcg / (idcg + 1e-8)

    return ndcg


def my_ndcg_in_k_batch(true_ratings, pred_ratings, k=100):

    n_users_currbatch = true_ratings.shape[0]
    idx_topk_part = bn.argpartition(-pred_ratings, k, axis=1)
    topk_part = pred_ratings[np.arange(n_users_currbatch)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(n_users_currbatch)[:, np.newaxis], idx_part]

    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (true_ratings[np.arange(n_users_currbatch)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum() for n in true_ratings.getnnz(axis=1)])
    ndcg = dcg / (idcg + 1e-8)

    return ndcg


# ALS updates for the WMF model
def get_row(S, i):
    lo, hi = S.indptr[i], S.indptr[i + 1]
    return S.data[lo:hi], S.indices[lo:hi]


def solve_sequential(As, Bs):
    X_stack = np.empty_like(As, dtype=As.dtype)

    for k in range(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k], As[k])

    return X_stack


def solve_batch(b, S, Y, WX, YTYpR, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f), dtype=dtype)
    B_stack = np.empty((current_batch_size, f, f), dtype=dtype)

    for ib, k in enumerate(range(lo, hi)):
        s_u, i_u = get_row(S, k)

        Y_u = Y[i_u]  # exploit sparsity
        A = (s_u + 1).dot(Y_u)

        if WX is not None:
            A += WX[:, k]

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B

    X_stack = solve_sequential(A_stack, B_stack)
    return X_stack


def compute_factor_wmf_deep(Y, S, lambda_reg, content_prior=None, dtype='float32', batch_size=1000, n_jobs=-1):

    m = S.shape[0]  # m = number of users (or items if tranposed beforehand)
    f = Y.shape[1]  # f = number of factors
    num_batches = int(np.ceil(m / float(batch_size)))

    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpR = YTY + lambda_reg * np.eye(f)
    if content_prior is not None:
        Bz = lambda_reg * content_prior.T
    else:
        Bz = None

    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, Bz, YTYpR, batch_size, m, f, dtype)
                                  for b in range(num_batches))
    X_new = np.concatenate(res, axis=0)

    return X_new


# Adapted to python 3 with Jit
@jit
def inner_prod(W, H, rows, cols):
    n_ratings = rows.size
    n_components = W.shape[1]
    assert H.shape[1] == n_components
    data = np.empty(n_ratings)
    for i in range(n_ratings):
        data[i] = 0.0
        for j in range(n_components):
            data[i] += W[rows[i], j] * H[cols[i], j]
    return data

# EOF

