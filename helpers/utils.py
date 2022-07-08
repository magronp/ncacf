#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import shutil
import numpy as np
import torch
import os
from joblib import Parallel, delayed


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def get_optimal_val_model_lambda(model, setting_list, variant_list, n_epochs, range_lW, range_lH=None):

    for setting in setting_list:
        for variant in variant_list:
            # Define the path where to perform validation
            path_val = 'outputs/' + setting + '/' + model + '/' + variant + '/'

            if variant == 'strict':
                get_optimal_val_model_lW(path_val, range_lW, n_epochs)
            else:
                get_optimal_val_model_lW_lH(path_val, range_lW, range_lH, n_epochs)

    return


def get_optimal_val_model_lW_lH(path_val, range_lW, range_lH, n_epochs):

    # Load the validation score for the pretrained models
    Nw, Nh = len(range_lW), len(range_lH)
    val_ndcg = np.zeros((Nw, Nh, n_epochs))
    for iW, lW in enumerate(range_lW):
        for iH, lH in enumerate(range_lH):
            path_ndcg = path_val + 'lW_' + str(lW) + '/lH_' + str(lH) + '/training.npz'
            val_ndcg[iW, iH, :] = np.load(path_ndcg)['val_ndcg'][:n_epochs] * 100

    # Get the optimal hyperparameters
    ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
    lW_opt, lH_opt = range_lW[ind_opt[0]], range_lH[ind_opt[1]]
    np.savez(path_val + 'hyperparams.npz', lW=lW_opt, lH=lH_opt)
    np.savez(path_val + 'val_ndcg.npz', val_ndcg=val_ndcg, range_lW=range_lW, range_lH=range_lH)

    # Get the optimal model / training log / (WMF) and copy it to the main parent folder
    path_opt = path_val + 'lW_' + str(lW_opt) + '/lH_' + str(lH_opt) + '/'
    files_opt = os.listdir(path_opt)
    for f in files_opt:
        shutil.copyfile(path_opt + f, path_val + f)

    return


def get_optimal_val_model_lW(path_val, range_lW, n_epochs):

    # Load the validation score for the pretrained models
    Nw = len(range_lW)
    val_ndcg = np.zeros((Nw, n_epochs))
    for iW, lW in enumerate(range_lW):
            path_ndcg = path_val + 'lW_' + str(lW) + '/training.npz'
            val_ndcg[iW, :] = np.load(path_ndcg)['val_ndcg'][:n_epochs] * 100

    # Get the optimal hyperparameters
    ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
    lW_opt = range_lW[ind_opt[0]]
    np.savez(path_val + 'hyperparams.npz', lW=lW_opt, lH=0.)
    np.savez(path_val + 'val_ndcg.npz', val_ndcg=val_ndcg, range_lW=range_lW)

    # Get the optimal model / training log / (WMF) and copy it to the main parent folder
    path_opt = path_val + 'lW_' + str(lW_opt) + '/'
    files_opt = os.listdir(path_opt)
    for f in files_opt:
        shutil.copyfile(path_opt + f, path_val + f)

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
def wpe_joint_ncf(count_i, pred_rat, w_mlp, h_mlp, lW, lH=0, alpha=2.0, epsilon=1e-6):

    loss = ((1 + alpha * torch.log(1 + count_i / epsilon)) * (pred_rat - count_i) ** 2).mean()\
           + lW * (w_mlp ** 2).mean() + lH * (h_mlp ** 2).mean()

    return loss


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

# EOF

