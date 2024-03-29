#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import torch
from tqdm import tqdm
import numpy as np
import os
from helpers.data_feeder import load_tp_data, DatasetAttributes
from torch.utils.data import DataLoader
import bottleneck as bn
from scipy import sparse


def predict_attributes(my_model, my_dataloader, n_songs, n_embeddings, device):

    my_model = my_model.to(device)
    # Compute the model output (predicted attributes)
    predicted_attributes = torch.zeros([n_songs, n_embeddings]).to(device)
    my_model.eval()
    with torch.no_grad():
        for data in tqdm(my_dataloader, desc='Computing predicted attributes', unit=' Songs'):
            predicted_attributes[data[2].to(device), :] = my_model(data[0].to(device))

    predicted_attributes = predicted_attributes.cpu().detach().numpy()

    return predicted_attributes


def evaluate_mf_hybrid_cold(params, W, my_model, split='val'):

    # Paths for features and TP
    path_features = os.path.join(params['data_dir'], split + '_feats.num.csv')
    path_tp_eval = os.path.join(params['data_dir'], split + '_tp.num.csv')

    # Define a data loader
    my_dataset_eval = DatasetAttributes(wmf_path=None, features_path=path_features)
    my_dataloader_eval = DataLoader(my_dataset_eval, params['batch_size'], shuffle=False, drop_last=False)

    # Predict attributes and binary playcounts
    n_songs = my_dataset_eval.n_songs
    pred_attributes = predict_attributes(my_model, my_dataloader_eval, n_songs, params['n_embeddings'], params['device'])
    pred_ratings = W.dot(pred_attributes.T)

    # Load the evaluation subset true ratings
    eval_data = load_tp_data(path_tp_eval, setting='cold')[0]

    # Get the score
    ndcg_mean = my_ndcg_cold(eval_data, pred_ratings, k=50)

    return ndcg_mean


def evaluate_mf_hybrid_warm(params, W, H, my_model, variant='relaxed', split='val'):

    # The attributes need to be predicted only for the 'strict' variant (since there is no H)
    if variant == 'strict':
        # Feature path - data loader
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')
        my_dataset_eval = DatasetAttributes(wmf_path=None, features_path=path_features)
        my_dataloader_eval = DataLoader(my_dataset_eval, params['batch_size'], shuffle=False, drop_last=False)

        # Predict attributes and binary playcounts
        n_songs = my_dataset_eval.n_songs
        pred_attributes = predict_attributes(my_model, my_dataloader_eval, n_songs, params['n_embeddings'], params['device'])
        pred_ratings = W.dot(pred_attributes.T)

    else:
        pred_ratings = W.dot(H.T)

    # Load playcount data
    train_data = load_tp_data(os.path.join(params['data_dir'], 'train_tp.num.csv'), setting='warm')[0]
    val_data = load_tp_data(os.path.join(params['data_dir'], 'val_tp.num.csv'), setting='warm')[0]

    # Get the score
    if split == 'val':
        ndcg_mean = my_ndcg_warm(val_data, pred_ratings, k=50, leftout_ratings=train_data)[0]
    else:
        test_data = load_tp_data(os.path.join(params['data_dir'], 'test_tp.num.csv'), setting='warm')[0]
        ndcg_mean = my_ndcg_warm(test_data, pred_ratings, k=50, leftout_ratings=train_data + val_data)[0]

    return ndcg_mean


def evaluate_mf_hybrid(params, W, H, my_model, setting='cold', variant='relaxed', split='val'):

    if setting == 'cold':
        ndcg_mean = evaluate_mf_hybrid_cold(params, W, my_model, split=split)
    else:
        ndcg_mean = evaluate_mf_hybrid_warm(params, W, H, my_model, variant=variant, split=split)

    return ndcg_mean


def evaluate_uni_cold(params, my_model, split='val'):

    # Paths for features and TP
    path_features = os.path.join(params['data_dir'], split + '_feats.num.csv')
    tp_path = os.path.join(params['data_dir'], split + '_tp.num.csv')

    # Define a data loader
    my_dataset_eval = DatasetAttributes(wmf_path=None, features_path=path_features)
    my_dataloader_eval = DataLoader(my_dataset_eval, params['batch_size'], shuffle=False, drop_last=False)

    # Get the number of users and songs
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs = my_dataset_eval.n_songs

    # Compute the model output (predicted ratings)
    us_total = torch.arange(0, n_users, dtype=torch.long).to(params['device'])
    pred_ratings = np.zeros((n_users, n_songs))
    it_inp = torch.tensor([-1], dtype=torch.long).to(params['device'])
    my_model.eval()
    with torch.no_grad():
        for data in tqdm(my_dataloader_eval, desc='Computing predicted ratings', unit=' Songs'):
            pred = my_model(us_total, data[0].to(params['device']), it_inp)[0]
            pred_ratings[:, data[2]] = pred.cpu().detach().numpy().squeeze()

    # Load the evaluation subset true ratings
    eval_data = load_tp_data(tp_path, setting='cold')[0]

    # Get the score
    ndcg_mean = my_ndcg_cold(eval_data, pred_ratings, k=50)

    return ndcg_mean


def evaluate_uni_warm(params, my_model, split='val'):

    # Paths for features
    path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Predict the attributes and ratings
    # Define a data loader
    my_dataset_eval = DatasetAttributes(wmf_path=None, features_path=path_features)
    my_dataloader_eval = DataLoader(my_dataset_eval, params['batch_size'], shuffle=False, drop_last=False)

    # Get the number of users and songs
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs = my_dataset_eval.n_songs

    # Compute the model output
    us_total = torch.arange(0, n_users, dtype=torch.long).to(params['device'])
    pred_ratings = np.zeros((n_users, n_songs))
    my_model.eval()
    with torch.no_grad():
        for data in tqdm(my_dataloader_eval, desc='Computing predicted ratings', unit=' Songs'):
            pred = my_model(us_total, data[0].to(params['device']), data[2].to(params['device']))[0]
            pred_ratings[:, data[2]] = pred.cpu().detach().numpy().squeeze()

    # Load playcount data
    train_data = load_tp_data(os.path.join(params['data_dir'], 'train_tp.num.csv'), setting='warm')[0]
    val_data = load_tp_data(os.path.join(params['data_dir'], 'val_tp.num.csv'), setting='warm')[0]

    # Get the score
    if split == 'val':
        ndcg_mean = my_ndcg_warm(val_data, pred_ratings, k=50, leftout_ratings=train_data)[0]
    else:
        test_data = load_tp_data(os.path.join(params['data_dir'], 'test_tp.num.csv'), setting='warm')[0]
        ndcg_mean = my_ndcg_warm(test_data, pred_ratings, k=50, leftout_ratings=train_data + val_data)[0]

    return ndcg_mean


def evaluate_uni(params, my_model, setting='cold', split='val'):

    if setting == 'cold':
        ndcg_mean = evaluate_uni_cold(params, my_model, split=split)
    else:
        ndcg_mean = evaluate_uni_warm(params, my_model, split=split)

    return ndcg_mean


# Generate of list of user indexes for each batch
def user_idx_generator(n_users, batch_users):
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


# NDCG for out-of-matrix prediction
def my_ndcg_cold(true_ratings, pred_ratings, batch_users=5000, k=None):

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
    # Replace 0s with Nans to take nanmean and avoid warnings
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
def my_ndcg_warm(true_ratings, pred_ratings, batch_users=5000, k=None, leftout_ratings=None):

    n_users, n_songs = true_ratings.shape
    predicted_ratings = np.copy(pred_ratings)

    # Remove predictions on the left-out ratings ('train' for validation, and 'train+val' for testing)
    if leftout_ratings is not None:
        item_idx = np.zeros((n_users, n_songs), dtype=bool)
        item_idx[leftout_ratings.nonzero()] = True
        predicted_ratings[item_idx] = -np.inf

    # Loop over user batches
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        # Take a batch
        true_ratings_batch = true_ratings[user_idx]
        pred_ratings_batch = predicted_ratings[user_idx, :]
        # Call the NDCG for the current batch (depending on k)
        # If k not specified, compute the whole (standard) NDCG instead of its truncated version NDCG@k
        if k is None:
            ndcg_curr_batch = my_ndcg_in_batch(true_ratings_batch, pred_ratings_batch)
        else:
            ndcg_curr_batch = my_ndcg_in_k_batch(true_ratings_batch, pred_ratings_batch, k)
        res.append(ndcg_curr_batch)

    # Stack and get mean and std over users
    ndcg = np.hstack(res)
    # Replace 0s with Nans to take nanmean and avoid warnings
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


def my_ndcg_in_k_batch(true_ratings_batch, pred_ratings_batch, k=100):

    n_users_currbatch = true_ratings_batch.shape[0]
    idx_topk_part = bn.argpartition(-pred_ratings_batch, k, axis=1)
    topk_part = pred_ratings_batch[np.arange(n_users_currbatch)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(n_users_currbatch)[:, np.newaxis], idx_part]

    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (true_ratings_batch[np.arange(n_users_currbatch)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum() for n in true_ratings_batch.getnnz(axis=1)])
    ndcg = dcg / (idcg + 1e-8)

    return ndcg

# EOF
