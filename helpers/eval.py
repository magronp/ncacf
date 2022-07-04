#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import torch
from tqdm import tqdm
import numpy as np
import os
from scipy import sparse
from helpers.utils import my_ndcg_cold, my_ndcg_in
from helpers.data_feeder import load_tp_data, DatasetAttributes
from torch.utils.data import DataLoader


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

    # Get the number of users and songs in the eval set as well as the dataset for evaluation
    """ 
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if split == 'val':
        n_songs = int(0.2 * n_songs_total)
    else:
        n_songs = int(np.ceil(0.1 * 0.8 * n_songs_total))
    """

    # Predict the attributes and ratings
    # Define a data loader
    my_dataset_eval = DatasetAttributes(wmf_path=None, features_path=path_features)
    my_dataloader_eval = DataLoader(my_dataset_eval, params['batch_size'], shuffle=False, drop_last=False)

    # Predict attributes and binary playcounts
    n_songs = my_dataset_eval.n_songs
    pred_attributes = predict_attributes(my_model, my_dataloader_eval, n_songs, params['n_embeddings'], params['device'])
    pred_ratings = W.dot(pred_attributes.T)

    # Sort the pred_ratings by corresponding SID order, using the data2sid
    data2sid = my_dataset_eval.datapoint2sid
    pred_ratings = pred_ratings[:, data2sid.argsort()]

    # Load the evaluation subset true ratings
    #eval_data, rows_eval, cols_eval, _ = load_tp_data(path_tp_eval, setting='cold')
    #cols_eval -= cols_eval.min()
    #eval_data = sparse.csr_matrix((eval_data.data, (rows_eval, cols_eval)), dtype=np.int16, shape=(n_users, n_songs))
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

        # Sort the pred_ratings by corresponding SID order, using the data2sid
        data2sid = my_dataset_eval.datapoint2sid
        pred_ratings = pred_ratings[:, data2sid.argsort()]

    else:
        pred_ratings = W.dot(H.T)

    # Load playcount data
    train_data = load_tp_data(os.path.join(params['data_dir'], 'train_tp.num.csv'), setting='warm')[0]
    val_data = load_tp_data(os.path.join(params['data_dir'], 'val_tp.num.csv'), setting='warm')[0]

    # Get the score
    if split == 'val':
        ndcg_mean = my_ndcg_in(val_data, pred_ratings, k=50, leftout_ratings=train_data)[0]
    else:
        test_data = load_tp_data(os.path.join(params['data_dir'], 'test_tp.num.csv'), setting='warm')[0]
        ndcg_mean = my_ndcg_in(test_data, pred_ratings, k=50, leftout_ratings=train_data+val_data)[0]

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
    """
    # Get the number of users and songs in the eval set as well as the dataset for evaluation
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if split == 'val':
        n_songs = int(0.2 * n_songs_total)
    else:
        n_songs = int(np.ceil(0.1 * 0.8 * n_songs_total))
    """
    # Predict the attributes and ratings
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

    # Sort the pred_ratings by corresponding SID order, using the data2sid
    data2sid = my_dataset_eval.datapoint2sid
    pred_ratings = pred_ratings[:, data2sid.argsort()]

    # Load the evaluation subset true ratings
    eval_data = load_tp_data(tp_path, setting='cold')[0]

    # Get the score
    ndcg_mean = my_ndcg_cold(eval_data, pred_ratings, k=50)

    return ndcg_mean


def evaluate_uni_warm(params, my_model, split='val'):

    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    #n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

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

    # Sort the pred_ratings by corresponding SID order, using the data2sid
    data2sid = my_dataset_eval.datapoint2sid
    pred_ratings = pred_ratings[:, data2sid.argsort()]

    # Load playcount data
    train_data = load_tp_data(os.path.join(params['data_dir'], 'train_tp.num.csv'), setting='warm')[0]
    val_data = load_tp_data(os.path.join(params['data_dir'], 'val_tp.num.csv'), setting='warm')[0]

    # Get the score
    if split == 'val':
        ndcg_mean = my_ndcg_in(val_data, pred_ratings, k=50, leftout_ratings=train_data)[0]
    else:
        test_data = load_tp_data(os.path.join(params['data_dir'], 'test_tp.num.csv'), setting='warm')[0]
        ndcg_mean = my_ndcg_in(test_data, pred_ratings, k=50, leftout_ratings=train_data+val_data)[0]

    return ndcg_mean


def evaluate_uni(params, my_model, setting='cold', split='val'):

    if setting == 'cold':
        ndcg_mean = evaluate_uni_cold(params, my_model, split=split)
    else:
        ndcg_mean = evaluate_uni_warm(params, my_model, split=split)

    return ndcg_mean


# EOF
