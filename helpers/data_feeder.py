#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.sparse


def log_surplus_confidence_matrix(R, alpha, epsilon):
    # To construct the surplus confidence matrix, we need to operate only on
    # the nonzero elements.
    # This is not possible: C = alpha * np.log(1 + R / epsilon)
    C = R.copy()
    C.data = alpha * np.log(1 + C.data / epsilon)
    return C


def load_tp_data(csv_file, shape, alpha=2.0, epsilon=1e-6):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['uid'], dtype=np.int32), np.array(tp['sid'], dtype=np.int32)
    count = tp['count']
    sparse_tp = scipy.sparse.csr_matrix((count, (rows, cols)), dtype=np.int16, shape=shape)
    # Binarize the playcounts
    sparse_tp.data = np.ones_like(sparse_tp.data)
    # Get the confidence
    conf = sparse_tp.copy()
    conf.data = alpha * np.log(1 + conf.data / epsilon)
    return sparse_tp, rows, cols, conf


def load_tp_data_old(csv_file, shape, alpha=2.0, epsilon=1e-6):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['uid'], dtype=np.int32), np.array(tp['sid'], dtype=np.int32)
    count = tp['count']
    sparse_tp = scipy.sparse.csr_matrix((count, (rows, cols)), dtype=np.int16, shape=shape)
    # Binarize the playcounts
    sparse_tp.data = np.ones_like(sparse_tp.data)
    # Get the confidence from binarized playcounts directly
    conf = sparse_tp.copy()
    conf.data[conf.data == 0] = 0.01
    conf.data -= 1  # conf surplus (used in WMF implementation)
    return sparse_tp, rows, cols, conf


def load_tp_data_sparse(csv_file, n_users, n_songs):

    tp = pd.read_csv(csv_file)
    indices = torch.tensor([tp['uid'], tp['sid']], dtype=torch.long)
    values = torch.ones_like(torch.tensor(tp['count'], dtype=torch.float32))

    # Construct a sparse tensor
    tp_sparse_tens = torch.sparse.FloatTensor(indices, values, torch.Size([n_users, n_songs]))

    return tp_sparse_tens


class DatasetAttributes(Dataset):

    def __init__(self, wmf_path, features_path):

        # Acoustic content features
        features = pd.read_csv(features_path).to_numpy()
        features = features[features[:, 0].argsort()]
        x = np.delete(features, 0, axis=1)

        # WMF song attributes: check if the song is none
        if wmf_path is None:
            h = x
        else:
            h = np.load(wmf_path)['H']
        self.x = torch.Tensor(x).float()
        self.h = torch.Tensor(h).float()

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, item):
        return self.x[item, :], self.h[item, :], item


class DatasetAttributesNegsamp(Dataset):

    def __init__(self, features_path, tp_path, tp_neg, n_users, n_songs):

        # Acoustic content features
        features = pd.read_csv(features_path).to_numpy()
        features = features[features[:, 0].argsort()]
        x = np.delete(features, 0, axis=1)
        self.x = torch.tensor(x).float()

        # TP data
        tp_data = load_tp_data_sparse(tp_path, n_users, n_songs)
        tp_data = tp_data.coalesce()
        self.us, self.it = tp_data.indices()
        self.count = tp_data.values()

        # Also load the list of negative samples (=items) per user
        self.neg_items = torch.tensor(np.load(tp_neg)['neg_items'], dtype=torch.long)

    def __len__(self):
        return self.count.__len__()

    def __getnegitem__(self, us_pos):
        return self.neg_items[us_pos, :]

    def __getitem__(self, data_point):

        us_pos, it_pos, count_pos = self.us[data_point], self.it[data_point], self.count[data_point]
        it_neg = self.__getnegitem__(us_pos)

        return self.x[it_pos, :], self.x[it_neg, :], us_pos, count_pos, it_pos, it_neg


class DatasetAttributesRatings(Dataset):

    def __init__(self, features_path, tp_path, n_users):

        # Acoustic content features
        features = pd.read_csv(features_path).to_numpy()
        features = features[features[:, 0].argsort()]
        x = np.delete(features, 0, axis=1)
        self.x = torch.tensor(x).float()

        # TP data
        self.tp_data = pd.read_csv(tp_path)
        self.n_users = n_users

    def __len__(self):
        return self.x.__len__()

    def __getitem__(self, data_point):
        u_pos = torch.tensor(self.tp_data[self.tp_data['sid'] == data_point]['uid'].to_numpy(), dtype=torch.long)
        u_counts = torch.zeros(self.n_users)
        u_counts[u_pos] = 1
        return self.x[data_point, :], u_counts, data_point


class DatasetJointEval(Dataset):

    def __init__(self, path_features, n_users, n_songs):

        # Acoustic content features
        features = pd.read_csv(path_features).to_numpy()
        features = features[features[:, 0].argsort()]
        x = np.delete(features, 0, axis=1)
        self.x = torch.tensor(x).float()

        # List with all users and items in the set
        self.n_users = n_users
        self.n_songs = n_songs

    def __len__(self):
        return self.n_users * self.n_songs

    def __getitem__(self, data_point):

        us = data_point // self.n_songs
        it = data_point % self.n_songs

        return self.x[it, :], us, it

# EOF
