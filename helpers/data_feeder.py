#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.sparse


def load_tp_data(csv_file, setting='warm', alpha=2.0, epsilon=1e-6):

    # Get the number of users and songs in the eval set as well as the dataset for evaluation
    n_users = len(open('data/unique_uid.txt').readlines())
    n_songs_total = len(open('data/unique_sid.txt').readlines())

    # Load the TP data in csv format
    tp = pd.read_csv(csv_file)

    # Get the list of songs, user, and counts
    rows, cols = np.array(tp['uid'], dtype=np.int32), np.array(tp['sid'], dtype=np.int32)
    count = tp['count']

    # Build the sparse matrix
    sparse_tp = scipy.sparse.csr_matrix((count, (rows, cols)), dtype=np.int16, shape=(n_users, n_songs_total))

    # Now only retain the non-empty cols of the matrix (useful in the cold-start setting)
    if setting == 'cold':
        sparse_tp = sparse_tp[:, np.unique(cols)]

    # Binarize the playcounts
    sparse_tp.data = np.ones_like(sparse_tp.data)

    # Get the confidence
    conf = sparse_tp.copy()
    conf.data = alpha * np.log(1 + conf.data / epsilon)

    return sparse_tp, rows, cols, conf


class DatasetAttributes(Dataset):

    def __init__(self, wmf_path, features_path):

        # Acoustic content features
        features = pd.read_csv(features_path).to_numpy()
        # Sort according to the SID to ensure consistent feature/TP
        #features = features[features[:, 0].argsort()]
        #x = np.delete(features, 0, axis=1)
        self.datapoint2sid = features[:, 0]
        # And now remove the SID colmn
        x = np.delete(features, 0, axis=1)

        # WMF song attributes: check if the song is none
        if wmf_path is None:
            h = x
        else:
            h = np.load(wmf_path)['H']
        self.x = torch.Tensor(x).float()
        self.h = torch.Tensor(h).float()

        # Store the number of users and songs in the current subset
        self.n_users = len(np.unique(self.tp_data['uid']))
        self.n_songs = len(np.unique(self.tp_data['sid']))

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, item):
        return self.x[item, :], self.h[item, :], item


class DatasetPlaycounts(Dataset):

    def __init__(self, features_path, tp_path):

        # Acoustic content features
        features = pd.read_csv(features_path).to_numpy()
        # The first column is the (num) SID, so keep it as a mapping between data point and sid
        self.datapoint2sid = features[:, 0]
        #features = features[features[:, 0].argsort()]
        # And now remove the SID colmn
        x = np.delete(features, 0, axis=1)
        self.x = torch.tensor(x).float()

        # TP data
        self.tp_data = pd.read_csv(tp_path)
        # Also need to care about SIDs, as for cold-start these do not match: we want them to range from 0 to n_songs-1
        #self.tp_data['sid'] -= self.tp_data['sid'].min()

        # Store the number of users and songs in the current subset
        self.n_users = len(np.unique(self.tp_data['uid']))
        self.n_songs = len(np.unique(self.tp_data['sid']))

    def __len__(self):
        return self.n_songs

    def __getitem__(self, data_point):
        data_sid = self.datapoint2sid[data_point]
        u_pos = torch.tensor(self.tp_data[self.tp_data['sid'] == data_sid]['uid'].to_numpy(), dtype=torch.long)
        u_counts = torch.zeros(self.n_users)
        u_counts[u_pos] = 1
        return self.x[data_point, :], u_counts, data_point


# EOF
