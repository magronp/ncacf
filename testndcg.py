#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import torch
from helpers.utils import create_folder
from torch.utils.data import DataLoader
from helpers.data_feeder import load_tp_data, DatasetAttributes
import os
from helpers.eval import predict_attributes
from helpers.utils import user_idx_generator, my_ndcg_in_k_batch
from helpers.utils import my_ndcg_in
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Run on GPU (if it's available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Process on: {}'.format(device))
create_folder('outputs/temp/')

# Set parameters
params = {'batch_size': 128,
          'n_embeddings': 128,
          'n_iter_wmf': 30,
          'n_epochs': 150,
          'lr': 1e-4,
          'n_features_hidden': 1024,
          'n_features_in': 168,
          'device': device}
data_dir = 'data/'
params['n_epochs'] = 150

# Define the list of splits
k_split = 0

# Define the data directory
params['data_dir'] = data_dir + 'warm' + '/split' + str(k_split) + '/'
n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

# Train and test
loadnmf = np.load('outputs/warm/2stages/lW_1/lH_1/wmf.npz')
W, H = loadnmf['W'], loadnmf['H']
pred_ratings = W.dot(H.T)

my_model = torch.load('outputs/warm/2stages/lW_1/lH_1/model_strict.pt', map_location=torch.device('cpu'))
# Feature path - data loader
path_features = os.path.join(params['data_dir'], 'feats.num.csv')
my_dataset = DatasetAttributes(wmf_path=None, features_path=path_features)
my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=False, drop_last=False)
# Predict attributes and binary playcounts
pred_attributes = predict_attributes(my_model, my_dataloader, n_songs_total, params['n_embeddings'], params['device'])
pred_ratings_torch = W.dot(pred_attributes.T)




# Load playcount data
train_data = load_tp_data(os.path.join(params['data_dir'], 'train_tp.num.csv'), setting='warm')[0]
val_data = load_tp_data(os.path.join(params['data_dir'], 'val_tp.num.csv'), setting='warm')[0]
test_data = load_tp_data(os.path.join(params['data_dir'], 'test_tp.num.csv'), setting='warm')[0]

test_data, test_rows, test_cols, _ = load_tp_data(os.path.join(params['data_dir'], 'test_tp.num.csv'), setting='warm')
val_data, val_rows, val_cols, _ = load_tp_data(os.path.join(params['data_dir'], 'val_tp.num.csv'), setting='warm')
train_data, train_rows, train_cols, _ = load_tp_data(os.path.join(params['data_dir'], 'train_tp.num.csv'), setting='warm')


# Get the score
#ndcg_val = my_ndcg_in(val_data, pred_ratings, k=50, leftout_ratings=train_data)[0]
#ndcg_test = my_ndcg_in(test_data, pred_ratings, k=50, leftout_ratings=train_data + val_data)[0]

plot_data = train_data + val_data + test_data
plt.hist(pred_ratings[plot_data.nonzero()])
