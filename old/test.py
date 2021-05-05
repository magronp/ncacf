#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributesNegsamp
from helpers.models import ModelNCACF, evaluate_uni
from helpers.metrics import wpe_joint
from helpers.functions import create_folder, init_model_joint, plot_grad_flow

torch.cuda.empty_cache()

# Set random seed for reproducibility
np.random.seed(1234)

# Run on GPU (if it's available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Process on: {}'.format(torch.cuda.get_device_name(device)))

# Set parameters
params = {'batch_size_deep': 1024,
          'n_embeddings': 128,
          'n_epochs': 10,
          'lr': 1e-4,
          'n_features_hidden': 1024,
          'n_features_in': 168,
          'neg_ratio': 2,
          'n_layers_di': 0,
          'data_dir': 'data/',
          'device': device,
           'out_dir': 'outputs/ncacf/test/'
          }

mod = 'relaxed'
inter = 'mult'
ndcg_joint = []
Nl = 0
#path_pretrain = 'outputs/pretraining/' + mod + '/'
path_pretrain = None

create_folder(params['out_dir'])

# Get the number of songs and users
n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

# Path for the TP training data, features and the WMF
path_tp_train = params['data_dir'] + 'train_tp.num.csv'
path_tp_neg = params['data_dir'] + 'train_tp_neg.num.npz'
path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')

# Get the playcount data, confidence, and precompute its transpose
train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

# Get the hyper parameters (lambda_H only relevant for the relaxed model)
lH = 0.0
hyper_loader = np.load('outputs/pretraining_hybrid/hyperparams.npz')
lW = float(hyper_loader['lW'])
if mod == 'relaxed':
    lH = float(hyper_loader['lH'])

# Define and initialize the model
my_model = ModelNCACF(n_users, n_songs_train, params['n_features_in'], params['n_features_hidden'],
                        params['n_embeddings'], params['n_layers_di'], mod, inter).to(params['device'])
if path_pretrain is not None:
    my_model = init_model_joint(my_model, path_pretrain, mod=mod)
my_model.to(params['device'])

# Training setup
my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
torch.autograd.set_detect_anomaly(True)

# Define the dataset
my_dataset = DatasetAttributesNegsamp(features_path=path_features, tp_path=path_tp_train, tp_neg=path_tp_neg,
                                        n_users=n_users, n_songs=n_songs_train)
my_dataloader = DataLoader(my_dataset, params['batch_size_deep'], shuffle=True, drop_last=True)

# Loop over epochs
total_time = 0
total_loss = []
val_ndcg_total = []
my_model.train()
for ep in range(params['n_epochs']):
    print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
    start_time_ep = time.time()
    epoch_losses = []
    for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
        my_optimizer.zero_grad()
        # Load the features and reshape to account for negative samples
        x_pos = data[0].to(params['device'])
        x_neg = data[1].to(params['device'])
        x_tot = torch.cat((x_pos, x_neg.reshape((params['batch_size_deep'] * params['neg_ratio'],
                                                 params['n_features_in']))))
        # Load the user and item indices and account for negative samples
        us_pos = data[2]
        u_tot = us_pos.repeat(params['neg_ratio'] + 1).to(params['device'])
        i_pos, i_neg = data[4], data[5]
        i_tot = torch.cat((i_pos, i_neg.reshape(-1))).to(params['device'])
        # Get the true ratings and generate the full list (account for negative samples)
        counts_pos = data[3]
        counts_tot = torch.cat((counts_pos, torch.zeros(counts_pos.shape[0] * params['neg_ratio']))).to(params['device'])
        # Forward pass
        pred_rat, w, h, h_con = my_model(u_tot, x_tot, i_tot)
        # Back-propagation
        loss = wpe_joint(counts_tot, pred_rat, w, h, h_con, lW, lH)
        loss.backward()
        clip_grad_norm_(my_model.parameters(), max_norm=1.)
        print(my_model.di_in[0].weight)
        plot_grad_flow(my_model.named_parameters())
        my_optimizer.step()
        epoch_losses.append(loss.item())

    # Overall stats for one epoch
    end_time_ep = time.time() - start_time_ep
    total_time += end_time_ep
    val_ndcg = evaluate_uni(params, my_model, split='val')
    val_ndcg_total.append(val_ndcg)
    total_loss.append(np.mean(epoch_losses))
    print('\nLoss: {l_:6.6f} | Time: {t:5.3f}'.format( l_=np.mean(epoch_losses), t=end_time_ep), flush=True)
    print(val_ndcg)

# EOF
