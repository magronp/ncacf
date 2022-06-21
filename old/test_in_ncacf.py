#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder
import os
import time
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetPlaycounts
from helpers.utils import wpe_joint, init_ncacf, plot_grad_flow
from helpers.models import ModelNCACF
from helpers.eval import evaluate_uni_train

# Set random seed for reproducibility
np.random.seed(1234)

# Run on GPU (if it's available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Process on: {}'.format(torch.cuda.get_device_name(device)))

# Set parameters
params = {'batch_size': 8,
          'n_embeddings': 128,
          'n_epochs': 100,
          'lr': 1e-4,
          'n_features_hidden': 1024,
          'n_features_in': 168,
          'data_dir': 'data/',
          'device': 'cuda',
          'n_layers_di': 2
          }

variant = 'relaxed'
path_pretrain = 'outputs/pretraining_uni/' + variant + '/'
inter = 'conc'
out_act = 'relu'
params['out_dir'] = 'outputs/NCACFin/' + variant + '/' + inter + '/' + out_act + '/' + 'layers_' + str(params['n_layers_di']) + '/'
create_folder(params['out_dir'])

# Get the number of songs and users
n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

# Path for the TP training data, features and the WMF
path_tp_train = params['data_dir'] + 'train_tp.num.csv'
path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')

# Get the playcount data, confidence, and precompute its transpose
train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

# Define and initialize the model
my_model = ModelNCACF(n_users, n_songs_train, params['n_features_in'], params['n_features_hidden'],
                      params['n_embeddings'], params['n_layers_di'], variant, inter, out_act)

#my_model = init_ncacf(my_model, path_pretrain, variant=variant)
lamb_load = np.load(os.path.join(path_pretrain, 'hyperparams.npz'))
lW, lH = float(lamb_load['lW']), float(lamb_load['lH'])
my_model.requires_grad_(True)
my_model.to(params['device'])

# Training setup
my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
torch.autograd.set_detect_anomaly(True)

# Define the dataset
my_dataset = DatasetPlaycounts(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

# Loop over epochs
u_total = torch.arange(0, n_users, dtype=torch.long).to(params['device'])
time_tot, loss_tot, val_ndcg_tot = 0, [], []
time_opt, ndcg_opt = time_tot, 0
my_model.train()
for ep in range(params['n_epochs']):
    print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
    start_time_ep = time.time()
    epoch_losses = []
    for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
        my_optimizer.zero_grad()
        # Load the user and item indices and account for negative samples
        x = data[0].to(params['device'])
        count_i = data[1].to(params['device'])
        it = data[2].to(params['device'])
        # Forward pass
        pred_rat, w, h, h_con = my_model(u_total, x, it)
        # Back-propagation
        loss = wpe_joint(count_i, torch.transpose(pred_rat, 1, 0), w, h, h_con, lW, lH)
        loss.backward()
        clip_grad_norm_(my_model.parameters(), max_norm=1.)
        my_optimizer.step()
        plot_grad_flow(my_model.named_parameters())
        epoch_losses.append(loss.item())

    # Overall stats for one epoch
    loss_ep = np.mean(epoch_losses)
    loss_tot.append(loss_ep)
    time_ep = time.time() - start_time_ep
    time_tot += time_ep
    val_ndcg = evaluate_uni_train(params, my_model)
    val_ndcg_tot.append(val_ndcg)
    print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
          flush=True)

    # Save the model if it performs the best
    if val_ndcg > ndcg_opt:
        ndcg_opt = val_ndcg
        time_opt = time_tot
        torch.save(my_model, os.path.join(params['out_dir'], 'model.pt'))

# Record the training log
np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)


# EOF
