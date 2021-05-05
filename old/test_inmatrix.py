#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import torch
from scipy import sparse
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributes, DatasetAttributesRatings
from torch.nn import Embedding, Module
from helpers.metrics import wpe_in, my_ndcg_out
from helpers.functions import init_weights, create_folder, init_model_joint, plot_grad_flow
from helpers.content_wmf import compute_factor_wmf_deep


def eval_mf_in(params, my_model):

    # Paths for features and TP
    path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    tp_path = os.path.join(params['data_dir'], 'train_tp.num.csv')

    # Get the number of users and songs in the eval set as well as the dataset for evaluation
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    n_songs = int(0.7 * n_songs_total)

    # Predict the attributes and ratings
    # Define a data loader
    my_dataset_eval = DatasetAttributes(wmf_path=None, features_path=path_features)
    my_dataloader_eval = DataLoader(my_dataset_eval, 1, shuffle=False, drop_last=False)

    # Compute the model output (predicted ratings)
    us_total = torch.arange(0, n_users).to(params['device'])
    pred_ratings = np.zeros((n_users, n_songs))
    my_model.eval()
    with torch.no_grad():
        for data in tqdm(my_dataloader_eval, desc='Computing predicted ratings', unit=' Songs'):
            pred = my_model(us_total, data[2].to(params['device']))
            pred_ratings[:, data[2]] = pred.cpu().detach().numpy().squeeze()

    # Load the evaluation subset true ratings
    eval_data, rows_eval, cols_eval, _ = load_tp_data(tp_path, shape=(n_users, n_songs_total))
    cols_eval -= cols_eval.min()
    eval_data = sparse.csr_matrix((eval_data.data, (rows_eval, cols_eval)), dtype=np.int16, shape=(n_users, n_songs))

    # Get the score
    ndcg_mean = my_ndcg_out(eval_data, pred_ratings, k=50)

    return ndcg_mean


class ModelMF(Module):

    def __init__(self, n_users, n_songs, n_embeddings):

        super(ModelMF, self).__init__()

        # embedding
        self.user_emb = Embedding(n_users, n_embeddings)
        self.item_emb = Embedding(n_songs, n_embeddings)

        self.user_emb.weight.data.uniform_(-1, 1)
        self.item_emb.weight.data.uniform_(-1, 1)

    def forward(self, u, i):

        # Get the factors
        w = self.user_emb(u)
        h = self.item_emb(i)

        # Interaction model
        pred_rat=torch.matmul(h, torch.transpose(w, 0, 1))

        return pred_rat


# Set random seed for reproducibility
np.random.seed(1234)

# Run on GPU (if it's available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Process on: {}'.format(torch.cuda.get_device_name(device)))

# Set parameters
params = {'batch_size': 128,
          'n_embeddings': 20,
          'n_epochs': 30,
          'n_wmf': 20,
          'lr': 1e-4,
          'data_dir': 'data/',
          'device': 'cuda',
          'out_dir': 'outputs/in/'
          }

create_folder(params['out_dir'])

# Get the number of songs and users
n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
n_songs_train = int(0.7 * n_songs_total)

# Path for the TP training data, features and the WMF
path_tp_train = params['data_dir'] + 'train_tp.num.csv'
path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')

# Get the playcount data, confidence, and precompute its transpose
train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))
confT = conf.T.tocsr()

'''
# WMF (ALS)
W, H = None, np.random.randn(n_songs_train, params['n_embeddings']).astype('float32') * 0.01
for iwmf in range(params['n_wmf']):
    W = compute_factor_wmf_deep(H, conf, 0)
    H = compute_factor_wmf_deep(W, confT, 0)
np.save('')
pred_wmf = W.dot(H.T)
print(my_ndcg_out(train_data, pred_wmf, k=50))
np.savez(params['out_dir'] + 'wmf.npz', W=W, H=H)
'''

# Define and initialize the model
my_model = ModelMF(n_users, n_songs_train, params['n_embeddings'])
#wmf_loader = np.load(params['out_dir'] + 'wmf.npz')
#my_model.user_emb = torch.nn.Embedding.from_pretrained(torch.tensor(wmf_loader['W']), freeze=False)
#my_model.item_emb = torch.nn.Embedding.from_pretrained(torch.tensor(wmf_loader['H']), freeze=False)
my_model.to(params['device'])

# Training setup
my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
torch.autograd.set_detect_anomaly(True)

# Define the dataset
my_dataset = DatasetAttributesRatings(path_features, path_tp_train, n_users)
my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

# Loop over epochs
u_total = torch.arange(0, n_users, dtype=torch.long).to(device)
time_tot, loss_tot, val_ndcg_tot = 0, [], []
my_model.train()
for ep in range(params['n_epochs']):
    print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
    start_time_ep = time.time()
    epoch_losses = []
    for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
        my_optimizer.zero_grad()
        # Load the user and item indices and account for negative samples
        count_i = data[1].to(device)
        it = data[2].to(device)
        # Forward pass
        pred_rat = my_model(u_total, it)
        # Back-propagation
        loss = wpe_in(count_i, pred_rat)
        loss.backward()
        clip_grad_norm_(my_model.parameters(), max_norm=1.)
        my_optimizer.step()
        epoch_losses.append(loss.item())

    # Overall stats for one epoch
    loss_ep = np.mean(epoch_losses)
    loss_tot.append(loss_ep)
    time_ep = time.time() - start_time_ep
    time_tot += time_ep
    val_ndcg = eval_mf_in(params, my_model)
    val_ndcg_tot.append(val_ndcg)
    print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg), flush=True)
