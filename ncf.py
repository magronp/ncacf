#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder, get_optimal_val_model_relaxed
from helpers.training import train_ncf_in
import os
import time
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributes, DatasetAttributesRatings, DatasetAttributesNegsamp
from helpers.utils import compute_factor_wmf_deep, wpe_hybrid_strict, wpe_joint, wpe_joint_ncf, wpe_joint_ncacfnew, wpe_joint_neg
from helpers.eval import evaluate_mf_hybrid, predict_attributes, evaluate_uni
from torch.nn import Module, ModuleList, Linear, Sequential, ReLU, Embedding, Sigmoid, Identity


class ModelMFuninocontent(Module):

    def __init__(self, n_users, n_songs, n_embeddings):

        super(ModelMFuninocontent, self).__init__()
        self.n_users = n_users
        # embedding layers and initialization
        self.item_emb = Embedding(n_songs, n_embeddings)
        self.user_emb = Embedding(n_users, n_embeddings)
        self.user_emb.weight.data.normal_(0, 0.01)
        self.item_emb.weight.data.normal_(0, 0.01)

    def forward(self, u, x, i):

        # Get the factors
        w = self.user_emb(u)
        h = self.item_emb(i)

        # Interaction model

        # Get the GMF output
        '''
        pred_rat = torch.matmul(h, torch.transpose(w, 0, 1))
        '''
        emb = w.unsqueeze(1) * h
        emb = emb.view(-1, emb.shape[-1])
        # feed to the output layers
        pred_rat = emb.sum(dim=-1)
        # Reshape as (n_users, batch_size)
        pred_rat = pred_rat.view(self.n_users, -1).transpose(0, 1)

        return pred_rat, w, h


class ModelGMF(Module):

    def __init__(self, n_users, n_songs, n_embeddings):

        super(ModelGMF, self).__init__()
        self.n_users = n_users
        # embedding layers and initialization
        self.item_emb = Embedding(n_songs, n_embeddings)
        self.user_emb = Embedding(n_users, n_embeddings)
        self.user_emb.weight.data.normal_(0, 0.01)
        self.item_emb.weight.data.normal_(0, 0.01)
        self.out_layer_gmf = Linear(n_embeddings, 1, bias=False)
        self.out_layer_gmf.weight.data.fill_(1)

    def forward(self, u, x, i):

        # Get the factors
        w = self.user_emb(u)
        h = self.item_emb(i)

        # Interaction model
        emb = w.unsqueeze(1) * h
        emb = emb.view(-1, emb.shape[-1])
        # feed to the output layers
        pred_rat = self.out_layer_gmf(emb)
        # Reshape as (n_users, batch_size)
        pred_rat = pred_rat.view(self.n_users, -1).transpose(0, 1)

        return pred_rat, w, h


class ModelNCF(Module):

    def __init__(self, n_users, n_songs, n_embeddings):

        super(ModelNCF, self).__init__()

        self.n_users = n_users

        # Same for the MLP part
        self.user_emb = Embedding(n_users, n_embeddings)
        self.item_emb = Embedding(n_songs, n_embeddings)
        self.user_emb.weight.data.data.normal_(0, 0.01)
        self.item_emb.weight.data.data.normal_(0, 0.01)

        # Deep interaction layers
        self.n_features_di_in = n_embeddings * 2

        # First create the intermediate layers
        self.di1 = Sequential(Linear(n_embeddings * 2, n_embeddings, bias=True), ReLU())
        self.di2 = Sequential(Linear(n_embeddings, n_embeddings // 2, bias=True), ReLU())

        # Output layers
        self.out_layer_mlp = Linear(n_embeddings // 2, 1, bias=False)
        self.out_layer_gmf = Linear(n_embeddings, 1, bias=False)
        self.out_layer_gmf.weight.data.fill_(0.5)
        self.out_layer_mlp.weight.data.fill_(0.5)

    def forward(self, u, x, i):

        # Get the user/item factors
        w = self.user_emb(u)
        h = self.item_emb(i)

        # Get the GMF output
        emb_gmf = w.unsqueeze(1) * h
        emb_gmf = emb_gmf.view(-1, emb_gmf.shape[-1])

        # Get the MLP output
        # Concatenate and reshape
        emb_mlp = torch.cat((w.unsqueeze(1).expand(*[-1, h.shape[0], -1]),
                             h.unsqueeze(0).expand(*[w.shape[0], -1, -1])), dim=-1)
        emb_mlp = emb_mlp.view(-1, self.n_features_di_in)
        # Deep interaction
        emb_mlp = self.di1(emb_mlp)
        emb_mlp = self.di2(emb_mlp)

        # feed to the output layers
        pred_rat = self.out_layer_gmf(emb_gmf) + self.out_layer_mlp(emb_mlp)
        # Reshape as (n_users, batch_size)
        pred_rat = pred_rat.view(self.n_users, -1)

        return pred_rat, w, h


class ModelMLP(Module):

    def __init__(self, n_users, n_songs, n_embeddings):
        super(ModelMLP, self).__init__()

        self.n_users = n_users

        # Same for the MLP part
        self.user_emb_mlp = Embedding(n_users, n_embeddings)
        self.item_emb_mlp = Embedding(n_songs, n_embeddings)
        self.user_emb_mlp.weight.data.data.normal_(0, 0.01)
        self.item_emb_mlp.weight.data.data.normal_(0, 0.01)

        # Deep interaction layers
        self.n_features_di_in = n_embeddings * 2

        # First create the intermediate layers
        self.di1 = Sequential(Linear(n_embeddings * 2, n_embeddings, bias=True), ReLU())
        self.di2 = Sequential(Linear(n_embeddings, n_embeddings // 2, bias=True), ReLU())

        # Output layer
        #self.out_layer = Sequential(Linear(n_embeddings // 2, 1, bias=False), Sigmoid())
        self.out_layer = Linear(n_embeddings // 2, 1, bias=False)
        self.out_layer.weight.data.fill_(1)

    def forward(self, u, x, i):
        # Get the user/item factors
        w_mlp = self.user_emb_mlp(u)
        h_mlp = self.item_emb_mlp(i)

        # Get the MLP output
        # Concatenate and reshape
        emb_mlp = torch.cat((w_mlp.unsqueeze(1).expand(*[-1, h_mlp.shape[0], -1]),
                             h_mlp.unsqueeze(0).expand(*[w_mlp.shape[0], -1, -1])), dim=-1)
        emb_mlp = emb_mlp.view(-1, self.n_features_di_in)

        # Reshape/flatten as (n_users * batch_size, n_embeddings)
        emb_mlp = emb_mlp.view(-1, self.n_features_di_in)

        # Deep interaction
        emb_mlp = self.di1(emb_mlp)
        emb_mlp = self.di2(emb_mlp)
        pred_rat = self.out_layer(emb_mlp)

        # Reshape as (n_users, batch_size)
        pred_rat = pred_rat.view(w_mlp.shape[0], -1)

        return pred_rat, w_mlp, h_mlp


def train_ncf(params, path_pretrain=None):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = params['data_dir'] + 'feats.num.csv'

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelNCF(n_users, n_songs_train, params['n_embeddings'])
    if not(path_pretrain is None):
        my_model.load_state_dict(torch.load(path_pretrain + 'model.pt'), strict=False)
    my_model.requires_grad_(True)
    my_model.to(params['device'])

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Define the dataset
    my_dataset = DatasetAttributesRatings(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
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
            # Load the user / item indices and (positive) counts
            counts_tot = torch.transpose(data[1], 1, 0).to(params['device'])
            it_batch = data[2].to(params['device'])
            # Forward pass
            pred_rat, w_mlp, h_mlp = my_model(u_total, None, it_batch)
            # Back-propagation
            loss = wpe_joint_ncf(counts_tot, pred_rat, w_mlp, h_mlp, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())
        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, in_out='in', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model.state_dict(), os.path.join(params['out_dir'], 'model.pt'))

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_ncf_negsamp(params, neg_ratio=5):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = params['data_dir'] + 'feats.num.csv'
    path_tp_train_neg = params['data_dir'] + 'train_tp_neg.num.npz'

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelNCF(n_users, n_songs_train, params['n_embeddings'])
    my_model.requires_grad_(True)
    my_model.to(params['device'])

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Define the dataset
    my_dataset = DatasetAttributesNegsamp(features_path=path_features, tp_path=path_tp_train,
                                          tp_neg=path_tp_train_neg, n_users=n_users, n_songs=n_songs_train)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Predefine the list of index to select after forward pass
    idx_neg, idx_pos = torch.arange(1, neg_ratio), torch.tensor([0])
    idx_neg_tot, idx_pos_tot = idx_neg, idx_pos
    for nn in range(params['batch_size'] - 1):
        idx_neg = idx_neg + (params['batch_size'] + 1) * neg_ratio
        idx_pos = idx_pos + (params['batch_size'] + 1) * neg_ratio
        idx_neg_tot = torch.cat((idx_neg_tot, idx_neg), -1)
        idx_pos_tot = torch.cat((idx_pos_tot, idx_pos), -1)

    # Loop over epochs
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()
        epoch_losses = []
        for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
            my_optimizer.zero_grad()
            # Load the user / item indices and (positive) counts
            count_pos = data[3].to(params['device'])
            it_pos = data[4]
            it_neg = data[5]
            it = torch.cat((it_pos.view(it_pos.shape[0], -1), it_neg), -1).view(-1).to(params['device'])
            ut = data[2].to(params['device'])
            # Forward pass
            pred_rat, w_mlp, h_mlp = my_model(ut, None, it)
            # Select the positive and negative predictions
            pred_rat = pred_rat.view(-1)
            pred_pos, pred_neg = pred_rat[idx_pos_tot], pred_rat[idx_neg_tot]
            # Back-propagation
            loss = wpe_joint_neg(count_pos, pred_pos, pred_neg, w_mlp, h_mlp, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, in_out='in', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model.state_dict(), os.path.join(params['out_dir'], 'model.pt'))

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_mf_uni_nocontent(params):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelMFuninocontent(n_users, n_songs_train, params['n_embeddings'])
    my_model.requires_grad_(True)
    my_model.to(params['device'])

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Define the dataset
    my_dataset = DatasetAttributesRatings(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
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
            count_i = data[1].to(params['device'])
            it = data[2].to(params['device'])
            # Forward pass
            pred_rat, w, h = my_model(u_total, None, it)
            # Back-propagation
            loss = wpe_joint_ncf(count_i, pred_rat, w, h, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, in_out='in', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model.state_dict(), os.path.join(params['out_dir'], 'model.pt'))

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_main_ncf(params, range_lW, range_lH, data_dir='data/', path_pretrain=None):

    val_b = not(len(range_lW) == 1 and len(range_lW) == 1)

    path_current = 'outputs/in/ncf/'
    params['data_dir'] = data_dir + 'in/'
    # Training with grid search on the hyperparameters
    if val_b:
        for lW in range_lW:
            for lH in range_lH:
                print(lW, lH)
                params['lW'], params['lH'] = lW, lH
                params['out_dir'] = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/'
                create_folder(params['out_dir'])
                train_ncf(params, path_pretrain)
        get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])
    else:
        params['lW'], params['lH'] = range_lW[0], range_lH[0]
        params['out_dir'] = path_current
        create_folder(params['out_dir'])
        train_ncf(params, path_pretrain)
    return


def train_main_mf_uni_nocontent(params, range_lW, range_lH, data_dir='data/'):

    val_b = not(len(range_lW) == 1 and len(range_lW) == 1)

    path_current = 'outputs/in/gmf_nocontent/'
    params['data_dir'] = data_dir + 'in/'
    # Training with grid search on the hyperparameters
    if val_b:
        for lW in range_lW:
            for lH in range_lH:
                print(lW, lH)
                params['lW'], params['lH'] = lW, lH
                params['out_dir'] = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/'
                create_folder(params['out_dir'])
                train_mf_uni_nocontent(params)
        get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])
    else:
        params['lW'], params['lH'] = range_lW[0], range_lH[0]
        params['out_dir'] = path_current
        create_folder(params['out_dir'])
        train_mf_uni_nocontent(params)
    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(torch.cuda.get_device_name(device)))

    # Set parameters
    params = {'batch_size': 8,
              'n_embeddings': 128,
              'n_epochs': 50,
              'lr': 1e-4,
              'device': device
              }

    data_dir = 'data/'
    # Training and validation for the hyperparameters
    #range_lW, range_lH = [0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10]
    #range_lW, range_lH = [0.1], [0.1]
    #train_main_ncf(params, range_lW, range_lH, data_dir)

    range_lW, range_lH = [0.1], [0.1]
    train_main_mf_uni_nocontent(params, range_lW, range_lH, data_dir)
    train_main_ncf(params, range_lW, range_lH, data_dir, path_pretrain='outputs/in/gmf_nocontent/')

# EOF
