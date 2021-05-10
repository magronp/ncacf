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
from helpers.data_feeder import load_tp_data, DatasetAttributes, DatasetAttributesRatings
from helpers.utils import compute_factor_wmf_deep, wpe_hybrid_strict, wpe_joint, wpe_joint_ncf, plot_grad_flow
from helpers.models import ModelAttributes, ModelMFuni, ModelNCACF, ModelNCF
from helpers.eval import evaluate_mf_hybrid, predict_attributes, evaluate_uni, evaluate_uni_in

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def train_wmf(params, in_out):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if in_out == 'out':
        n_songs_train = int(0.7 * n_songs_train)

    # Path for the features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_wmf = os.path.join(params['out_dir'], 'wmf.npz')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))
    confT = conf.T.tocsr()

    print('\n Update WMF factors...')
    start_time_wmf = time.time()
    W, H = None, np.random.randn(n_songs_train, params['n_embeddings']).astype('float32') * 0.01
    for iwmf in range(params['n_iter_wmf']):
        W = compute_factor_wmf_deep(H, conf, lW)
        H = compute_factor_wmf_deep(W, confT, lH)
    time_wmf = time.time() - start_time_wmf

    # Save the WMF parameters (and computational time)
    np.savez(path_wmf, W=W, H=H, time_wmf=time_wmf)

    return


def train_2stages_relaxed(params, in_out):

    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if in_out == 'out':
        n_songs_train = int(0.7 * n_songs_train)

    # Path for the features and the pre-calculated WMF
    path_wmf = os.path.join(params['out_dir'], 'wmf.npz')
    if in_out == 'out':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Model parameters and definition
    my_model = ModelAttributes(params['n_features_in'], params['n_features_hidden'],
                               params['n_embeddings']).to(params['device'])
    print('Amount of parameters: {}'.format(sum([p.numel() for p in my_model.parameters()])), flush=True)

    # Load the W matrix (used for validation) and WMF time
    wmf_loader = np.load(path_wmf)
    time_wmf, W = wmf_loader['time_wmf'], wmf_loader['W']

    # Define the dataset and loader
    my_dataset = DatasetAttributes(wmf_path=path_wmf, features_path=path_features)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Model update
    time_tot, loss_tot, val_ndcg_tot = time_wmf, [], []
    time_opt, ndcg_opt = time_tot, 0
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()
        epoch_losses = []
        for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
            my_optimizer.zero_grad()
            x = data[0].to(params['device'])
            h = data[1].to(params['device'])
            h_hat = my_model(x)
            loss = torch.nn.MSELoss()(h_hat, h)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_mf_hybrid(params, W, None, my_model, in_out=in_out, variant='relaxed', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model, os.path.join(params['out_dir'], 'model_relaxed.pt'))

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training_relaxed.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_2stages_strict(params, in_out):

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_wmf = os.path.join(params['out_dir'], 'wmf.npz')
    if in_out == 'out':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Model parameters and definition
    my_model = ModelAttributes(params['n_features_in'], params['n_features_hidden'],
                               params['n_embeddings']).to(params['device'])
    print('Amount of parameters: {}'.format(sum([p.numel() for p in my_model.parameters()])), flush=True)

    # Load the WMF matrix (used for validation) and WMF time
    wmf_loader = np.load(path_wmf)
    time_wmf, W = wmf_loader['time_wmf'], wmf_loader['W']

    # Dataset
    my_dataset = DatasetAttributesRatings(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Model training
    time_tot, loss_tot, val_ndcg_tot = time_wmf, [], []
    time_opt, ndcg_opt = time_tot, 0
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep+1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()
        epoch_losses = []
        for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
            # Load the user and item indices and account for negative samples
            x = data[0].to(params['device'])
            count_i = data[1].to(params['device'])
            # Forward pass
            h_hat = my_model(x)
            # Back-propagation
            loss = wpe_hybrid_strict(h_hat, torch.tensor(W).to(params['device']), count_i)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_mf_hybrid(params, W, None, my_model, in_out=in_out, variant='strict', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model, os.path.join(params['out_dir'], 'model_strict.pt'))

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training_strict.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_2stages(params, variant, in_out):

    if variant == 'relaxed':
        train_2stages_relaxed(params, in_out)
    else:
        train_2stages_strict(params, in_out)

    return


def train_mf_hybrid_relaxed(params, in_out):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if in_out == 'out':
        n_songs_train = int(0.7 * n_songs_train)

    # Path for the TP data, WMF, and features
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_wmf_temp = os.path.join(params['out_dir'], 'params_wmf_temp.npz')
    if in_out == 'out':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))
    confT = conf.T.tocsr()

    # Model parameters and definition
    my_model = ModelAttributes(params['n_features_in'], params['n_features_hidden'],
                               params['n_embeddings']).to(params['device'])
    print('Amount of parameters: {}'.format(sum([p.numel() for p in my_model.parameters()])), flush=True)

    # Init WMF factors
    W, H = None, np.random.randn(n_songs_train, params['n_embeddings']).astype('float32') * 0.01
    np.savez(path_wmf_temp, W=W, H=H)

    # Define the dataset and loader
    my_dataset = DatasetAttributes(wmf_path=path_wmf_temp, features_path=path_features)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Model training
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()

        # Update the MF model every ep_it epochs
        if ep % params['n_ep_it'] == 0:
            #  Predict the content attributes using the deep model and update the WMF factors
            predicted_attributes = predict_attributes(my_model, my_dataloader, n_songs_train, params['n_embeddings'],
                                                      params['device'])
            W = compute_factor_wmf_deep(H, conf, lW)
            H = compute_factor_wmf_deep(W, confT, lH, content_prior=predicted_attributes)

            # Save the MF parameters and define the dataset for training the attribute model
            np.savez(path_wmf_temp, W=W, H=H)
            my_dataset = DatasetAttributes(wmf_path=path_wmf_temp, features_path=path_features)
            my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

        # Attribute model update
        epoch_losses = []
        for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
            my_optimizer.zero_grad()
            x = data[0].to(params['device'])
            h = data[1].to(params['device'])
            h_hat = my_model(x)
            loss = torch.nn.MSELoss()(h_hat, h)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_mf_hybrid(params, W, H, my_model, in_out=in_out, variant='relaxed', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model, os.path.join(params['out_dir'], 'model.pt'))
            np.savez(os.path.join(params['out_dir'], 'wmf.npz'), W=W, H=H)

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_mf_hybrid_strict(params, in_out):

    # Get the hyperparameter
    lW = params['lW']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if in_out == 'out':
        n_songs_train = int(0.7 * n_songs_train)

    # Path for the TP training data and features
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    if in_out == 'out':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Get the playcount data and confidence
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Load the pre-trained model
    my_model = ModelAttributes(params['n_features_in'], params['n_features_hidden'],
                               params['n_embeddings']).to(params['device'])
    print('Amount of parameters: {}'.format(sum([p.numel() for p in my_model.parameters()])), flush=True)

    # Dataloader for predicting the attributes
    my_dataset_attr = DatasetAttributes(wmf_path=None, features_path=path_features)
    my_dataloader_attr = DataLoader(my_dataset_attr, params['batch_size'], shuffle=False, drop_last=False)

    # Define the dataset
    my_dataset_tr = DatasetAttributesRatings(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
    my_dataloader_tr = DataLoader(my_dataset_tr, params['batch_size'], shuffle=True, drop_last=True)

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Model training
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()

        # Update the MF model every ep_it epochs
        if ep % params['n_ep_it'] == 0:
            #  Predict the content attributes using the deep model and update the WMF factors
            predicted_attributes = predict_attributes(my_model, my_dataloader_attr, n_songs_train, params['n_embeddings'],
                                                      params['device'])
            W = compute_factor_wmf_deep(predicted_attributes, conf, lW)

        # Model update
        epoch_losses = []
        for data in tqdm(my_dataloader_tr, desc='Training', unit=' Batches(s)'):
            # Load the content features and item indices
            x = data[0].to(params['device'])
            count_i = data[1].to(params['device'])
            # Forward pass
            h_hat = my_model(x)
            # Back-propagation
            loss = wpe_hybrid_strict(h_hat, torch.tensor(W).to(params['device']), count_i)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_mf_hybrid(params, W, None, my_model, in_out=in_out, variant='strict', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model, os.path.join(params['out_dir'], 'model.pt'))
            np.savez(os.path.join(params['out_dir'], 'wmf.npz'), W=W, H=0)

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_mf_hybrid(params, variant, in_out):

    if variant == 'relaxed':
        train_mf_hybrid_relaxed(params, in_out)
    else:
        train_mf_hybrid_strict(params, in_out)

    return


def train_mf_uni(params, variant='relaxed', in_out='out'):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if in_out == 'out':
        n_songs_train = int(0.7 * n_songs_train)

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    if in_out == 'out':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelMFuni(n_users, n_songs_train, params['n_embeddings'], params['n_features_in'],
                          params['n_features_hidden'], variant, params['out_sigm'])
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
            x = data[0].to(params['device'])
            count_i = data[1].to(params['device'])
            it = data[2].to(params['device'])
            # Forward pass
            pred_rat, w, h, h_con = my_model(u_total, x, it)
            # Back-propagation
            loss = wpe_joint(count_i, pred_rat, w, h, h_con, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, in_out, split='val')
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


def train_ncf_in(params):

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
    my_model = ModelNCF(n_users, n_songs_train, params['n_embeddings'])
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
            # Load the user and item indices
            count_i = data[1].to(params['device'])
            it = data[2].to(params['device'])
            # Forward pass
            pred_rat, w_gmf, h_gmf, w_mlp, h_mlp = my_model(u_total, it)
            # Back-propagation
            loss = wpe_joint_ncf(count_i, torch.transpose(pred_rat, 1, 0), w_gmf, h_gmf, w_mlp, h_mlp, lW, lH)
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


def train_ncacf(params, path_pretrain=None, variant='relaxed', inter='mult', out_act='sigmoid'):

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
    if path_pretrain is None:
        lW, lH = params['lW'], params['lH']
    else:
        my_model.load_state_dict(torch.load(path_pretrain + 'model.pt'), strict=False)
        lamb_load = np.load(os.path.join(path_pretrain, 'hyperparams.npz'))
        lW, lH = float(lamb_load['lW']), float(lamb_load['lH'])
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
            #plot_grad_flow(my_model.named_parameters())
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, split='val')
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

    return

