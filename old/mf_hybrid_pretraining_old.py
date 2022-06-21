#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from helpers.utils import create_folder
import numpy as np
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributes, DatasetPlaycounts
from helpers.utils import compute_factor_wmf_deep, wpe_hybrid_strict
from helpers.eval import evaluate_mf_hybrid, predict_attributes


def train_baseline_relaxed_pretraining(params, path_pretrain):

    # Defining the features and WMF paths
    path_features = params['data_dir'] + 'train_feats.num.csv'
    path_wmf = os.path.join(path_pretrain, 'wmf.npz')

    # Load pre-trained model and WMF factors
    my_model = torch.load(os.path.join(path_pretrain, 'model.pt'))
    wmfparams = np.load(path_wmf)
    W, H = wmfparams['W'], wmfparams['H']

    # Dataset
    my_dataset = DatasetAttributes(wmf_path=path_wmf, features_path=path_features)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Model training
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep+1, e=params['n_epochs']), flush=True)
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
        val_ndcg = evaluate_mf_hybrid(params, W, my_model, split='val')
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


def train_baseline_strict_pretraining(params, path_pretrain):

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = params['data_dir'] + 'train_feats.num.csv'

    # Load pre-trained model and WMF factors
    my_model = torch.load(os.path.join(path_pretrain, 'model.pt'))
    W = np.load(os.path.join(path_pretrain, 'wmf.npz'))['W']

    # Dataset
    my_dataset = DatasetPlaycounts(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Model training
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
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
        val_ndcg = evaluate_mf_hybrid(params, W, my_model, split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model, os.path.join(params['out_dir'], 'model.pt'))
            np.savez(os.path.join(params['out_dir'], 'wmf.npz'), W=W)

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_baseline(params, path_pretrain, variant):

    if variant == 'relaxed':
        train_baseline_relaxed_pretraining(params, path_pretrain)
    else:
        train_baseline_strict_pretraining(params, path_pretrain)

    return


def train_mf_hybrid_relaxed_pretraining(params, path_pretrain):

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Path for the features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    path_wmf_temp = os.path.join(params['out_dir'], 'params_wmf_temp.npz')
    path_pretrain_wmf = os.path.join(path_pretrain, 'wmf.npz')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))
    confT = conf.T.tocsr()

    # Load pre-trained model and WMF
    my_model = torch.load(os.path.join(path_pretrain, 'model.pt'))
    wmf_loader = np.load(os.path.join(path_pretrain, 'wmf.npz'))
    W, H = wmf_loader['W'], wmf_loader['H']

    # Load the optimal hyper-parameters
    lamb_load = np.load(os.path.join(path_pretrain, 'hyperparams.npz'))
    lW, lH = lamb_load['lW'], lamb_load['lH']

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Initial dataset for computing the predicted attributes (use the pretrained WMF)
    my_dataset_attr = DatasetAttributes(wmf_path=path_pretrain_wmf, features_path=path_features)
    my_dataloader_attr = DataLoader(my_dataset_attr, params['batch_size'], shuffle=False, drop_last=False)
    my_dataloader = None

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
        val_ndcg = evaluate_mf_hybrid(params, W, my_model, split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model, os.path.join(params['out_dir'], 'model_' + str(params['n_epochs']) + '.pt'))
            np.savez(os.path.join(params['out_dir'], 'wmf_' + str(params['n_epochs']) + '.npz'), W=W, H=H)

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training_' + str(params['n_epochs']) + '.npz'), loss=loss_tot,
             time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_mf_hybrid_strict_pretraining(params, path_pretrain):

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')

    # Get the playcount data and confidence
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Load the pre-trained model and training log
    my_model = torch.load(os.path.join(path_pretrain, 'model.pt')).to(params['device'])

    # Load the optimal hyper-parameter
    lW = np.load(os.path.join(path_pretrain, 'hyperparams.npz'))['lW']

    # Dataloader for predicting the attributes (faster than doing it using neg sampling)
    my_dataset_attr = DatasetAttributes(wmf_path=None, features_path=path_features)
    my_dataloader_attr = DataLoader(my_dataset_attr, params['batch_size'], shuffle=False, drop_last=False)

    # Define the dataset
    my_dataset_tr = DatasetPlaycounts(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
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
        val_ndcg = evaluate_mf_hybrid(params, W, my_model, split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model, os.path.join(params['out_dir'], 'model_' + str(params['n_epochs']) + '.pt'))
            np.savez(os.path.join(params['out_dir'], 'wmf_' + str(params['n_epochs']) + '.npz'), W=W)

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training_' + str(params['n_epochs']) + '.npz'), loss=loss_tot,
             time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_mf_hybrid(params, path_pretrain, variant):

    if variant == 'relaxed':
        train_mf_hybrid_relaxed_pretraining(params, path_pretrain)
    else:
        train_mf_hybrid_strict_pretraining(params, path_pretrain)

    return


def train_all_mf(params):

    for variant in ['relaxed', 'strict']:
        path_pretrain = 'outputs/pretraining_hybrid/' + variant + '/'

        # Train the baseline (no MF updates)
        params['out_dir'] = 'outputs/baseline/' + variant + '/'
        create_folder(params['out_dir'])
        train_baseline(params, path_pretrain, variant)

        # Train the MF-Hybrid models
        params['out_dir'] = 'outputs/MFHybrid/' + variant + '/'
        create_folder(params['out_dir'])
        for ep_per_iter in [1, 2, 5]:
            params['n_ep_it'] = ep_per_iter
            train_mf_hybrid(params, path_pretrain, variant)

    return


def plot_val_ndcg_mf(n_epochs):

    val_ndcg = np.zeros((n_epochs, 4, 2))
    for iv, variant in enumerate(['relaxed', 'strict']):
        val_ndcg[:, 0, iv] = np.load('outputs/baseline/' + variant + '/training.npz')['val_ndcg']
        for ie, ep in enumerate([1, 2, 5]):
            val_ndcg[:, ie+1, iv] = np.load('outputs/MFHybrid/' + variant + '/training_' + str(ep) + '.npz')['val_ndcg']

    val_ndcg = val_ndcg * 100

    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(n_epochs) + 1, val_ndcg[:, :, 0])
    plt.xlabel('Epochs')
    plt.ylabel('NDCG (%)')
    plt.title('Relaxed')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(n_epochs) + 1, val_ndcg[:, :, 1])
    plt.legend(['Baseline', 'MH-Hybrid, ' + r'$N_{sgd}=1$', 'MH-Hybrid, ' + r'$N_{sgd}=2$', 'MH-Hybrid, ' + r'$N_{sgd}=5$'])
    plt.xlabel('Epochs')
    plt.title('Strict')
    plt.show()

    return


def test_all_mf(params):

    res = np.zeros((4, 2, 2))
    for iv, variant in enumerate(['relaxed', 'strict']):

        # Test the baseline
        my_model = torch.load('outputs/baseline/' + variant + '/model.pt')
        W = np.load('outputs/baseline/' + variant + '/wmf.npz')['W']
        res[0, iv, 0] = evaluate_mf_hybrid(params, W, my_model, split='test') * 100
        res[0, iv, 1] = np.load('outputs/baseline/' + variant + '/training.npz')['time']

        # Test MF-Hybrid
        for ie, ep_per_iter in enumerate([1, 2, 5]):
            my_model = torch.load('outputs/MFHybrid/' + variant + '/model_' + str(ep_per_iter) + '.pt')
            W = np.load('outputs/MFHybrid/' + variant + '/wmf_' + str(ep_per_iter) + '.npz')['W']
            res[ie+1, iv, 0] = evaluate_mf_hybrid(params, W, my_model, split='test') * 100
            res[ie+1, iv, 1] = np.load('outputs/MFHybrid/' + variant + '/training_' + str(ep_per_iter) + '.npz')['time']

    #time_als = np.load('outputs/pretraining_hybrid/relaxed/training.npz')['time_als']
    time_als = 0
    time_pre_relaxed = np.load('outputs/pretraining_hybrid/relaxed/training.npz')['time'] + time_als
    time_pre_strict = np.load('outputs/pretraining_hybrid/strict/training.npz')['time'] + time_als

    res[:, 0, 1] += time_pre_relaxed
    res[:, 1, 1] += time_pre_strict

    return res


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(torch.cuda.get_device_name(device)))

    # Set parameters
    params = {'batch_size': 128,
              'n_embeddings': 128,
              'n_epochs': 30,
              'lr': 1e-4,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'data_dir': 'data/',
              'device': device}

    # Training
    train_all_mf(params)

    # Display NDCG on the validation set
    plot_val_ndcg_mf(params['n_epochs'])

    # Print test NDCG
    res = test_all_mf(params)
    print('-- Relaxed -- ')
    print(res[:, 0, :])
    print('-- Strict -- ')
    print(res[:, 1, :])

# EOF
