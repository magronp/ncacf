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
from helpers.utils import compute_factor_wmf_deep, wpe_hybrid_strict
from helpers.models import ModelAttributes
from helpers.eval import predict_attributes
from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from matplotlib import pyplot as plt
from helpers.eval import evaluate_mf_hybrid

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def train_mf_hybrid_relaxed_out(params):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Path for the features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    path_wmf_temp = os.path.join(params['out_dir'], 'params_wmf_temp.npz')

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


def train_mf_hybrid_strict_out(params):

    # Get the hyperparameter
    lW = params['lW']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')

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


def train_mf_hybrid_out(params, variant):

    if variant == 'relaxed':
        train_mf_hybrid_relaxed_out(params)
    else:
        train_mf_hybrid_strict_out(params)

    return


def train_val_mh_hybrid_out(params, range_lW, range_lH):

    path_current = 'outputs/out/mf_hybrid/'
    params['n_ep_it'] = 1

    # Relaxed variant
    variant = 'relaxed'
    for lW in range_lW:
        for lH in range_lH:
            print(lW, lH)
            params['lW'], params['lH'] = lW, lH
            params['out_dir'] = path_current + variant + '/lW_' + str(lW) + '/lH_' + str(lH) + '/'
            create_folder(params['out_dir'])
            train_mf_hybrid_out(params, variant=variant)
    get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])

    # Strict variant
    variant = 'strict'
    for lW in range_lW:
        print(lW)
        params['lW'] = lW
        params['out_dir'] = path_current + variant + '/lW_' + str(lW) + '/'
        create_folder(params['out_dir'])
        train_mf_hybrid_out(params, variant=variant)
    get_optimal_val_model_strict(path_current, range_lW, params['n_epochs'])

    return


def train_mh_hybrid_out_epiter(params):

    for variant in ['relaxed', 'strict']:
        # Load the optimal hyper-parameters
        lamb_load = np.load('outputs/out/mf_hybrid/' + variant + '/hyperparams.npz')
        params['lW'], params['lH'] = lamb_load['lW'], lamb_load['lH']

        # Try other ep_it
        for n_ep_it in [2, 5]:
            # Define the output directory
            params['out_dir'] = 'outputs/out/mf_hybrid/' + variant + '/gd_' + str(n_ep_it) + '/'
            create_folder(params['out_dir'])
            params['n_ep_it'] = n_ep_it
            train_mf_hybrid_out(params, variant=variant)

    return


def check_mh_hybrid_out_epiter(params, variant='relaxed'):

    path_out_dir = 'outputs/out/mf_hybrid/' + variant + '/'

    val_ndcg_epit = np.zeros((3, params['n_epochs']))
    test_ndcg_epit = np.zeros((3, 1))

    # For N_GD = 1
    # Validation NDCG
    val_ndcg_epit[0, :] = np.load(path_out_dir + 'training.npz')['val_ndcg'] * 100
    # Test NDCG
    mod_ep = torch.load(path_out_dir + 'model.pt')
    W = np.load(path_out_dir + '/wmf.npz')['W']
    test_ndcg_epit[0] = evaluate_mf_hybrid(params, W, mod_ep, split='test') * 100

    # For N_GD = 2 and 5
    for inep, n_ep_it in enumerate([2, 5]):
        path_out_dir = 'outputs/out/mf_hybrid/' + variant + '/gd_' + str(n_ep_it) + '/'
        # Validation NDCG
        val_ndcg_epit[inep+1, :] = np.load(path_out_dir + 'training.npz')['val_ndcg'] * 100
        # Test NDCG
        mod_ep = torch.load(path_out_dir + 'model.pt')
        W = np.load(path_out_dir + '/wmf.npz')['W']
        test_ndcg_epit[inep+1] = evaluate_mf_hybrid(params, W, mod_ep, split='test') * 100

    # Plot the validation NDCG
    plt.figure()
    plt.plot(np.arange(params['n_epochs']) + 1, val_ndcg_epit.T)
    plt.xlabel('Epochs')
    plt.ylabel('NDCG (%)')
    plt.legend(['N=1', 'N=2', 'N=5'])

    # Check the test NDCG
    print(test_ndcg_epit)

    return


def train_noval_mf_hybrid_relaxed_out(params, lW=0.1, lH=1.):

    params['lW'], params['lH'] = lW, lH
    params['out_dir'] = 'outputs/out/mf_hybrid/relaxed/'
    create_folder(params['out_dir'])
    train_mf_hybrid_out(params, variant='relaxed')

    return


def train_noval_mf_hybrid_strict_out(params, lW=0.1):

    params['lW'], params['lH'] = lW, 0.
    params['out_dir'] = 'outputs/out/mf_hybrid/strict/'
    create_folder(params['out_dir'])
    train_mf_hybrid_out(params, variant='strict')

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(torch.cuda.get_device_name(device)))

    # Set parameters
    params = {'batch_size': 128,
              'n_embeddings': 128,
              'n_epochs': 100,
              'lr': 1e-4,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'data_dir': 'data/out/',
              'device': device}

    train_mfuni = False
    val_mfuni = True

    if train_mfuni:
        if val_mfuni:
            # Training and validation for the hyperparameters
            range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10]
            train_val_mh_hybrid_out(params, range_lW, range_lH)

            # Check what happens if ep_it varies and display the results on the test set
            train_mh_hybrid_out_epiter(params)
            check_mh_hybrid_out_epiter(params, variant='relaxed')
            check_mh_hybrid_out_epiter(params, variant='strict')
        else:
            # Single training with pre-defined hyperparameter
            train_noval_mf_hybrid_relaxed_out(params, lW=0.1, lH=1.)
            train_noval_mf_hybrid_strict_out(params, lW=1.)

    # Plot the validation results
    plot_val_ndcg_lW_lH('outputs/out/mf_hybrid/relaxed/')
    plot_val_ndcg_lW('outputs/out/mf_hybrid/strict/')

    # Test

# EOF
