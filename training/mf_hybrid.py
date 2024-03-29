#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

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
from helpers.models import ModelAttributes
from helpers.eval import evaluate_mf_hybrid, predict_attributes
from helpers.utils import create_folder, get_optimal_val_model_lambda
from helpers.plotters import plot_val_ndcg_mf_hybrid_relaxed, plot_val_ndcg_mf_hybrid_strict, plot_val_mf_hybrid_epiter
import copy


def train_mf_hybrid_relaxed(params, setting, rec_model=True, seed=1234):

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Path for the TP data, WMF, and features
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_wmf_temp = os.path.join(params['out_dir'], 'params_wmf_temp.npz')
    if setting == 'cold':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Get the playcount data, confidence, and precompute its transpose
    _, _, _, conf = load_tp_data(path_tp_train, setting)
    confT = conf.T.tocsr()
    n_songs_train = conf.shape[1]

    # Model parameters and definition
    my_model = ModelAttributes(params['n_features_in'], params['n_features_hidden'],
                               params['n_embeddings']).to(params['device'])
    print('Amount of parameters: {}'.format(sum([p.numel() for p in my_model.parameters()])), flush=True)

    # Init WMF factors
    W, H = None, np.random.randn(n_songs_train, params['n_embeddings']).astype('float32') * 0.01
    np.savez(path_wmf_temp, W=W, H=H)

    # Define the dataset and loader
    my_dataset = DatasetAttributes(wmf_path=path_wmf_temp, features_path=path_features)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=False)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Initialize training log and optimal copies
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    model_opt = copy.deepcopy(my_model)
    W_opt, H_opt = W, H
    
    # Training loop
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
            my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=False)

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
        val_ndcg = evaluate_mf_hybrid(params, W, H, my_model, setting=setting, variant='relaxed', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            W_opt, H_opt = W, H
            model_opt = copy.deepcopy(my_model)
            
    # Record the training log and model
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)
    if rec_model:
        torch.save(model_opt, os.path.join(params['out_dir'], 'model.pt'))
        np.savez(os.path.join(params['out_dir'], 'wmf.npz'), W=W_opt, H=H_opt)

    return model_opt, W, H, time_opt


def train_mf_hybrid_strict(params, setting, rec_model=True, seed=1234):

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get the hyperparameter
    lW = params['lW']

    # Path for the TP training data and features
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    if setting == 'cold':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Get the playcount data and confidence
    _, _, _, conf = load_tp_data(path_tp_train, setting)
    n_songs_train = conf.shape[1]

    # Load the pre-trained model
    my_model = ModelAttributes(params['n_features_in'], params['n_features_hidden'],
                               params['n_embeddings']).to(params['device'])
    print('Amount of parameters: {}'.format(sum([p.numel() for p in my_model.parameters()])), flush=True)
    
    # Dataloader for predicting the attributes
    my_dataset_attr = DatasetAttributes(wmf_path=None, features_path=path_features)
    my_dataloader_attr = DataLoader(my_dataset_attr, params['batch_size'], shuffle=False, drop_last=False)

    # Define the dataset
    my_dataset_tr = DatasetPlaycounts(features_path=path_features, tp_path=path_tp_train)
    my_dataloader_tr = DataLoader(my_dataset_tr, params['batch_size'], shuffle=True, drop_last=False)

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Initialize training log and optimal copies
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    model_opt = copy.deepcopy(my_model)
    W_opt = 0
    
    # Training loop
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()

        # Update the MF model every ep_it epochs
        if ep % params['n_ep_it'] == 0:
            #  Predict the content attributes using the deep model and update the WMF factors
            predicted_attributes = predict_attributes(my_model, my_dataloader_attr, n_songs_train,
                                                      params['n_embeddings'], params['device'])
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
        val_ndcg = evaluate_mf_hybrid(params, W, None, my_model, setting=setting, variant='strict', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            model_opt = copy.deepcopy(my_model)
            W_opt = W

    # Record the training log and model
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)
    if rec_model:
        torch.save(model_opt, os.path.join(params['out_dir'], 'model.pt'))
        np.savez(os.path.join(params['out_dir'], 'wmf.npz'), W=W_opt, H=0)
        
    return model_opt, W, time_opt


def train_mf_hybrid(params, variant, setting, rec_model=True):

    if variant == 'relaxed':
        model_opt, W, H, time_opt = train_mf_hybrid_relaxed(params, setting, rec_model)
    else:
        model_opt, W, time_opt = train_mf_hybrid_strict(params, setting, rec_model)
        H = None
        
    return model_opt, W, H, time_opt


def train_val_mf_hybrid(setting_list, variant_list, params, range_lW, range_lH, data_dir='data/'):

    # For validation over lambda, set N_gd at 1
    params['n_ep_it'] = 1

    for setting in setting_list:
        # Define the dataset and output path depending on if it's in/out task
        path_current = 'outputs/' + setting + '/mf_hybrid/'
        params['data_dir'] = data_dir + setting + '/split0/'

        for variant in variant_list:
            print('MF-Hybrid -- Setting: ' + setting + ' -  Variant: ' + variant)

            if variant == 'relaxed':
                for lW in range_lW:
                    for lH in range_lH:
                        print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
                        params['lW'], params['lH'] = lW, lH
                        params['out_dir'] = path_current + 'relaxed/lW_' + str(lW) + '/lH_' + str(lH) + '/'
                        create_folder(params['out_dir'])
                        train_mf_hybrid(params, variant=variant, setting=setting)
            else:
                for lW in range_lW:
                    print('lambda_W=' + str(lW))
                    params['lW'], params['lH'] = lW, 0.
                    params['out_dir'] = path_current + 'strict/lW_' + str(lW) + '/'
                    create_folder(params['out_dir'])
                    train_mf_hybrid(params, variant='strict', setting=setting)

    return


def check_NGD_mf_hybrid(setting_list, variant_list, n_ep_it_list, params, data_dir='data/'):

    for setting in setting_list:
        path_current = 'outputs/' + setting + '/mf_hybrid/'
        params['data_dir'] = data_dir + setting + '/split0/'
        for variant in variant_list:
            print('MF-Hybrid -- Setting: ' + setting + ' -  Variant: ' + variant)

            # Load the optimal hyper-parameters
            lamb_load = np.load(path_current + variant + '/hyperparams.npz')
            params['lW'], params['lH'] = lamb_load['lW'], lamb_load['lH']
            # Try other ep_it
            for n_ep_it in n_ep_it_list:
                print('N_GD=' + str(n_ep_it))
                params['out_dir'] = path_current + variant + '/gd_' + str(n_ep_it) + '/'
                create_folder(params['out_dir'])
                params['n_ep_it'] = n_ep_it
                train_mf_hybrid(params, variant=variant, setting=setting)

    return


if __name__ == '__main__':

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(device))

    # Set parameters
    params = {'batch_size': 128,
              'n_embeddings': 128,
              'n_epochs': 150,
              'lr': 1e-4,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'device': device}
    data_dir = 'data/'

    # Define the settings (warm and cold start) and the variants (relaxed and strict)
    setting_list = ['warm', 'cold']
    variant_list = ['relaxed', 'strict']

    # Define the hyperparameters over which performing a grid search
    range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]
    
    # Train MF Hybrid models
    train_val_mf_hybrid(setting_list, variant_list, params, range_lW, range_lH, data_dir)

    # Select the best performing model (both variants and scenarios)
    get_optimal_val_model_lambda('mf_hybrid', setting_list, variant_list, params['n_epochs'], range_lW, range_lH)

    # Check what happens if N_GD varies (more epochs at each iteration)
    n_ep_it_list = [2, 5, 10]
    check_NGD_mf_hybrid(setting_list, variant_list, n_ep_it_list, params, data_dir)

    # Display validation results:
    # the impact of lammbda_W and lambda_H
    plot_val_ndcg_mf_hybrid_relaxed('warm')
    plot_val_ndcg_mf_hybrid_strict('warm')
    plot_val_ndcg_mf_hybrid_relaxed('cold')
    plot_val_ndcg_mf_hybrid_strict('cold')

    # the impact of N_GD
    n_ep_it_list.insert(0, 1)
    plot_val_mf_hybrid_epiter('cold', 'relaxed', params['n_epochs'], n_ep_it_list)
    plot_val_mf_hybrid_epiter('cold', 'strict', params['n_epochs'], n_ep_it_list)
    plot_val_mf_hybrid_epiter('warm', 'relaxed', params['n_epochs'], n_ep_it_list)
    plot_val_mf_hybrid_epiter('warm', 'strict', params['n_epochs'], n_ep_it_list)

# EOF
