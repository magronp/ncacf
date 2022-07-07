#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

from helpers.utils import create_folder
from helpers.plotters import plot_val_ndcg_hybrid_relaxed
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
from helpers.eval import evaluate_mf_hybrid
import copy


def train_wmf(params, setting, rec_model=True, seed=1234):

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Path for the features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, setting)
    confT = conf.T.tocsr()

    # Get the number of songs and users
    #n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    #if setting == 'cold':
    #    n_songs_train = int(0.8 * 0.9 * n_songs_train)
    n_songs_train = train_data.shape[1]

    print('\n Update WMF factors...')
    start_time_wmf = time.time()
    W, H = None, np.random.randn(n_songs_train, params['n_embeddings']).astype('float32') * 0.01
    for iwmf in range(params['n_iter_wmf']):
        W = compute_factor_wmf_deep(H, conf, lW)
        H = compute_factor_wmf_deep(W, confT, lH)
    time_wmf = time.time() - start_time_wmf

    # Save the WMF parameters (and computational time)
    if rec_model:
        path_wmf = os.path.join(params['out_dir'], 'wmf.npz')
        np.savez(path_wmf, W=W, H=H, time_wmf=time_wmf)

    return W, H


def train_2stages_relaxed(params, setting, rec_model=True, seed=1234):

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Path for the features and the pre-calculated WMF
    path_wmf = os.path.join(params['out_dir'], 'wmf.npz')
    if setting == 'cold':
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
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=False)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Initialize training log and optimal copies
    time_tot, loss_tot, val_ndcg_tot = time_wmf, [], []
    time_opt, ndcg_opt = time_tot, 0
    model_opt = copy.deepcopy(my_model)

    # Training loop
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

        val_ndcg = evaluate_mf_hybrid(params, W, None, my_model, setting=setting, variant='relaxed', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            model_opt = copy.deepcopy(my_model)

    # Record the training log and the optimal model (if needed)
    np.savez(os.path.join(params['out_dir'], 'training_relaxed.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)
    if rec_model:
        torch.save(model_opt, os.path.join(params['out_dir'], 'model_relaxed.pt'))

    return model_opt


def train_2stages_strict(params, setting, rec_model=True, seed=1234):

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_wmf = os.path.join(params['out_dir'], 'wmf.npz')
    if setting == 'cold':
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
    my_dataset = DatasetPlaycounts(features_path=path_features, tp_path=path_tp_train)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=False)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Initialize training log and optimal copies
    time_tot, loss_tot, val_ndcg_tot = time_wmf, [], []
    time_opt, ndcg_opt = time_tot, 0
    model_opt = copy.deepcopy(my_model)

    # Training loop
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
        val_ndcg = evaluate_mf_hybrid(params, W, None, my_model, setting=setting, variant='strict', split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            model_opt = copy.deepcopy(my_model)

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training_strict.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)
    if rec_model:
        torch.save(model_opt, os.path.join(params['out_dir'], 'model_strict.pt'))

    return model_opt


def train_2stages(params, variant, setting, rec_model=True):

    if variant == 'relaxed':
        model_opt = train_2stages_relaxed(params, setting, rec_model)
    else:
        model_opt = train_2stages_strict(params, setting, rec_model)

    return model_opt


def train_val_wmf_2stages(setting_list, variant_list, params, range_lW, range_lH, data_dir='data/'):

    for setting in setting_list:

        # Define data/outputs paths
        path_current = 'outputs/' + setting + '/2stages/'
        params['data_dir'] = data_dir + setting + '/split0/'

        # Loop over hyperparameters
        for lW in range_lW:
            for lH in range_lH:
                print('Task: ' + setting)
                print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
                params['lW'], params['lH'] = lW, lH
                params['out_dir'] = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/'
                create_folder(params['out_dir'])
                # First train the WMF
                train_wmf(params, setting=setting)
                # Then train the relaxed and strict variant on top of these WMFs
                for variant in variant_list:
                    print('Variant: ' + variant)
                    # Useless to train for the relaxed variant in the warm-start setting (it's juste WMF)
                    if not(variant == 'relaxed' and setting == 'warm'):
                        train_2stages(params, variant=variant, setting=setting)

    return


def get_optimal_2stages(setting_list, variant_list, range_lW, range_lH, n_epochs):

    for setting in setting_list:
        for variant in variant_list:
            if not (variant == 'relaxed' and setting == 'warm'):

                # Define data/outputs paths
                path_current = 'outputs/' + setting + '/2stages/'
                path_out = path_current + variant + '/'
                create_folder(path_out)

                # Load the validation score for the various models
                Nw, Nh = len(range_lW), len(range_lH)
                val_ndcg = np.zeros((Nw, Nh, n_epochs))
                for iW, lW in enumerate(range_lW):
                    for iH, lH in enumerate(range_lH):
                        path_load = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/training_' + variant + '.npz'
                        val_ndcg[iW, iH, :] = np.load(path_load)['val_ndcg'][:n_epochs] * 100

                # Get the optimal hyperparameters
                ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
                lW_opt, lH_opt = range_lW[ind_opt[0]], range_lH[ind_opt[1]]

                # Record the optimal hyperparameters and the overall validation NDCG
                np.savez(path_out + 'hyperparams.npz', lW=lW_opt, lH=lH_opt)
                np.savez(path_out + 'val_ndcg.npz', val_ndcg=val_ndcg, range_lW=range_lW, range_lH=range_lH)

                # Get the optimal model and corresponding training log and copy it
                path_opt = path_current + 'lW_' + str(lW_opt) + '/lH_' + str(lH_opt) + '/'
                train_log = np.load(path_opt + 'training_' + variant + '.npz')
                model_opt = torch.load(path_opt + 'model_' + variant + '.pt')
                wmf_opt = np.load(path_opt + 'wmf.npz')
                np.savez(path_out + 'training.npz', loss=train_log['loss'], time=train_log['time'], val_ndcg=train_log['val_ndcg'])
                np.savez(path_out + 'wmf.npz', W=wmf_opt['W'], H=wmf_opt['H'])
                torch.save(model_opt, path_out + 'model.pt')

    return


def get_optimal_wmf(params, range_lW, range_lH):

    params['data_dir'] = 'data/warm/split0/'

    # Selecting the best WMF model (these have already been trained)
    val_ndcg_opt, lW_opt, lH_opt = 0, 0, 0
    for lW in range_lW:
        for lH in range_lH:
            print('Validation...')
            print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
            path_wmf = 'outputs/warm/2stages/lW_' + str(lW) + '/lH_' + str(lH) + '/wmf.npz'
            W, H = np.load(path_wmf)['W'], np.load(path_wmf)['H']
            val_ndcg = evaluate_mf_hybrid(params, W, H, None, setting='warm', variant='relaxed', split='val')
            if val_ndcg > val_ndcg_opt:
                val_ndcg_opt = val_ndcg
                lW_opt, lH_opt = lW, lH

    # Load the best performing WMF model and store it conveniently
    path_wmf_opt = 'outputs/warm/2stages/lW_' + str(lW_opt) + '/lH_' + str(lH_opt) + '/wmf.npz'
    path_out = 'outputs/warm/WMF/'
    wmf_opt = np.load(path_wmf_opt)
    create_folder(path_out)
    np.savez(path_out + 'hyperparams.npz', lW=lW_opt, lH=lH_opt)
    np.savez(path_out + 'wmf.npz', W=wmf_opt['W'], H=wmf_opt['H'])

    return


if __name__ == '__main__':

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(device))

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

    # Define the settings (warm and cold start) and the variants (relaxed and strict)
    setting_list = ['warm', 'cold']
    variant_list = ['relaxed', 'strict']

    # Define the hyperparameters over which performing a grid search
    range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]

    # Training with validation. Then, select the best model (and also WMF, which is trained as well)
    train_val_wmf_2stages(setting_list, variant_list, params, range_lW, range_lH, data_dir)
    get_optimal_2stages(setting_list, variant_list, range_lW, range_lH, params['n_epochs'])
    get_optimal_wmf(params, range_lW, range_lH)

# EOF
