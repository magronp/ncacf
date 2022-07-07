#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

from helpers.eval import evaluate_uni
from helpers.models import ModelMFuni
from helpers.utils import create_folder, wpe_joint, get_optimal_val_model_lambda
from helpers.plotters import plot_val_ndcg_mf_uni_relaxed, plot_val_ndcg_mf_uni_strict
import numpy as np
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import DatasetPlaycounts
import copy
from helpers.data_feeder import load_tp_data


def train_mf_uni(params, variant='relaxed', setting='cold', rec_model=True, seed=1234):

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    if setting == 'cold':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Define the dataset
    my_dataset = DatasetPlaycounts(features_path=path_features, tp_path=path_tp_train)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Get the number of users and songs
    train_data = load_tp_data(path_tp_train, setting)[0]
    n_users, n_songs_train = train_data.shape

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelMFuni(n_users, n_songs_train, params['n_embeddings'], params['n_features_in'],
                          params['n_features_hidden'], variant)
    my_model.requires_grad_(True)
    my_model.to(params['device'])

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Initialize training log and optimal copies
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    model_opt = copy.deepcopy(my_model)

    # Training loop
    my_model.train()
    u_total = torch.arange(0, n_users, dtype=torch.long).to(params['device'])
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()
        epoch_losses = []
        for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
            my_optimizer.zero_grad()
            # Load the user and item indices and account for negative samples
            x = data[0].to(params['device'])
            count_i = torch.transpose(data[1], 1, 0).to(params['device'])
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
        val_ndcg = evaluate_uni(params, my_model, setting, split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            model_opt = copy.deepcopy(my_model)

    # Record the training log and model
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)
    if rec_model:
        torch.save(model_opt.state_dict(), os.path.join(params['out_dir'], 'model.pt'))

    return model_opt, time_opt


def train_val_mf_uni(setting_list, variant_list, params, range_lW, range_lH, data_dir='data/'):

    for setting in setting_list:
        # Define the dataset and output path
        path_current = 'outputs/' + setting + '/mf_uni/'
        params['data_dir'] = data_dir + setting + '/split0/'

        for variant in variant_list:
            print('MF-Uni -- Setting: ' + setting + ' -  Variant: ' + variant)

            if variant == 'relaxed':
                for lW in range_lW:
                    for lH in range_lH:
                        print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
                        params['lW'], params['lH'] = lW, lH
                        params['out_dir'] = path_current + 'relaxed/lW_' + str(lW) + '/lH_' + str(lH) + '/'
                        create_folder(params['out_dir'])
                        train_mf_uni(params, variant=variant, setting=setting)
            else:
                for lW in range_lW:
                    print('lambda_W=' + str(lW))
                    params['lW'], params['lH'] = lW, 0.
                    params['out_dir'] = path_current + 'strict/lW_' + str(lW) + '/'
                    create_folder(params['out_dir'])
                    train_mf_uni(params, variant='strict', setting=setting)

    return


if __name__ == '__main__':

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(torch.cuda.get_device_name(device)))

    # Set parameters
    params = {'batch_size': 128,
              'n_embeddings': 128,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'n_epochs': 150,
              'lr': 1e-4,
              'device': device
              }
    data_dir = 'data/'

    # Define the settings (warm and cold start) and the variants (relaxed and strict)
    setting_list = ['warm', 'cold']
    variant_list = ['relaxed', 'strict']
    
    # Training
    range_lW, range_lH = [0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10]
    train_val_mf_uni(setting_list, variant_list, params, range_lW, range_lH, data_dir)

    # Select the best performing model (both variants and scenarios)
    get_optimal_val_model_lambda('mf_uni', setting_list, variant_list, params['n_epochs'], range_lW, range_lH)

    # Plot the validation loss as a function of the hyperparameters
    plot_val_ndcg_mf_uni_relaxed('warm')
    plot_val_ndcg_mf_uni_strict('warm')
    plot_val_ndcg_mf_uni_relaxed('cold')
    plot_val_ndcg_mf_uni_strict('cold')

# EOF
