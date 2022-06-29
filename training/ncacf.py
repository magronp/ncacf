#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import torch
from helpers.utils import create_folder, wpe_joint, get_optimal_val_model_lW_lH, get_optimal_val_model_lW
import os
import time
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from helpers.models import ModelNCACF
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetPlaycounts
from helpers.eval import evaluate_uni
import copy
from helpers.plotters import plot_val_ndcg_ncacf


def train_ncacf(params, path_pretrain=None, n_layers_di=2, setting='cold', variant='relaxed', inter='mult',
                rec_model=True, seed=1234):
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(variant)
    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if setting == 'cold':
        n_songs_train = int(0.8 * 0.9 * n_songs_train)

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    if setting == 'cold':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Get the playcount data and confidence
    train_data, _, _, conf = load_tp_data(path_tp_train, setting)

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelNCACF(n_users, n_songs_train, params['n_features_in'], params['n_features_hidden'],
                          params['n_embeddings'], n_layers_di, variant, inter)
    if not (path_pretrain is None):
        my_model.load_state_dict(torch.load(path_pretrain + 'model.pt'), strict=False)
    my_model.requires_grad_(True)
    my_model.to(params['device'])

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Define the dataset
    my_dataset = DatasetPlaycounts(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Initialize training log and optimal copies
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    model_opt = copy.deepcopy(my_model)

    # Loop over epochs
    u_total = torch.arange(0, n_users, dtype=torch.long).to(params['device'])
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()
        epoch_losses = []
        for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
            my_optimizer.zero_grad()
            # Load the user and item indices and account for negative samples
            x = data[0].to(params['device'])
            counts_tot = torch.transpose(data[1], 1, 0).to(params['device'])
            it = data[2].to(params['device'])
            # Forward pass
            pred_rat, w, h, h_con = my_model(u_total, x, it)
            print(h - h_con)
            # Back-propagation
            loss = wpe_joint(counts_tot, pred_rat, w, h, h_con, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())
        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, setting=setting, split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            model_opt = copy.deepcopy(my_model)

    # Record the training log
    #np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)
    #if rec_model:
    #    torch.save(model_opt.state_dict(), os.path.join(params['out_dir'], 'model.pt'))

    return model_opt


def train_val_ncacf(setting_list, variant_list, params, range_lW, range_lH, range_inter, range_nl_di, data_dir='data/'):

    val_lambda = not(len(range_lW) == 1 and len(range_lW) == 1)

    for setting in setting_list:
        params['data_dir'] = data_dir + setting + '/split0/'
        for inter in range_inter:
            for nl_di in range_nl_di:
                path_current = 'outputs/' + setting + '/ncacf/' + inter + '/' + str(nl_di) + '/'

                for variant in variant_list:
                    print('NCACF -- Setting: ' + setting + ' -  Inter: ' + inter + ' - N_layers: ' + str(
                        nl_di) + ' - Variant: ' + variant)
                    if nl_di == -1:
                        path_pretrain = None
                    else:
                        path_pretrain = 'outputs/' + setting + '/ncacf/' + inter + '/' + str(-1) + '/' + variant + '/'
                    # Grid search on the hyperparameters
                    if val_lambda:
                        if variant == 'relaxed':
                            for lW in range_lW:
                                for lH in range_lH:
                                    print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
                                    params['lW'], params['lH'] = lW, lH
                                    params['out_dir'] = path_current + 'relaxed/lW_' + str(lW) + '/lH_' + str(lH) + '/'
                                    create_folder(params['out_dir'])
                                    train_ncacf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, setting=setting,
                                                variant=variant, inter=inter)
                            get_optimal_val_model_lW_lH(path_current + 'relaxed/', range_lW, range_lH, params['n_epochs'])
                        else:
                            for lW in range_lW:
                                print('lambda_W=' + str(lW))
                                params['lW'], params['lH'] = lW, 0.
                                params['out_dir'] = path_current + 'strict/lW_' + str(lW) + '/'
                                create_folder(params['out_dir'])
                                train_ncacf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, setting=setting,
                                            variant=variant, inter=inter)
                            get_optimal_val_model_lW(path_current + 'strict/', range_lW, params['n_epochs'])
                    else:
                        params['lW'], params['lH'] = range_lW[0], range_lH[0]
                        params['out_dir'] = path_current + variant + '/'
                        create_folder(params['out_dir'])
                        train_ncacf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, setting=setting,
                                    variant=variant, inter=inter)
                        np.savez(params['out_dir'] + 'hyperparams.npz', lW=params['lW'], lH=params['lH'])
    return


def get_optimal_ncacf(setting_list, variant_list, range_inter, range_nl_di):

    val_ndcg = np.zeros((len(setting_list), len(range_inter), len(range_nl_di), len(variant_list)))
    lambW = np.zeros((len(setting_list), len(range_inter), len(range_nl_di), len(variant_list)))
    lambH = np.zeros((len(setting_list), len(range_inter), len(range_nl_di), len(variant_list)))

    # Load all validation results
    for ise, setting in enumerate(setting_list):
        for ii, inter in enumerate(range_inter):
            for inl, nl_di in enumerate(range_nl_di):
                for iv, variant in enumerate(variant_list):
                    path_current = 'outputs/' + setting + '/ncacf/' + inter + '/' + str(nl_di) + '/' + variant + '/'
                    val_ndcg[ise, ii, inl, iv] = np.max(np.load(path_current + 'training.npz')['val_ndcg'])
                    lambload = np.load(path_current + 'hyperparams.npz')
                    lambW[ise, ii, inl, iv], lambH[ise, ii, inl, iv] = lambload['lW'], lambload['lH']

    # Find the optimal set of hyperparams and record it for each setting
    for ise, setting in enumerate(setting_list):
        ind_opt = np.unravel_index(np.argmax(val_ndcg[ise, :], axis=None), val_ndcg[ise, :].shape)
        inter_opt, nl_di_opt, var_opt = range_inter[ind_opt[0]], range_nl_di[ind_opt[1]], variant_list[ind_opt[2]]
        lW_opt = lambW[ise, ind_opt[0], ind_opt[1], ind_opt[2]]
        lH_opt = lambH[ise, ind_opt[0], ind_opt[1], ind_opt[2]]
        np.savez('outputs/' + setting + '/ncacf/hyperparams.npz', lW=lW_opt, lH=lH_opt, inter=inter_opt, nl_di=nl_di_opt,
                 variant=var_opt)

        # Also record the overall validation scores (for plotting)
        np.savez('outputs/' + setting + '/ncacf/val_results.npz', val_ndcg=val_ndcg[ise, :])

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
              'n_epochs': 100,
              'lr': 1e-4,
              'device': device
              }
    data_dir = 'data/'

    # Define the settings (warm and cold start) and the variants (relaxed and strict)
    setting_list = ['warm', 'cold']
    variant_list = ['relaxed', 'strict']

    # Define the hyperparameters over which performing a grid search
    range_lW, range_lH = [0.1], [0.1]
    range_inter, range_nl_di = ['mult', 'conc'], [-1, 0, 1, 2, 3, 4]

    # Training with validation
    train_val_ncacf(setting_list, variant_list, params, range_lW, range_lH, range_inter, range_nl_di, data_dir='data/')
    get_optimal_ncacf(setting_list, variant_list, range_inter, range_nl_di)

    # Plot validation results
    plot_val_ndcg_ncacf()

# EOF
