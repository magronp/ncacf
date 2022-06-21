#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import torch
import os
import time
from helpers.utils import create_folder, get_optimal_val_model_lW_lH
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetPlaycounts
from helpers.utils import wpe_joint_ncf
from helpers.eval import evaluate_uni
from helpers.models import ModelNCF
import copy
from helpers.plotters import plot_val_ndcg_ncf


def train_ncf(params, path_pretrain=None, n_layers_di=2, inter='mult', rec_model=True):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = params['data_dir'] + 'feats.num.csv'

    # Get the playcount data and confidence
    train_data, _, _, conf = load_tp_data(path_tp_train, 'warm')

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelNCF(n_users, n_songs_train, params['n_embeddings'], n_layers_di, inter)
    if not(path_pretrain is None):
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
            # Load the user / item indices and (positive) counts
            counts_tot = torch.transpose(data[1], 1, 0).to(params['device'])
            it_batch = data[2].to(params['device'])
            # Forward pass
            pred_rat, w, h = my_model(u_total, None, it_batch)
            # Back-propagation
            loss = wpe_joint_ncf(counts_tot, pred_rat, w, h, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())
        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, setting='warm', split='val')
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

    return model_opt


def train_val_ncf(params, range_lW, range_lH, range_inter, range_nl_di, data_dir='data/'):

    val_lambda = not(len(range_lW) == 1 and len(range_lW) == 1)
    params['data_dir'] = data_dir + 'warm' + '/split0/'

    for inter in range_inter:
        for nl_di in range_nl_di:

            # Define the current working path and the pretraining path, if needed
            path_current = 'outputs/warm/ncf/' + inter + '/' + str(nl_di) + '/'
            if nl_di == -1:
                path_pretrain = None
            else:
                path_pretrain = 'outputs/warm/ncf/' + inter + '/' + str(-1) + '/'

            # Training with grid search on the hyperparameters
            if val_lambda:
                for lW in range_lW:
                    for lH in range_lH:
                        print(lW, lH)
                        params['lW'], params['lH'] = lW, lH
                        params['out_dir'] = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/'
                        create_folder(params['out_dir'])
                        train_ncf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, inter=inter)
                get_optimal_val_model_lW_lH(path_current, range_lW, range_lH, params['n_epochs'])

            else:
                params['lW'], params['lH'] = range_lW[0], range_lH[0]
                params['out_dir'] = path_current
                create_folder(params['out_dir'])
                train_ncf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, inter=inter)
                np.savez(path_current + 'hyperparams.npz', lW=params['lW'], lH=params['lH'])

    return


def get_optimal_ncf(range_inter, range_nl_di):

    val_ndcg = np.zeros((len(range_inter), len(range_nl_di)))
    lambW, lambH = np.zeros((len(range_inter), len(range_nl_di))), np.zeros((len(range_inter), len(range_nl_di)))

    # Load all validation results
    for ii, inter in enumerate(range_inter):
        for idi, nl_di in enumerate(range_nl_di):
            path_current = 'outputs/warm/ncf/' + inter + '/' + str(nl_di) + '/'
            val_ndcg[ii, idi] = np.max(np.load(path_current + 'training.npz')['val_ndcg'])
            lambload = np.load(path_current + 'hyperparams.npz')
            lambW[ii, idi], lambH[ii, idi] =  lambload['lW'], lambload['lH']

    # Find the optimal set of hyperparams and record it
    ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
    inter_opt, nl_di_opt = range_inter[ind_opt[0]], range_nl_di[ind_opt[1]]
    lW_opt, lH_opt = lambW[ind_opt[0], ind_opt[1]], lambH[ind_opt[0], ind_opt[1]]
    np.savez('outputs/warm/ncf/hyperparams.npz', lW=lW_opt, lH=lH_opt, inter=inter_opt, nl_di=nl_di_opt)

    # Also record the overall validation scores (for plotting)
    np.savez('outputs/warm/ncf/val_results.npz', val_ndcg=val_ndcg)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(torch.cuda.get_device_name(device)))

    # Set parameters
    params = {'batch_size': 16, # 128
              'n_embeddings': 128,
              'n_epochs': 1, # 100
              'lr': 1e-4,
              'device': device
              }
    data_dir = 'data/'

    # Define the hyperparameters over which performing a grid search
    range_lW, range_lH = [0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10]
    #range_lW, range_lH,  = [0.1], [0.1]
    range_inter, range_nl_di = ['mult', 'conc'], [-1, 0, 1, 2, 3, 4, 5]

    # Training with validation
    train_val_ncf(params, range_lW, range_lH, range_inter, range_nl_di, data_dir=data_dir)
    plot_val_ndcg_ncf()

# EOF
