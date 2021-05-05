#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributesRatings
from helpers.models import ModelMFuni, evaluate_uni
from helpers.metrics import wpe_joint
from helpers.functions import create_folder, init_model_joint
from matplotlib import pyplot as plt
from helpers.training import train_mf_uni



def train_mf_uni(params, path_pretrain=None, variant='relaxed'):

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelMFuni(n_users, n_songs_train, params['n_embeddings'], params['n_features_in'],
                          params['n_features_hidden'], variant)
    if path_pretrain is None:
        lW, lH = params['lW'], params['lH']
    else:
        #my_model = torch.load(path_pretrain + 'model.pt')
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


def pretrain_val_mf_uni_relaxed(params, range_lW, range_lH, plotval=True):

    # Training with many hyperparameters
    for lW in range_lW:
        for lH in range_lH:
            print(lW, lH)
            params['lW'], params['lH'] = lW, lH
            params['out_dir'] = 'outputs/MFUni/relaxed/lW_' + str(lW) + '/lH_' + str(lH) + '/'
            create_folder(params['out_dir'])
            train_mf_uni(params, path_pretrain=None, variant='relaxed')

    # Load the validation score for the pretrained models
    Nw, Nh = len(range_lW), len(range_lH)
    val_ndcg = np.zeros((Nw, Nh, params['n_epochs']))
    for iW, lW in enumerate(range_lW):
        for iH, lH in enumerate(range_lH):
            path_ndcg = 'outputs/MFUni/relaxed/lW_' + str(lW) + '/lH_' + str(lH) + '/training.npz'
            val_ndcg[iW, iH, :] = np.load(path_ndcg)['val_ndcg'] * 100

    # Get the optimal hyperparameters
    ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
    lW_opt, lH_opt = range_lW[ind_opt[0]], range_lH[ind_opt[1]]

    # Get the optimal model and corresponding training log and copy it
    path_opt = 'outputs/MFUni/relaxed/lW_' + str(lW_opt) + '/lH_' + str(lH_opt) + '/'
    train_log = np.load(path_opt + 'training.npz')
    model_opt = torch.load(path_opt + 'model.pt')
    np.savez('outputs/MFUni/relaxed/training.npz', loss=train_log['loss'], time=train_log['time'], val_ndcg=train_log['val_ndcg'])
    torch.save(model_opt, 'outputs/MFUni/relaxed/model.pt')

    # Plot the results
    if plotval:
        plt.figure()
        for il, l in enumerate(range_lW):
            plt.subplot(2, Nw//2, il+1)
            plt.plot(np.arange(params['n_epochs'])+1, val_ndcg[il, :, :].T)
            plt.title(r'$\lambda_W$=' + str(l))
            if il > 2:
                plt.xlabel('Epochs')
            if il == 0 or il == 3:
                plt.ylabel('NDCG (%)')
        leg_lambda = [r'$\lambda_H$=' + str(lh) for lh in range_lH]
        plt.legend(leg_lambda)
        plt.show()

    return


def pretrain_val_mf_uni_strict(params, range_lW, plotval=True):

    # Training with cross validation
    for lW in range_lW:
        print(lW)
        params['lW'], params['lH'] = lW, 0.
        params['out_dir'] = 'outputs/MFUni/strict/lW_' + str(lW) + '/'
        create_folder(params['out_dir'])
        train_mf_uni(params, path_pretrain=None, variant='strict')

    # Load the validation score for the pretrained models
    Nw = len(range_lW)
    val_ndcg = np.zeros((Nw, params['n_epochs']))
    for iW, lW in enumerate(range_lW):
        path_ndcg = 'outputs/MFUni/strict/lW_' + str(lW) + '/training.npz'
        val_ndcg[iW, :] = np.load(path_ndcg)['val_ndcg'] * 100

    # Get the optimal hyperparameters
    ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
    lW_opt = range_lW[ind_opt[0]]

    # Get the optimal model and corresponding training log and copy it
    path_opt = 'outputs/MFUni/strict/lW_' + str(lW_opt) + '/'
    train_log = np.load(path_opt + 'training.npz')
    model_opt = torch.load(path_opt + 'model.pt')
    np.savez('outputs/MFUni/strict/training.npz', loss=train_log['loss'], time=train_log['time'], val_ndcg=train_log['val_ndcg'])
    torch.save(model_opt, 'outputs/MFUni/strict/model.pt')

    # Plot the results
    if plotval:
        plt.figure()
        for il, l in enumerate(range_lW):
            plt.subplot(2, Nw//2, il+1)
            plt.plot(np.arange(params['n_epochs'])+1, val_ndcg[il, :].T)
            plt.title(r'$\lambda_W$=' + str(l))
            if il > 2:
                plt.xlabel('Epochs')
            if il == 0 or il == 3:
                plt.ylabel('NDCG (%)')
        plt.show()

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)

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
              'data_dir': 'data/',
              'device': 'cuda'
              }

    range_lW, range_lH = [0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10]
    pretrain_val_mf_uni_relaxed(params, range_lW, range_lH, plotval=True)
    pretrain_val_mf_uni_strict(params, range_lW, plotval=True)


    # Training
    #train_all_mf_uni(params)

    # Evaluation on the test set
    #my_model = torch.load('outputs/MFUni/relaxed/model.pt')
    #print(evaluate_uni(params, my_model, split='test'))
    #my_model = torch.load('outputs/MFUni/strict/model.pt')
    #print(evaluate_uni(params, my_model, split='test'))

# EOF
