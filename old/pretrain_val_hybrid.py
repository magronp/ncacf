#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers.utils import create_folder
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributes, DatasetAttributesRatings
from helpers.utils import compute_factor_wmf_deep, init_weights, wpe_hybrid_strict
from helpers.models import ModelAttributes
from helpers.eval import evaluate_mf_hybrid

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def pretrain_relaxed(params):

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Path for the features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = params['data_dir'] + 'train_feats.num.csv'
    path_wmf_temp = os.path.join(params['out_dir'], 'params_wmf_temp.npz')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))
    confT = conf.T.tocsr()
    # Model parameters and definition
    my_model = ModelAttributes(params['n_features_in'], params['n_features_hidden'],
                               params['n_embeddings']).to(params['device'])
    my_model.apply(init_weights)
    print('Amount of parameters: {}'.format(sum([p.numel() for p in my_model.parameters()])), flush=True)

    print('\n Update WMF factors...')
    start_time_wmf = time.time()
    W, H = None, np.random.randn(n_songs_train, params['n_embeddings']).astype('float32') * 0.01
    for iwmf in range(params['n_iter_wmf']):
        W = compute_factor_wmf_deep(H, conf, params['lW'])
        H = compute_factor_wmf_deep(W, confT, params['lH'])
    time_als = time.time() - start_time_wmf

    # Save the WMF parameters for defining the dataset subsequently
    np.savez(path_wmf_temp, W=W, H=H)
    my_dataset = DatasetAttributes(wmf_path=path_wmf_temp, features_path=path_features)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Model update
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
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
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, time_als=time_als,
             val_ndcg=val_ndcg_tot)

    return


def pretrain_strict(params):

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = params['data_dir'] + 'train_feats.num.csv'

    # Model parameters and definition
    my_model = ModelAttributes(params['n_features_in'], params['n_features_hidden'],
                               params['n_embeddings']).to(params['device'])
    my_model.apply(init_weights)
    print('Amount of parameters: {}'.format(sum([p.numel() for p in my_model.parameters()])), flush=True)

    # Define the dataset
    my_dataset = DatasetAttributesRatings(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Optimizer
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Load the WMF factors (no need to recompute them: same as in baseline_relaxed)
    wmf_loader = np.load('outputs/pretraining_hybrid/relaxed/lW_' + str(params['lW']) + '/lH_' + str(params['lH']) + '/wmf.npz')
    W, H = wmf_loader['W'], wmf_loader['H']

    # Model update
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
            np.savez(os.path.join(params['out_dir'], 'wmf.npz'), W=W, H=H)

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def pretrain(params, variant):

    if variant == 'relaxed':
        pretrain_relaxed(params)
    else:
        pretrain_strict(params)

    return


def pretrain_val_mh_hybrid_out(params, range_lW, range_lH):

    # Pretraining
    for variant in ['relaxed', 'strict']:
        for lW in range_lW:
            for lH in range_lH:
                print(lW, lH)
                params['lW'], params['lH'] = lW, lH
                params['out_dir'] = 'outputs/out/pretraining_hybrid/' + variant + '/lW_' + str(lW) + '/lH_' + str(lH) + '/'
                create_folder(params['out_dir'])
                pretrain(params, variant)

    return


def opt_plot_mf_hybrid(range_lW, range_lH, n_epochs, variant='relaxed'):

    path_current = 'outputs/pretraining_hybrid/' + variant + '/'

    # Load the validation score for the pretrained models
    if variant == 'strict':
        range_lH = range_lH[:-1]
    Nw, Nh = len(range_lW), len(range_lH)
    val_ndcg = np.zeros((Nw, Nh, n_epochs))
    for iW, lW in enumerate(range_lW):
        for iH, lH in enumerate(range_lH):
            path_ndcg = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/training.npz'
            val_ndcg[iW, iH, :] = np.load(path_ndcg)['val_ndcg'][:n_epochs] * 100

    # Get the optimal hyperparameters
    ind_opt = np.unravel_index(np.argmax(val_ndcg, axis=None), val_ndcg.shape)
    lW_opt, lH_opt = range_lW[ind_opt[0]], range_lH[ind_opt[1]]
    np.savez(path_current + 'hyperparams.npz', lW=lW_opt, lH=lH_opt)

    # Get the optimal model and corresponding training log and copy it
    path_opt = path_current + 'lW_' + str(lW_opt) + '/lH_' + str(lH_opt) + '/'
    train_log = np.load(path_opt + 'training.npz')
    model_opt = torch.load(path_opt + 'model.pt')
    np.savez(path_current + 'training.npz', loss=train_log['loss'], time=train_log['time'], val_ndcg=train_log['val_ndcg'])
    torch.save(model_opt, path_current + 'model.pt')

    # Plot the results
    plt.figure()
    for il, l in enumerate(range_lW):
        plt.subplot(2, Nw//2, il+1)
        plt.plot(np.arange(n_epochs)+1, val_ndcg[il, :, :].T)
        plt.title(r'$\lambda_W$=' + str(l))
        if il > 2:
            plt.xlabel('Epochs')
        if il == 0 or il == 3:
            plt.ylabel('NDCG (%)')
    leg_lambda = [r'$\lambda_H$=' + str(lh) for lh in range_lH]
    #plt.legend(leg_lambda)
    plt.show()

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(torch.cuda.get_device_name(device)))

    # Parameters and range of the hyper parameters
    params = {'batch_size': 128, 'n_embeddings': 128, 'n_epochs': 50, 'n_iter_wmf': 20, 'lr': 1e-4,
              'n_features_hidden': 1024, 'n_features_in': 168, 'data_dir': 'data/out/', 'device': device}

    range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]
    
    # Pretraining - out-of-matrix
    pretrain_val_mh_hybrid_out(params, range_lW, range_lH)

    # Get the optimal models and plot the validation results
    opt_plot_mf_hybrid(range_lW, range_lH, params['n_epochs'], variant='relaxed')
    opt_plot_mf_hybrid(range_lW, range_lH, params['n_epochs'], variant='strict')

# EOF
