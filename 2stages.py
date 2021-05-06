#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers.utils import create_folder, plot_val_ndcg_lW_lH
from helpers.training import train_wmf, train_2stages_relaxed, train_2stages_strict
import numpy as np
import torch

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def train_val_2stages_out(path_current, params, range_lW, range_lH):

    # Loop over hyperparameters
    for lW in range_lW:
        for lH in range_lH:
            print(lW, lH)
            params['lW'], params['lH'] = lW, lH
            params['out_dir'] = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/'
            create_folder(params['out_dir'])
            # First train the WMF
            train_wmf(params)
            # Then train the relaxed and strict variant on top of these WMFs
            train_2stages_relaxed(params)
            train_2stages_strict(params)

    # Get the optimal models after grid search
    get_optimal_2stages(path_current, range_lW, range_lH, params['n_epochs'], variant='relaxed')
    get_optimal_2stages(path_current, range_lW, range_lH, params['n_epochs'], variant='strict')

    return


def train_noval_2stages_out(path_current, params, lW=0.1, lH=1.):

    # Loop over hyperparameters
    params['lW'], params['lH'] = lW, lH
    params['out_dir'] = path_current + 'noval/'
    create_folder(params['out_dir'])
    # First train the WMF
    train_wmf(params)
    # Then train the relaxed and strict variant on top of these WMFs
    train_2stages_relaxed(params)
    train_2stages_strict(params)

    return


def get_optimal_2stages(path_current, range_lW, range_lH, n_epochs, variant='relaxed'):

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
              'n_iter_wmf': 30,
              'n_epochs': 100,
              'lr': 1e-4,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'data_dir': 'data/out/',
              'device': device}

    path_current = 'outputs/out/2stages/'
    train_b = False
    val_b = True

    if train_b:
        if val_b:
            # Training and validation for the hyperparameters
            range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]
            train_val_2stages_out(path_current, params, range_lW, range_lH)
        else:
            # Single training with pre-defined hyperparameter
            train_noval_2stages_out(path_current, params, lW=0.1, lH=1.)

    # Plot the validation results
    plot_val_ndcg_lW_lH(path_current + 'relaxed/')
    plot_val_ndcg_lW_lH(path_current + 'strict/')

# EOF
