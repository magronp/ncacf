#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from helpers.training import train_mf_uni_out


def train_val_mf_uni_out(params, range_lW, range_lH):

    path_current = 'outputs/out/mf_uni/'
    '''
    # Pretraining with grid search on the hyperparameters
    # Relaxed variant
    variant = 'relaxed'
    for lW in range_lW:
        for lH in range_lH:
            print(lW, lH)
            params['lW'], params['lH'] = lW, lH
            params['out_dir'] = path_current + variant + '/lW_' + str(lW) + '/lH_' + str(lH) + '/'
            create_folder(params['out_dir'])
            train_mf_uni_out(params, variant=variant)
    get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])

    # Strict variant
    variant = 'strict'
    for lW in range_lW:
        print(lW)
        params['lW'], params['lH'] = lW, 0.
        params['out_dir'] = path_current + variant + '/lW_' + str(lW) + '/'
        create_folder(params['out_dir'])
        train_mf_uni_out(params, variant=variant)
    '''
    get_optimal_val_model_strict(path_current, range_lW, params['n_epochs'])

    return


def train_noval_mf_uni_relaxed_out(params, lW=0.1, lH=1.):

    params['lW'], params['lH'] = lW, lH
    params['out_dir'] = 'outputs/out/mf_uni/relaxed/'
    create_folder(params['out_dir'])
    train_mf_uni_out(params, variant='relaxed')

    return


def train_noval_mf_uni_strict_out(params, lW=0.1):

    params['lW'], params['lH'] = lW, 0.
    params['out_dir'] = 'outputs/out/mf_uni/strict/'
    create_folder(params['out_dir'])
    train_mf_uni_out(params, variant='strict')

    return


'''
def train_all_mf_uni(params):

    for variant in ['relaxed', 'strict']:
        params['out_dir'] = 'outputs/MFUni/' + variant + '/'
        create_folder(params['out_dir'])
        path_pretrain = 'outputs/pretraining_uni/' + variant + '/'
        train_mf_uni(params, path_pretrain=path_pretrain, variant=variant)

    return
'''

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
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'n_epochs': 100,
              'lr': 1e-4,
              'data_dir': 'data/out/',
              'device': device
              }

    val_lambda = True
    if val_lambda:
        # Training and validation for the hyperparameters
        range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10]
        train_val_mf_uni_out(params, range_lW, range_lH)

        # Plot the validation results
        plot_val_ndcg_lW_lH('outputs/out/mf_uni/relaxed/')
        plot_val_ndcg_lW('outputs/out/mf_uni/strict/')
    else:
        # Single training with pre-defined hyperparameter
        train_noval_mf_uni_relaxed_out(params, lW=0.1, lH=1.)
        train_noval_mf_uni_strict_out(params, lW=1.)

    # Test
    '''
    my_model = torch.load('outputs/MFUni/relaxed/model.pt')
    print(evaluate_uni(params, my_model, split='test'))
    print(np.load('outputs/MFUni/relaxed/training.npz')['time'] + np.load('outputs/pretraining_uni/relaxed/training.npz')['time'])
    my_model = torch.load('outputs/MFUni/strict/model.pt')
    print(evaluate_uni(params, my_model, split='test'))
    print(np.load('outputs/MFUni/strict/training.npz')['time'] + np.load('outputs/pretraining_uni/strict/training.npz')['time'])
    '''

# EOF
