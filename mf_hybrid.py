#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from helpers.training import train_mf_hybrid_out
import numpy as np
from matplotlib import pyplot as plt
import torch
from helpers.eval import evaluate_mf_hybrid


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

    val_lambda = True
    if val_lambda:
        # Training and validation for the hyperparameters
        range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10]
        train_val_mh_hybrid_out(params, range_lW, range_lH)

        # Plot the validation results
        plot_val_ndcg_lW_lH('outputs/out/mf_hybrid/relaxed/')
        plot_val_ndcg_lW('outputs/out/mf_hybrid/strict/')

        # Check what happens if ep_it varies
        train_mh_hybrid_out_epiter(params)
        check_mh_hybrid_out_epiter(params, variant='relaxed')
        check_mh_hybrid_out_epiter(params, variant='strict')
    else:
        # Single training with pre-defined hyperparameter
        train_noval_mf_hybrid_relaxed_out(params, lW=0.1, lH=1.)
        train_noval_mf_hybrid_strict_out(params, lW=1.)


# EOF
