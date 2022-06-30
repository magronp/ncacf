#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import torch
from helpers.utils import create_folder
from training.twostages import train_wmf, train_2stages
from training.mf_hybrid import train_mf_hybrid
from training.mf_uni import train_mf_uni
from training.ncf import train_ncf
from training.ncacf import train_ncacf
from helpers.eval import evaluate_mf_hybrid, evaluate_uni
from os.path import exists
import sys


def train_test_wmf(params, k_split, data_dir='data/'):

    # Define the data directory
    params['data_dir'] = data_dir + 'warm' + '/split' + str(k_split) + '/'

    # Hyperparameters
    path_hyperparams = 'outputs/warm/WMF/hyperparams.npz'
    file_exists = exists(path_hyperparams)
    if file_exists:
        lamda_opt = np.load('outputs/warm/WMF/hyperparams.npz')
        params['lW'], params['lH'] = lamda_opt['lW'], lamda_opt['lH']
    else:
        params['lW'], params['lH'] = 100, 100

    # Train and test
    W, H = train_wmf(params, setting='warm', rec_model=False)
    test_ndcg = evaluate_mf_hybrid(params, W, H, None, setting='warm', variant='relaxed', split='test')
    return test_ndcg


def train_test_2stages(params, setting, variant, k_split, data_dir='data/'):

    # Define the data directory
    params['data_dir'] = data_dir + setting + '/split' + str(k_split) + '/'

    # Hyperparameters
    path_hyperparams = 'outputs/' + setting + '/2stages/' + variant + '/hyperparams.npz'
    file_exists = exists(path_hyperparams)
    if file_exists:
        lamda_opt = np.load(path_hyperparams)
        params['lW'], params['lH'] = lamda_opt['lW'], lamda_opt['lH']
    else:
        if setting == 'warm' and variant == 'strict':
            params['lW'], params['lH'] = 100, 100
        elif setting == 'cold' and variant == 'relaxed':
            params['lW'], params['lH'] = 0.1, 1
        elif setting == 'cold' and variant == 'strict':
            params['lW'], params['lH'] = 0.01, 0

    # Train and test
    params['out_dir'] = 'outputs/temp/'
    W, H = train_wmf(params, setting=setting)
    model_opt = train_2stages(params, variant=variant, setting=setting, rec_model=False)
    test_ndcg = evaluate_mf_hybrid(params, W, H, model_opt, setting=setting, variant=variant, split='test')
    return test_ndcg


def train_test_mfhybrid(params, setting, variant, k_split, data_dir='data/'):

    # Define the data directory
    params['data_dir'] = data_dir + setting + '/split' + str(k_split) + '/'

    # Hyperparameters
    path_hyperparams = 'outputs/' + setting + '/mf_hybrid/' + variant + '/hyperparams.npz'
    file_exists = exists(path_hyperparams)
    if file_exists:
        lamda_opt = np.load(path_hyperparams)
        params['lW'], params['lH'] = lamda_opt['lW'], lamda_opt['lH']
    else:
        if setting == 'warm' and variant == 'relaxed':
            params['lW'], params['lH'] = 1000, 10
        elif setting == 'warm' and variant == 'strict':
            params['lW'], params['lH'] = 0.1, 0
        elif setting == 'cold' and variant == 'relaxed':
            params['lW'], params['lH'] = 0.1, 10
        elif setting == 'cold' and variant == 'strict':
            params['lW'], params['lH'] = 0.01, 0

    # Train and test
    params['out_dir'] = 'outputs/temp/'
    params['n_ep_it'] = 1
    model_opt, W, H, _ = train_mf_hybrid(params, variant=variant, setting=setting, rec_model=False)
    test_ndcg = evaluate_mf_hybrid(params, W, H, model_opt, setting=setting, variant=variant, split='test')
    return test_ndcg


def train_test_mfuni(params, setting, variant, k_split, data_dir='data/'):

    # Define the data directory
    params['data_dir'] = data_dir + setting + '/split' + str(k_split) + '/'

    # Hyperparameters
    path_hyperparams = 'outputs/' + setting + '/mf_uni/' + variant + '/hyperparams.npz'
    file_exists = exists(path_hyperparams)
    if file_exists:
        lamda_opt = np.load(path_hyperparams)
        params['lW'], params['lH'] = float(lamda_opt['lW']), float(lamda_opt['lH'])
    else:
        if setting == 'warm' and variant == 'relaxed':
            params['lW'], params['lH'] = 0, 0
        elif setting == 'warm' and variant == 'strict':
            params['lW'], params['lH'] = 0.1, 0
        elif setting == 'cold' and variant == 'relaxed':
            params['lW'], params['lH'] = 0.1, 1
        elif setting == 'cold' and variant == 'strict':
            params['lW'], params['lH'] = 1, 0

    # Train and test
    params['out_dir'] = 'outputs/temp/'
    model_opt, _ = train_mf_uni(params, variant=variant, setting=setting, rec_model=False)
    test_ndcg = evaluate_uni(params, model_opt, setting=setting, split='test')
    return test_ndcg


def train_test_ncf(params, k_split, data_dir='data/'):

    # Define the data directory
    params['data_dir'] = data_dir + 'warm' + '/split' + str(k_split) + '/'

    # Hyperparameters
    path_hyperparams = 'outputs/warm/ncacf//hyperparams.npz'
    file_exists = exists(path_hyperparams)
    if file_exists:
        hyper_opt = np.load(path_hyperparams)
        params['lW'], params['lH'] = float(hyper_opt['lW']), float(hyper_opt['lH'])
        ni_dl, inter = int(hyper_opt['nl_di']), hyper_opt['inter']
    else:
        params['lW'], params['lH'] = 0.1, 0.1
        ni_dl, inter = 2, 'mult'

    # Train and test
    params['out_dir'] = 'outputs/temp/'
    # first pretrain a shallow NCF model (no deep interaction layer, no nonlinear activation)
    train_ncf(params, path_pretrain=None, n_layers_di=-1, inter=inter, rec_model=True)
    # and then train the complete model with the right amount of deep layers
    model_opt = train_ncf(params, path_pretrain=params['out_dir'], n_layers_di=ni_dl, inter=inter, rec_model=False)
    test_ndcg = evaluate_uni(params, model_opt, setting='warm', split='test')
    return test_ndcg


def train_test_ncacf(params, setting, k_split, data_dir='data/'):

    # Define the data directory
    params['data_dir'] = data_dir + setting + '/split' + str(k_split) + '/'

    # Hyperparameters
    path_hyperparams = 'outputs/' + setting + '/ncacf//hyperparams.npz'
    file_exists = exists(path_hyperparams)
    if file_exists:
        hyper_opt = np.load(path_hyperparams)
        params['lW'], params['lH'] = float(hyper_opt['lW']), float(hyper_opt['lH'])
        ni_dl, inter, variant = int(hyper_opt['nl_di']), hyper_opt['inter'], hyper_opt['variant']
    else:
        params['lW'], params['lH'] = 0.1, 0.1
        ni_dl, inter, variant = 2, 'mult', 'relaxed'

    # Train and test
    params['out_dir'] = 'outputs/temp/'
    # first pretrain a shallow NCACF model (no deep interaction layer, no nonlinear activation)
    train_ncacf(params, path_pretrain=None, n_layers_di=-1, setting=setting, variant=variant, inter=inter,
                rec_model=True)
    # and then train the complete model with the right amount of deep layers
    model_opt = train_ncacf(params, path_pretrain=params['out_dir'], n_layers_di=-ni_dl, setting=setting,
                            variant=variant, inter=inter, rec_model=False)
    test_ndcg = evaluate_uni(params, model_opt, setting=setting, split='test')
    return test_ndcg


if __name__ == '__main__':

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(device))
    create_folder('outputs/temp/')

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

    # Create the dictionaries for storing the results
    path_out = 'outputs/test_results.npz'
    file_exists = exists(path_out)
    n_splits = 10

    if not(file_exists):
        ndcg_warm = {'wmf': np.zeros(n_splits),
                     'dcb': np.zeros(n_splits),
                     'cdl': np.zeros(n_splits),
                     'dcue': np.zeros(n_splits),
                     'cccfnet': np.zeros(n_splits),
                     'ncf': np.zeros(n_splits),
                     'ncacf': np.zeros(n_splits)
                     }
        ndcg_cold = {'dcb': np.zeros(n_splits),
                     'cdl': np.zeros(n_splits),
                     'dcue': np.zeros(n_splits),
                     'cccfnet': np.zeros(n_splits),
                     'ncacf': np.zeros(n_splits)
                     }
        np.savez(path_out, ndcg_warm=ndcg_warm, ndcg_cold=ndcg_cold)

    split_list = range(n_splits)

    # List of baselines and methods to test
    #model_list = sys.argv[1:]
    model_list = ['wmf', 'dcb', 'cdl', 'dcue', 'cccfnet', 'ncf', 'ncacf']

    for model in model_list:
        for k_split in split_list:
            print('Model : ' + model + ' ------ Split : ' + str(k_split))

            # WMF - only in the warm-start setting
            if model == 'wmf':
                params['n_epochs'] = 150
                testndcg = train_test_wmf(params, k_split, data_dir)
                ndcg_loader = np.load(path_out, allow_pickle=True)['ndcg_warm']
                ndcg_warm, ndcg_cold = ndcg_loader['ndcg_warm'], ndcg_loader['ndcg_cold']
                ndcg_warm['wmf'][k_split] = testndcg
                np.savez(path_out, ndcg_warm=ndcg_warm, ndcg_cold=ndcg_cold)

            # DCB (corresponds to the '2 stage'-approach) - warm  (strict variant) and cold (relaxed variant)
            elif model == 'dcb':
                params['n_epochs'] = 150
                testndcg_w = train_test_2stages(params, 'warm', 'strict', k_split, data_dir)
                testndcg_c = train_test_2stages(params, 'cold', 'relaxed', k_split, data_dir)
                ndcg_loader = np.load(path_out, allow_pickle=True)['ndcg_warm']
                ndcg_warm, ndcg_cold = ndcg_loader['ndcg_warm'], ndcg_loader['ndcg_cold']
                ndcg_warm['dcb'][k_split], ndcg_cold['dcb'][k_split] = testndcg_w, testndcg_c
                np.savez(path_out, ndcg_warm=ndcg_warm, ndcg_cold=ndcg_cold)

            # CDL - correspond to MF-Hybrid in the relaxed variant
            elif model == 'cdl':
                params['n_epochs'] = 150
                testndcg_w = train_test_mfhybrid(params, 'warm', 'relaxed', k_split, data_dir)
                testndcg_c = train_test_mfhybrid(params, 'cold', 'relaxed', k_split, data_dir)
                ndcg_loader = np.load(path_out, allow_pickle=True)['ndcg_warm']
                ndcg_warm, ndcg_cold = ndcg_loader['ndcg_warm'], ndcg_loader['ndcg_cold']
                ndcg_warm['cdl'][k_split], ndcg_cold['cdl'][k_split] = testndcg_w, testndcg_c
                np.savez(path_out, ndcg_warm=ndcg_warm, ndcg_cold=ndcg_cold)

            # DCUE - correponds to MF-Uni in the strict variant
            elif model == 'dcue':
                params['n_epochs'] = 150
                testndcg_w = train_test_mfuni(params, 'warm', 'strict', k_split, data_dir)
                testndcg_c = train_test_mfuni(params, 'cold', 'strict', k_split, data_dir)
                ndcg_loader = np.load(path_out, allow_pickle=True)['ndcg_warm']
                ndcg_warm, ndcg_cold = ndcg_loader['ndcg_warm'], ndcg_loader['ndcg_cold']
                ndcg_warm['dcue'][k_split], ndcg_cold['dcue'][k_split] = testndcg_w, testndcg_c
                np.savez(path_out, ndcg_warm=ndcg_warm, ndcg_cold=ndcg_cold)

            # CCCFnet - corresponds to MF-Uni in the relaxed variant
            elif model == 'cccfnet':
                params['n_epochs'] = 150
                testndcg_w = train_test_mfuni(params, 'warm', 'relaxed', k_split, data_dir)
                testndcg_c = train_test_mfuni(params, 'cold', 'relaxed', k_split, data_dir)
                ndcg_loader = np.load(path_out, allow_pickle=True)['ndcg_warm']
                ndcg_warm, ndcg_cold = ndcg_loader['ndcg_warm'], ndcg_loader['ndcg_cold']
                ndcg_warm['cccfnet'][k_split], ndcg_cold['cccfnet'][k_split] = testndcg_w, testndcg_c
                np.savez(path_out, ndcg_warm=ndcg_warm, ndcg_cold=ndcg_cold)

            # NCF (only 1 variant and for the warm-start scenario)
            elif model == 'ncf':
                params['n_epochs'] = 100
                testndcg_w = train_test_ncf(params, k_split, data_dir=data_dir)
                ndcg_loader = np.load(path_out, allow_pickle=True)['ndcg_warm']
                ndcg_warm, ndcg_cold = ndcg_loader['ndcg_warm'], ndcg_loader['ndcg_cold']
                ndcg_warm['ncf'][k_split] = testndcg_w
                np.savez(path_out, ndcg_warm=ndcg_warm, ndcg_cold=ndcg_cold)

            # NCACF (get the optimal variant in the warm- and cold-start scenarios)
            elif model == 'ncf':
                params['n_epochs'] = 100
                testndcg_w = train_test_ncacf(params, 'warm', k_split, data_dir=data_dir)
                testndcg_c = train_test_ncacf(params, 'cold', k_split, data_dir=data_dir)
                ndcg_loader = np.load(path_out, allow_pickle=True)['ndcg_warm']
                ndcg_warm, ndcg_cold = ndcg_loader['ndcg_warm'], ndcg_loader['ndcg_cold']
                ndcg_warm['ncacf'][k_split], ndcg_cold['ncacf'][k_split] = testndcg_w, testndcg_c
                np.savez(path_out, ndcg_warm=ndcg_warm, ndcg_cold=ndcg_cold)

# EOF
