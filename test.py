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
from helpers.eval import evaluate_mf_hybrid, evaluate_uni


def train_test_wmf(params, k_split, data_dir='data/'):
    params['data_dir'] = data_dir + 'warm' + '/split' + str(k_split) + '/'
    lamda_opt = np.load('outputs/warm/WMF/hyperparams.npz')
    params['lW'], params['lH'] = lamda_opt['lW'], lamda_opt['lH']
    W, H = train_wmf(params, setting='warm', rec_model=False)
    test_ndcg = evaluate_mf_hybrid(params, W, H, None, setting='warm', variant='relaxed', split='test')
    return test_ndcg


def train_test_2stages(params, setting, variant, k_split, data_dir='data/'):
    params['data_dir'] = data_dir + setting + '/split' + str(k_split) + '/'
    lamda_opt = np.load('outputs/' + setting + '/2stages/' + variant + '/hyperparams.npz')
    params['lW'], params['lH'] = lamda_opt['lW'], lamda_opt['lH']
    params['out_dir'] = 'outputs/temp/'
    W, H = train_wmf(params, setting=setting)
    model_opt = train_2stages(params, variant=variant, setting=setting, rec_model=False)
    test_ndcg = evaluate_mf_hybrid(params, W, H, model_opt, setting=setting, variant=variant, split='test')
    return test_ndcg


def train_test_mfhybrid(params, setting, variant, k_split, data_dir='data/'):
    params['data_dir'] = data_dir + setting + '/split' + str(k_split) + '/'
    lamda_opt = np.load('outputs/' + setting + '/mf_hybrid/' + variant + '/hyperparams.npz')
    params['lW'], params['lH'] = lamda_opt['lW'], lamda_opt['lH']
    params['out_dir'] = 'outputs/temp/'
    params['n_ep_it'] = 1
    model_opt, W, H, time_opt = train_mf_hybrid(params, variant=variant, setting=setting, rec_model=False)
    test_ndcg = evaluate_mf_hybrid(params, W, H, model_opt, setting=setting, variant=variant, split='test')
    return test_ndcg, time_opt


def train_test_mfuni(params, setting, variant, k_split, data_dir='data/'):
    params['data_dir'] = data_dir + setting + '/split' + str(k_split) + '/'
    lamda_opt = np.load('outputs/' + setting + '/mf_uni/' + variant + '/hyperparams.npz')
    params['lW'], params['lH'] = float(lamda_opt['lW']), float(lamda_opt['lH'])
    params['out_dir'] = 'outputs/temp/'
    model_opt, time_opt = train_mf_uni(params, variant=variant, setting=setting, rec_model=False)
    test_ndcg = evaluate_uni(params, model_opt, setting=setting, split='test')
    return test_ndcg, time_opt


def train_test_ncf(params, k_split, data_dir='data/'):
    params['data_dir'] = data_dir + 'warm' + '/split' + str(k_split) + '/'
    hyper_opt = np.load('outputs/warm/ncf//hyperparams.npz')
    params['lW'], params['lH'] = float(hyper_opt['lW']), float(hyper_opt['lH'])
    ni_dl, inter = int(hyper_opt['nl_di']), hyper_opt['inter']
    params['out_dir'] = 'outputs/temp/'
    # first pretrain a shallow NCF model (no deep interaction layer, no nonlinear activation)
    train_ncf(params, path_pretrain=None, n_layers_di=-1, inter=inter, rec_model=True)
    # and then train the complete model with the right amount of deep layers
    model_opt = train_ncf(params, path_pretrain=params['out_dir'], n_layers_di=ni_dl, inter=inter, rec_model=False)
    test_ndcg = evaluate_uni(params, model_opt, setting='warm', split='test')
    return test_ndcg


# Set random seed for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

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
n_splits = 10
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
time_mf_cold = {'hybrid_relaxed': np.zeros(n_splits),
                'hybrid_strict': np.zeros(n_splits),
                'uni_relaxed': np.zeros(n_splits),
                'uni_strict': np.zeros(n_splits),
                }
time_mf_warm = time_mf_cold


k_split = 0

# WMF - only in the warm-start setting
ndcg_warm['wmf'][k_split] = train_test_wmf(params, k_split, data_dir)

# DCB (corresponds to the '2 stage'-approach) - warm  (strict variant) and cold (relaxed variant)
ndcg_warm['dcb'][k_split] = train_test_2stages(params, 'warm', 'strict', k_split, data_dir)
ndcg_cold['dcb'][k_split] = train_test_2stages(params, 'cold', 'relaxed', k_split, data_dir)

# CDL - correspond to MF-Hybrid in the relaxed variant
ndcg_warm['cdl'][k_split], time_mf_warm['hybrid_relaxed'] = train_test_mfhybrid(params, 'warm', 'relaxed', k_split, data_dir)
ndcg_cold['cdl'][k_split], time_mf_cold['hybrid_relaxed'] = train_test_mfhybrid(params, 'cold', 'relaxed', k_split, data_dir)

# get also mf hybrid strict results, just for the computation time
ndcg_warm['cdl'][k_split], time_mf_warm['hybrid_strict'] = train_test_mfhybrid(params, 'warm', 'strict', k_split, data_dir)
ndcg_cold['cdl'][k_split], time_mf_cold['hybrid_strict'] = train_test_mfhybrid(params, 'cold', 'strict', k_split, data_dir)

# DCUE - correponds to MF-Uni in the strict variant
ndcg_warm['dcue'][k_split], time_mf_warm['uni_strict'] = train_test_mfuni(params, 'warm', 'strict', k_split, data_dir)
ndcg_cold['dcue'][k_split], time_mf_cold['uni_strict'] = train_test_mfuni(params, 'cold', 'strict', k_split, data_dir)

# CCCFnet - corresponds to MF-Uni in the relaxed variant
ndcg_warm['cccfnet'][k_split], time_mf_warm['uni_relaxed'] = train_test_mfuni(params, 'warm', 'relaxed', k_split, data_dir)
ndcg_cold['cccfnet'][k_split], time_mf_cold['uni_relaxed'] = train_test_mfuni(params, 'cold', 'relaxed', k_split, data_dir)

# NCF (only 1 variant and for the warm-start scenario)
params['n_epochs'] = 100
ndcg_warm['ncf'][k_split] = train_test_ncf(params, k_split, data_dir=data_dir)


