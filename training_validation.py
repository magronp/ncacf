#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import torch
import sys
from training.twostages import train_val_wmf_2stages, get_optimal_2stages, get_optimal_wmf
from training.mf_hybrid import train_val_mf_hybrid, check_NGD_mf_hybrid
from training.mf_uni import train_val_mf_uni
from training.ncf import train_val_ncf, get_optimal_ncf
from training.ncacf import train_val_ncacf, get_optimal_ncacf


if __name__ == '__main__':

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(device))

    # Set parameters
    params = {'batch_size': 128,
              'n_embeddings': 128,
              'lr': 1e-4,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'device': device}
    data_dir = 'data/'

    # Define the settings (warm and cold start) and the variants (relaxed and strict)
    setting_list = ['warm', 'cold']
    variant_list = ['relaxed', 'strict']

    # List of models to train/validate
    model_list = sys.argv[1:]
    #model_list = ['ncf']

    for model in model_list:

        # WMF and 2-stage approaches - training with validation and model selection
        if model == 'twostages':
            params['n_iter_wmf'] = 30
            params['n_epochs'] = 150
            range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]
            train_val_wmf_2stages(setting_list, variant_list, params, range_lW, range_lH, data_dir)
            get_optimal_2stages(setting_list, variant_list, range_lW, range_lH, params['n_epochs'])
            get_optimal_wmf(params, range_lW, range_lH)

        # MF-Hybrid models - training with validation, and check the impact of N_GD
        elif model == 'mf_hybrid':
            params['n_epochs'] = 150
            range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]
            train_val_mf_hybrid(setting_list, variant_list, params, range_lW, range_lH, data_dir)
            n_ep_it_list = [2, 5, 10]
            check_NGD_mf_hybrid(setting_list, variant_list, n_ep_it_list, params, data_dir)

        # MF-Uni models - training with validation
        elif model == 'mf_uni':
            params['n_epochs'] = 150
            range_lW, range_lH = [0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10]
            train_val_mf_uni(setting_list, variant_list, params, range_lW, range_lH, data_dir)

        # NCF baseline - training with validation (lambda, interaction model, and number of layers)
        elif model == 'ncf':
            range_lW, range_lH, = [0.1], [0.1]
            params['n_epochs'] = 100
            #range_inter, range_nl_di = ['mult', 'conc'], [-1, 0, 1, 2, 3, 4]
            range_inter, range_nl_di = ['conc'], [2, 3, 4]
            train_val_ncf(params, range_lW, range_lH, range_inter, range_nl_di, data_dir)
            range_inter, range_nl_di = ['mult', 'conc'], [-1, 0, 1, 2, 3, 4]
            get_optimal_ncf(range_inter, range_nl_di)

        # NCACF - training with validation (interaction model, number of layers, variant)
        elif model == 'ncacf':
            range_lW, range_lH, = [0.1], [0.1]
            params['n_epochs'] = 100
            #range_inter, range_nl_di = ['mult', 'conc'], [-1, 0, 1, 2, 3, 4]
            range_inter, range_nl_di = ['conc'], [4]
            train_val_ncacf(setting_list, variant_list, params, range_lW, range_lH, range_inter, range_nl_di, data_dir)
            get_optimal_ncacf(setting_list, variant_list, range_inter, range_nl_di)

        else:
            print('Unknown model')

# EOF
