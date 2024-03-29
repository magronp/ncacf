#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import argparse
import torch
from training.twostages import train_val_wmf_2stages, get_optimal_2stages, get_optimal_wmf
from training.mf_hybrid import train_val_mf_hybrid, check_NGD_mf_hybrid
from training.mf_uni import train_val_mf_uni
from training.ncf import train_val_ncf, get_optimal_ncf
from training.ncacf import train_val_ncacf, get_optimal_ncacf
from helpers.utils import get_optimal_val_model_lambda


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

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--Models", nargs='*', help="Models to test",
                        default=['twostages', 'mf_hybrid', 'mf_uni', 'ncf', 'ncacf'])
    parser.add_argument("-s", "--Settings", nargs='*', help="Warm and/or cold start settings",
                        default=['warm', 'cold'])
    parser.add_argument("-v", "--Variants", nargs='*', help="Variant (strict or relaxed)",
                        default=['relaxed', 'strict'])
    parser.add_argument("-i", "--Inter", nargs='*', help="Interaction model (mult or conc)",
                        default=['mult', 'conc'])
    parser.add_argument("-l", "--Layers", nargs='*', help="Number of layers in the interaction network",
                        default=[-1, 0, 1, 2, 3, 4, 5])
    args = parser.parse_args()

    model_list = args.Models
    setting_list = args.Settings
    variant_list = args.Variants
    range_inter = args.Inter
    range_nl_di = list(map(int, args.Layers))

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
            get_optimal_val_model_lambda('mf_hybrid', setting_list, variant_list, params['n_epochs'], range_lW,
                                         range_lH)
            n_ep_it_list = [2, 5, 10]
            check_NGD_mf_hybrid(setting_list, variant_list, n_ep_it_list, params, data_dir)

        # MF-Uni models - training with validation
        elif model == 'mf_uni':
            params['n_epochs'] = 150
            range_lW, range_lH = [0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10]
            train_val_mf_uni(setting_list, variant_list, params, range_lW, range_lH, data_dir)
            get_optimal_val_model_lambda('mf_uni', setting_list, variant_list, params['n_epochs'], range_lW,
                                         range_lH)

        # NCF baseline - training with validation (lambda, interaction model, and number of layers)
        elif model == 'ncf':
            range_lW, range_lH, = [0.1], [0.1]
            params['n_epochs'] = 100
            train_val_ncf(params, range_lW, range_lH, range_inter, range_nl_di, data_dir)
            get_optimal_ncf(range_inter, range_nl_di)

        # NCACF - training with validation (interaction model, number of layers, variant)
        elif model == 'ncacf':
            params['n_epochs'] = 100
            train_val_ncacf(setting_list, variant_list, params, range_inter, range_nl_di, data_dir)
            get_optimal_ncacf(setting_list, variant_list, range_inter, range_nl_di)

        else:
            print('Unknown model')

# EOF
