#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from helpers.training import train_ncf_in


def train_main_ncf(params, range_lW, range_lH, data_dir = 'data/'):

    path_current = 'outputs/in/ncf/'
    params['data_dir'] = data_dir + '/in/'
    # Training with grid search on the hyperparameters
    for lW in range_lW:
        for lH in range_lH:
            print(lW, lH)
            params['lW'], params['lH'] = lW, lH
            params['out_dir'] = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/'
            create_folder(params['out_dir'])
            train_ncf_in(params)
    get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(torch.cuda.get_device_name(device)))

    # Set parameters
    params = {'batch_size': 8,
              'n_embeddings': 128,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'n_epochs': 100,
              'lr': 1e-4,
              'device': device
              }

    data_dir = 'data/'
    # Training and validation for the hyperparameters
    #range_lW, range_lH = [0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10]
    range_lW, range_lH = [1], [1]
    train_main_ncf(params, range_lW, range_lH, data_dir)

# EOF
