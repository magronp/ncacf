#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from mf_uni import train_val_mf_uni_relaxed_out, train_val_mf_uni_strict_out, train_noval_mf_uni_out, test_mf_uni_out

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
              'out_sigm': True,
              'data_dir': 'data/out/',
              'device': device
              }

    path_current = 'outputs/out/gmf/'
    train_b = True
    val_b = True

    if train_b:
        if val_b:
            # Training and validation for the hyperparameters
            range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10]
            train_val_mf_uni_relaxed_out(path_current, params, range_lW, range_lH)
            train_val_mf_uni_strict_out(path_current, params, range_lW)
        else:
            # Single training with pre-defined hyperparameter
            train_noval_mf_uni_out(path_current, 'relaxed', params, lW=1., lH=1.)
            train_noval_mf_uni_out(path_current, 'strict', params, lW=1., lH=0.)

    # Plot the validation results
    plot_val_ndcg_lW_lH(path_current + 'relaxed/')
    plot_val_ndcg_lW(path_current + 'strict/')

    # Test
    test_mf_uni_out(path_current, 'relaxed', params)
    test_mf_uni_out(path_current, 'strict', params)

# EOF
