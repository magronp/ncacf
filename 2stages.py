#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers.utils import create_folder, plot_val_ndcg_lW_lH
from helpers.training import train_wmf, train_2stages
from helpers.eval import evaluate_mf_hybrid
import numpy as np
import torch

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def train_main_2stages(in_out_list, variant_list, params, range_lW, range_lH, data_dir='data/'):

    for in_out in in_out_list:

        # Define data/outputs paths
        path_current = 'outputs/' + in_out + '/2stages/'
        params['data_dir'] = data_dir + in_out + '/'

        # Loop over hyperparameters
        for lW in range_lW:
            for lH in range_lH:
                print('Task: ' + in_out)
                print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
                params['lW'], params['lH'] = lW, lH
                params['out_dir'] = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/'
                create_folder(params['out_dir'])
                # First train the WMF
                train_wmf(params, in_out=in_out)
                # Then train the relaxed and strict variant on top of these WMFs
                for variant in variant_list:
                    print('Variant: ' + variant)
                    # Useless to train for 'in' and 'relaxed' (it's juste WMF)
                    if not(variant == 'relaxed' and in_out == 'in'):
                        train_2stages(params, variant=variant, in_out=in_out)

        # Get the optimal models after grid search
        for variant in variant_list:
            if not (variant == 'relaxed' and in_out == 'in'):
                get_optimal_2stages(path_current, range_lW, range_lH, params['n_epochs'], variant=variant)

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


def test_main_2stages(in_out_list, variant_list, params, data_dir='data/'):

    for in_out in in_out_list:
        # Define the dataset and output path depending on if it's in/out task
        params['data_dir'] = data_dir + in_out + '/'
        # Loop over variants
        for variant in variant_list:
            if not (variant == 'relaxed' and in_out == 'in'):
                path_current = 'outputs/' + in_out + '/2stages/' + variant + '/'
                my_model = torch.load(path_current + 'model.pt').to(params['device'])
                W, H = np.load(path_current + 'wmf.npz')['W'], np.load(path_current + 'wmf.npz')['H']
                print('Task: ' + in_out + ' -  Variant: ' + variant)
                print('NDCG: ' + str(evaluate_mf_hybrid(params, W, H, my_model, in_out=in_out, variant=variant, split='test')))
                print('Time: ' + str(np.load(path_current + 'training.npz')['time']))

    return


def test_main_wmf(params):

    params['data_dir'] = 'data/in/'
    # Validation for selecting the best hyperparameters
    val_ndcg_opt, lW_opt, lH_opt = 0, 0, 0
    for lW in range_lW:
        for lH in range_lH:
            print('Validation...')
            print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
            path_wmf = 'outputs/in/2stages/lW_' + str(lW) + '/lH_' + str(lH) + '/wmf.npz'
            W, H = np.load(path_wmf)['W'], np.load(path_wmf)['H']
            val_ndcg = evaluate_mf_hybrid(params, W, H, None, in_out='in', variant='relaxed', split='val')
            if val_ndcg > val_ndcg_opt:
                val_ndcg_opt = val_ndcg
                lW_opt, lH_opt = lW, lH

    # Test with the best hyperparameters
    path_wmf_opt = 'outputs/in/2stages/lW_' + str(lW_opt) + '/lH_' + str(lH_opt) + '/wmf.npz'
    wmf_load = np.load(path_wmf_opt)
    W, H, time_wmf = wmf_load['W'], wmf_load['H'], wmf_load['time_wmf']
    test_ndcg = evaluate_mf_hybrid(params, W, H, None, in_out='in', variant='relaxed', split='val')
    print('WMF (lambda_W=' + str(lW_opt) + ' lambda_H=' + str(lH_opt) + ')')
    print('NDCG: ' + str(test_ndcg))
    print('Time: ' + str(time_wmf))

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
              'device': device}
    data_dir = 'data/'

    # Training (and display the validation plots)
    range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]
    train_main_2stages(['out', 'in'], ['relaxed', 'strict'], params, range_lW, range_lH, data_dir)
    # plot_val_ndcg_lW_lH('outputs/out/2stages/relaxed/')
    # plot_val_ndcg_lW_lH('outputs/out/2stages/strict/')
    # plot_val_ndcg_lW_lH('outputs/in/2stages/strict/')

    # Test dans display results
    test_main_2stages(['out', 'in'], ['relaxed', 'strict'], params, data_dir)

    # Test WMF (for in-matrix, equivalent to 2stages-relaxed)
    test_main_wmf(params)

# EOF
