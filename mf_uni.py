#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from helpers.training import train_mf_uni
from helpers.eval import evaluate_uni
from helpers.models import ModelMFuni


def train_main_mf_uni(in_out_list, variant_list, params, range_lW, range_lH, data_dir='data/'):

    # Check if this is a validation scenario: if more than 1 value is given for lW / lH
    val_b = not(len(range_lW) == 1 and len(range_lW) == 1)

    for in_out in in_out_list:
        # Define the dataset and output path depending on if it's in/out task
        path_current = 'outputs/' + in_out + '/mf_uni/'
        params['data_dir'] = data_dir + in_out + '/'

        for variant in variant_list:
            if val_b:
                if variant == 'relaxed':
                    for lW in range_lW:
                        for lH in range_lH:
                            print('Task: ' + in_out + ' -  Variant: ' + variant)
                            print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
                            params['lW'], params['lH'] = lW, lH
                            params['out_dir'] = path_current + 'relaxed/lW_' + str(lW) + '/lH_' + str(lH) + '/'
                            create_folder(params['out_dir'])
                            train_mf_uni(params, variant=variant, in_out=in_out)
                    get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])
                else:
                    for lW in range_lW:
                        print('Task: ' + in_out + ' -  Variant: ' + variant)
                        print('lambda_W=' + str(lW))
                        params['lW'], params['lH'] = lW, 0.
                        params['out_dir'] = path_current + 'strict/lW_' + str(lW) + '/'
                        create_folder(params['out_dir'])
                        train_mf_uni(params, variant='strict', in_out=in_out)
                    get_optimal_val_model_strict(path_current, range_lW, params['n_epochs'])
            else:
                print('Task: ' + in_out + ' -  Variant: ' + variant)
                params['lW'], params['lH'] = range_lW[0], range_lH[0]
                params['out_dir'] = path_current + variant + '/'
                create_folder(params['out_dir'])
                train_mf_uni(params, variant=variant, in_out=in_out)
                np.savez(path_current + 'hyperparams.npz', lW=params['lW'], lH=params['lH'])

    return


def test_main_mf_uni(in_out_list, variant_list, params, data_dir='data/'):

    for in_out in in_out_list:
        # Define the dataset and output path depending on if it's in/out task
        path_current = 'outputs/' + in_out + '/mf_uni/'
        params['data_dir'] = data_dir + in_out + '/'
        # Number of users and songs for the test
        n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
        n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
        if in_out == 'out':
            n_songs_train = int(0.7 * n_songs_total)
        else:
            n_songs_train = n_songs_total
        # Loop over variants
        for variant in variant_list:
            my_model = ModelMFuni(n_users, n_songs_train, params['n_embeddings'], params['n_features_in'],
                                  params['n_features_hidden'], variant)
            my_model.load_state_dict(torch.load(path_current + variant + '/model.pt'))
            my_model.to(params['device'])
            print('Task: ' + in_out + ' -  Variant: ' + variant)
            print('NDCG: ' + str(evaluate_uni(params, my_model, in_out=in_out, split='test')))
            print('Time: ' + str(np.load(path_current + variant + '/training.npz')['time']))

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
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'n_epochs': 100,
              'lr': 1e-4,
              'device': device
              }
    data_dir = 'data/'

    # Training
    range_lW, range_lH = [0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10]
    train_main_mf_uni(['out', 'in'], ['relaxed', 'strict'], params, range_lW, range_lH, data_dir)

    # Plot the validation loss as a function of the hyperparameters
    # plot_val_ndcg_lW_lH('outputs/out/mf_uni/relaxed/')
    # plot_val_ndcg_lW('outputs/out/mf_uni/strict/')
    # plot_val_ndcg_lW_lH('outputs/in/mf_uni/relaxed/')
    # plot_val_ndcg_lW('outputs/in/mf_uni/strict/')

    # Testing
    test_main_mf_uni(['out', 'in'], ['relaxed', 'strict'], params, data_dir)

# EOF
