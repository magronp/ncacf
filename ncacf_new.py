#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from helpers.training import train_ncacf_new
from helpers.eval import evaluate_uni
from helpers.models import ModelMFuni


def train_main_ncacf(in_out_list, variant_list, inter_list, nl_list, params, data_dir='data/'):

    for in_out in in_out_list:
        params['data_dir'] = data_dir + in_out + '/'
        for variant in variant_list:
            print('Task: ' + in_out + ' -  Variant: ' + variant)
            path_pretrain = 'outputs/' + in_out + '/mf_uni/' + variant + '/'
            for inter in inter_list:
                for nl in nl_list:
                    params['n_layers_di'] = nl
                    params['out_dir'] = 'outputs/' + in_out + '/ncacf/' + variant + '/' + inter +  '/layers_di_' + str(nl) + '/'
                    create_folder(params['out_dir'])
                    train_ncacf_new(params, path_pretrain=path_pretrain, in_out=in_out, variant=variant, inter=inter)
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
                                  params['n_features_hidden'], variant, params['out_sigm'])
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
    #train_main_ncacf(['out', 'in'], ['relaxed', 'strict'], ['mult', 'conc'], [0, 1, 2, 3], params, data_dir)
    train_main_ncacf(['out'], ['relaxed'], ['conc'], [2], params, data_dir)

# EOF
