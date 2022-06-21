#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder
from old.training import train_ncacf
from helpers.eval import evaluate_uni
from matplotlib import pyplot as plt


def train_all_ncacf():

    for variant in ['relaxed', 'strict']:
        for inter in ['mult']:
            if inter == 'conc':
                params['batch_size'] = 4
            for nl in [0, 1, 2, 3, 4]:
                params['n_layers_di'] = nl
                path_pretrain = 'outputs/pretraining_uni/' + variant + '/'
                params['out_dir'] = 'outputs/NCACF/' + variant + '/' + inter + '/' + 'layers_' + str(nl) + '/'
                create_folder(params['out_dir'])
                train_ncacf(params, path_pretrain=path_pretrain, variant=variant, inter=inter)

    return

def test_main_ncacf(setting_list, variant_list, range_inter, range_nl_di, params, data_dir='data/'):

    test_results_ncacf = np.zeros((2, 2, 2, 7, 2))
    for i_io, setting in enumerate(setting_list):
        params['data_dir'] = data_dir + setting + '/'
        # Number of users and songs for the test
        n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
        n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
        if setting == 'cold':
            n_songs_train = int(0.8 * 0.9 * n_songs_total)
        else:
            n_songs_train = n_songs_total

        for ii, inter in enumerate(range_inter):
            for inl, nl_di in enumerate(range_nl_di):
                for iv, variant in enumerate(variant_list):
                    # Load model
                    path_current = 'outputs/' + setting + '/ncacf/' + inter + '/' + str(nl_di) + '/' + variant + '/'
                    my_model = ModelNCACF(n_users, n_songs_train, params['n_features_in'], params['n_features_hidden'],
                                  params['n_embeddings'], nl_di, variant, inter)
                    my_model.load_state_dict(torch.load(path_current + '/model.pt'))
                    my_model.to(params['device'])
                    # Evaluate the model on the test set
                    ncacf_ndcg = evaluate_uni(params, my_model, setting=setting, split='test') * 100
                    ncacf_time = np.load(path_current + '/training.npz')['time']
                    # Display and store the results
                    print('Task: ' + setting + ' -  Inter: ' + inter + ' - N_layers: ' + str(nl_di) + ' - Variant: ' + variant)
                    print('NDCG: ' + str(ncacf_ndcg) + 'Time: ' + str(ncacf_time))
                    test_results_ncacf[i_io, ii, iv, inl, 0] = ncacf_ndcg
                    test_results_ncacf[i_io, ii, iv, inl, 1] = ncacf_time
    # Record the results
    np.savez('outputs/test_results_ncacf.npz', test_results_ncacf=test_results_ncacf)

    return

def subtrain_all_ncacf():

    for variant in ['strict']:
        for inter in ['mult']:
            for nl in [1]:
                params['n_layers_di'] = nl
                path_pretrain = 'outputs/pretraining_uni/' + variant + '/'
                #params['lW'], params['lH'] = 0.1, 0
                params['out_dir'] = 'outputs/NCACF/' + variant + '/' + inter + '/' + 'layers_' + str(nl) + '/'
                create_folder(params['out_dir'])
                train_ncacf(params, path_pretrain=path_pretrain, variant=variant, inter=inter, out_act='identity')

    return


def subtrain_multrel():
    variant = 'relaxed'
    path_pretrain = 'outputs/pretraining_uni/' + variant + '/'
    inter = 'mult'
    for out_act in ['identity', 'sigmoid', 'relu']:
        for nl in [0, 1, 2, 3, 4]:
            params['n_layers_di'] = nl
            params['out_dir'] = 'outputs/NCACF/' + variant + '/' + inter + '/' + out_act + '/' + 'layers_' + str(nl) + '/'
            create_folder(params['out_dir'])
            train_ncacf(params, path_pretrain=path_pretrain, variant=variant, inter=inter, out_act=out_act)

    return


def test_all_ncacf(params):

    test_ndcg = np.zeros((2, 2, 5))
    for iv, variant in enumerate(['relaxed', 'strict']):
        for ii, inter in enumerate(['mult', 'conc']):
            for nl in [0, 1, 2, 3, 4]:
                my_model = torch.load('outputs/NCACF/' + variant + '/' + inter + '/layers_' + str(nl) + '/model.pt')
                test_ndcg[iv, ii, nl] = evaluate_uni(params, my_model, split='test') * 100

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(test_ndcg[0, 0, :])
    plt.title('Relaxed')
    plt.ylabel('NDCG (%)')
    plt.subplot(2, 2, 2)
    plt.title('Strict')
    plt.plot(test_ndcg[1, 0, :])
    plt.subplot(2, 2, 3)
    plt.ylabel('NDCG (%)')
    plt.xlabel('Q')
    plt.plot(test_ndcg[0, 1, :])
    plt.subplot(2, 2, 4)
    plt.plot(test_ndcg[1, 1, :])
    plt.xlabel('Q')

    return


def test_ncacf(params):

    test_ndcg = np.zeros((2, 3, 5))
    inter = 'mult'
    list_out_act = ['identity', 'sigmoid', 'relu']
    for iv, variant in enumerate(['relaxed', 'strict']):
        for io, out_act in enumerate(list_out_act):
            for nl in [0, 1, 2, 3, 4]:
                my_model = torch.load('outputs/NCACF/' + variant + '/' + inter + '/' + out_act + '/' + '/layers_' + str(nl) + '/model.pt')
                test_ndcg[iv, io, nl] = evaluate_uni(params, my_model, split='test') * 100

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(test_ndcg[0,:,:].T)
    plt.xlabel('Q')
    plt.subplot(1,2,2)
    plt.plot(test_ndcg[1,:,:].T)
    plt.legend(list_out_act)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)

    # Run on GPU (if it's available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Process on: {}'.format(torch.cuda.get_device_name(device)))

    # Set parameters
    params = {'batch_size': 8,
              'n_embeddings': 128,
              'n_epochs': 30,
              'lr': 1e-4,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'data_dir': 'data/',
              'device': 'cuda',
              }

    #subtrain_all_ncacf()
    subtrain_multrel()
    #test_all_ncacf(params)
    #test_ncacf(params)
# EOF
