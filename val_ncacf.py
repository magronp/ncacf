#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder
from helpers.training import train_ncacf
from helpers.eval import evaluate_uni
from matplotlib import pyplot as plt


def train_ncacf_mult(params):

    inter = 'mult'
    for variant in ['relaxed', 'strict']:
        for out_act in ['identity', 'sigmoid', 'relu']:
            for nl in [0, 1, 2, 3]:
                params['n_layers_di'] = nl
                params['out_dir'] = 'outputs/NCACF/' + variant + '/' + inter + '/' + out_act + '/layers_' + str(nl) + '/'
                create_folder(params['out_dir'])
                train_ncacf(params, path_pretrain='outputs/pretraining_uni/' + variant + '/', variant=variant, inter=inter, out_act=out_act)

    return


def subtest(params):

    inter = 'mult'
    for variant in ['relaxed']:
        for out_act in ['sigmoid']:
            for nl in [2]:
                params['n_layers_di'] = nl
                params['out_dir'] = 'outputs/NCACF/' + variant + '/' + inter + '/' + out_act + '/layers_' + str(nl) + '/'
                create_folder(params['out_dir'])
                train_ncacf(params, path_pretrain='outputs/pretraining_uni/' + variant + '/', variant=variant, inter=inter, out_act=out_act)

    my_model = torch.load('outputs/NCACF/' + variant + '/mult/' + out_act + '/layers_' + str(nl) + '/model.pt')
    print(evaluate_uni(params, my_model, split='test') * 100)

    return


def train_ncacf_conc(params):

    inter = 'conc'
    for variant in ['relaxed', 'strict']:
        for out_act in ['identity', 'sigmoid', 'relu']:
            for nl in [0, 1, 2, 3]:
                params['n_layers_di'] = nl
                params['out_dir'] = 'outputs/NCACF/' + variant + '/' + inter + '/' + out_act + '/layers_' + str(nl) + '/'
                create_folder(params['out_dir'])
                train_ncacf(params, path_pretrain=None, variant=variant, inter=inter, out_act=out_act)

    return


def test_ncacf_mult(params):

    test_ndcg = np.zeros((2, 3, 4))
    for iv, variant in enumerate(['relaxed', 'strict']):
        for io, out_act in enumerate(['identity', 'sigmoid', 'relu']):
            for nl in [0, 1, 2, 3]:
                my_model = torch.load('outputs/NCACF/' + variant + '/mult/' + out_act + '/layers_' + str(nl) + '/model.pt')
                test_ndcg[iv, io, nl] = evaluate_uni(params, my_model, split='test') * 100

    print(test_ndcg)
    '''
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
    '''

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
              'n_epochs': 30,
              'lr': 1e-4,
              'data_dir': 'data/',
              'device': device
              }

    #train_ncacf_mult(params)
    #test_ncacf_mult(params)
    subtest(params)

# EOF
