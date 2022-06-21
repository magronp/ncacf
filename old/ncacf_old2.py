
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder
from old.training import train_ncacf
from helpers.eval import evaluate_uni, evaluate_random
from matplotlib import pyplot as plt


def train_ncacf_layers(params, inter='mult', variant='relaxed'):

    path_pretrain = 'outputs/out/pretraining_uni/' + variant + '/'
    for nl in [1, 2, 3, 4]:
        params['n_layers_di'] = nl
        params['out_dir'] = 'outputs/NCACF/' + variant + '/' + inter + '/layers_' + str(nl) + '/'
        create_folder(params['out_dir'])
        train_ncacf(params, path_pretrain=path_pretrain, variant=variant, inter=inter, out_act='sigmoid')
        my_model = torch.load('outputs/NCACF/' + variant + '/' + inter + '/layers_' + str(nl) + '/model.pt')
        print(evaluate_uni(params, my_model, split='test') * 100)

    return


def test_ncacf(params):

    test_ndcg = np.zeros((2, 2, 5))
    for iv, variant in enumerate(['relaxed', 'strict']):
        for ii, inter in enumerate(['mult', 'conc']):
            for nl in [0, 1, 2, 3, 4]:
                my_model = torch.load('outputs/out/NCACF/' + variant + '/' + inter + '/layers_' + str(nl) + '/model.pt')
                test_ndcg[iv, ii, nl] = evaluate_uni(params, my_model, split='test') * 100

    np.savez('../outputs/out/NCACF/test_ndcg.npz', test_ndcg=test_ndcg)

    return


def plot_test_ndcg():

    test_ndcg = np.load('outputs/NCACF/test_ndcg.npz')['test_ndcg']
    Qplot = np.arange(np.size(test_ndcg, 2))+1

    plt.figure(0)
    plt.subplot(2, 2, 1)
    plt.plot(Qplot, test_ndcg[0, 0, :])
    plt.title('Relaxed')
    plt.ylabel('NDCG (%)')
    plt.subplot(2, 2, 2)
    plt.title('Strict')
    plt.plot(Qplot, test_ndcg[1, 0, :])
    plt.subplot(2, 2, 3)
    plt.plot(Qplot, test_ndcg[0, 1, :])
    plt.ylabel('NDCG (%)')
    plt.xlabel('Q')
    plt.subplot(2, 2, 4)
    plt.plot(Qplot, test_ndcg[1, 1, :])
    plt.xlabel('Q')

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
              'data_dir': 'data/out/',
              'device': device
              }

    # Train all models
    train_ncacf_layers(params, inter='mult', variant='relaxed')
    #train_ncacf_layers(params, inter='mult', variant='strict')
    #train_ncacf_layers(params, inter='conc', variant='relaxed')
    #train_ncacf_layers(params, inter='conc', variant='strict')

    # Test
    test_ncacf(params)
    plot_test_ndcg()

    # Random recom for reference
    evaluate_random(params)

# EOF
