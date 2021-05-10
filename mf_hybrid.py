#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from helpers.training import train_mf_hybrid
import numpy as np
from matplotlib import pyplot as plt
import torch
from helpers.eval import evaluate_mf_hybrid


def train_main_mf_hybrid(in_out_list, variant_list, params, range_lW, range_lH, data_dir='data/'):

    # In this case, set N_gd at 1
    params['n_ep_it'] = 1
    # Check if this is a validation scenario: if more than 1 value is given for lW / lH
    val_b = not(len(range_lW) == 1 and len(range_lW) == 1)

    for in_out in in_out_list:
        # Define the dataset and output path depending on if it's in/out task
        path_current = 'outputs/' + in_out + '/mf_hybrid/'
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
                            train_mf_hybrid(params, variant=variant, in_out=in_out)
                    get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])
                else:
                    for lW in range_lW:
                        print('Task: ' + in_out + ' -  Variant: ' + variant)
                        print('lambda_W=' + str(lW))
                        params['lW'], params['lH'] = lW, 0.
                        params['out_dir'] = path_current + 'strict/lW_' + str(lW) + '/'
                        create_folder(params['out_dir'])
                        train_mf_hybrid(params, variant='strict', in_out=in_out)
                    get_optimal_val_model_strict(path_current, range_lW, params['n_epochs'])
            else:
                print('Task: ' + in_out + ' -  Variant: ' + variant)
                params['lW'], params['lH'] = range_lW[0], range_lH[0]
                params['out_dir'] = path_current + variant + '/'
                create_folder(params['out_dir'])
                train_mf_hybrid(params, variant=variant, in_out=in_out)
                np.savez(path_current + 'hyperparams.npz', lW=params['lW'], lH=params['lH'])

    return


def train_mh_hybrid_epiter(in_out_list, variant_list, params, data_dir='data/'):

    for in_out in in_out_list:
        path_current = 'outputs/' + in_out + '/mf_hybrid/'
        params['data_dir'] = data_dir + in_out + '/'
        for variant in variant_list:
            # Load the optimal hyper-parameters
            lamb_load = np.load(path_current + variant + '/hyperparams.npz')
            params['lW'], params['lH'] = lamb_load['lW'], lamb_load['lH']
            # Try other ep_it
            for n_ep_it in [2, 5, 10]:
                print('Task: ' + in_out + ' -  Variant: ' + variant)
                print('N_GD=' + str(n_ep_it))
                # Define the output directory
                params['out_dir'] = path_current + variant + '/gd_' + str(n_ep_it) + '/'
                create_folder(params['out_dir'])
                params['n_ep_it'] = n_ep_it
                train_mf_hybrid(params, variant=variant, in_out=in_out)

    return


def plot_val_mf_hybrid_epiter(in_out, variant, n_epochs):

    n_ep_it_list = [1, 2, 5, 10]
    # Load the validation NDCGs
    val_ndcg_epit = np.zeros((len(n_ep_it_list), n_epochs))
    for inep, n_ep_it in enumerate(n_ep_it_list):
        if n_ep_it == 1:
            path_ep = 'outputs/' + in_out + '/mf_hybrid/' + variant + '/'
        else:
            path_ep = 'outputs/' + in_out + '/mf_hybrid/' + variant + '/gd_' + str(n_ep_it) + '/'
        val_ndcg_epit[inep, :] = np.load(path_ep + 'training.npz')['val_ndcg'] * 100

    # Plot the validation NDCG
    plt.figure()
    plt.plot(np.arange(n_epochs) + 1, val_ndcg_epit.T)
    plt.xlabel('Epochs')
    plt.ylabel('NDCG (%)')
    plt.legend(['N_GD=1', 'N_GD=2', 'N_GD=5'])

    return


def test_main_mf_hybrid(in_out_list, variant_list, params, data_dir='data/'):
    n_ep_it_list = [1, 2, 5, 10]
    for in_out in in_out_list:
        # Define the dataset and output path depending on if it's in/out task
        path_current = 'outputs/' + in_out + '/mf_hybrid/'
        params['data_dir'] = data_dir + in_out + '/'
        # Loop over variants
        for variant in variant_list:
            for n_ep_it in n_ep_it_list:
                if n_ep_it == 1:
                    path_ep = path_current + variant + '/'
                else:
                    path_ep = path_current + variant + '/gd_' + str(n_ep_it) + '/'
                my_model = torch.load(path_ep + 'model.pt').to(params['device'])
                W, H = np.load(path_ep + 'wmf.npz')['W'], np.load(path_ep + 'wmf.npz')['H']
                print('Task: ' + in_out + ' -  Variant: ' + variant + ' - N_gd: ' + str(n_ep_it))
                print('NDCG: ' + str(evaluate_mf_hybrid(params, W, H, my_model, in_out=in_out, variant=variant, split='test')))
                print('Time: ' + str(np.load(path_ep + 'training.npz')['time']))

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
              'n_epochs': 150,
              'lr': 1e-4,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'device': device}
    data_dir = 'data/'

    # Training (and display the validation plots)
    range_lW, range_lH = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]
    train_main_mf_hybrid(['out', 'in'], ['relaxed', 'strict'], params, range_lW, range_lH, data_dir)
    # plot_val_ndcg_lW_lH('outputs/out/mf_hybrid/relaxed/')
    # plot_val_ndcg_lW('outputs/out/mf_hybrid/strict/')
    # plot_val_ndcg_lW_lH('outputs/in/mf_hybrid/relaxed/')
    # plot_val_ndcg_lW('outputs/in/mf_hybrid/strict/')

    # Check what happens if ep_it varies (and display the validation plots)
    train_mh_hybrid_epiter(['out', 'in'], ['relaxed', 'strict'], params, data_dir)
    #plot_val_mf_hybrid_epiter('out', 'relaxed', params['n_epochs'])
    #plot_val_mf_hybrid_epiter('out', 'strict', params['n_epochs'])
    #plot_val_mf_hybrid_epiter('in', 'relaxed', params['n_epochs'])
    #plot_val_mf_hybrid_epiter('in', 'strict', params['n_epochs'])

    # Display results on the test set
    test_main_mf_hybrid(['out', 'in'], ['relaxed', 'strict'], params, data_dir)


# EOF
