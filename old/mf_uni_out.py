#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
from helpers.utils import plot_val_ndcg_lW_lH, plot_val_ndcg_lW
from helpers.eval import evaluate_uni
from helpers.models import ModelMFuni


def train_mf_uni_out(params, variant='relaxed'):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelMFuni(n_users, n_songs_train, params['n_embeddings'], params['n_features_in'],
                          params['n_features_hidden'], variant, params['out_sigm'])
    my_model.requires_grad_(True)
    my_model.to(params['device'])

    # Training setup
    my_optimizer = Adam(params=my_model.parameters(), lr=params['lr'])
    torch.autograd.set_detect_anomaly(True)

    # Define the dataset
    my_dataset = DatasetAttributesRatings(features_path=path_features, tp_path=path_tp_train, n_users=n_users)
    my_dataloader = DataLoader(my_dataset, params['batch_size'], shuffle=True, drop_last=True)

    # Loop over epochs
    u_total = torch.arange(0, n_users, dtype=torch.long).to(params['device'])
    time_tot, loss_tot, val_ndcg_tot = 0, [], []
    time_opt, ndcg_opt = time_tot, 0
    my_model.train()
    for ep in range(params['n_epochs']):
        print('\nEpoch {e_:4d}/{e:4d}'.format(e_=ep + 1, e=params['n_epochs']), flush=True)
        start_time_ep = time.time()
        epoch_losses = []
        for data in tqdm(my_dataloader, desc='Training', unit=' Batches(s)'):
            my_optimizer.zero_grad()
            # Load the user and item indices and account for negative samples
            x = data[0].to(params['device'])
            count_i = data[1].to(params['device'])
            it = data[2].to(params['device'])
            # Forward pass
            pred_rat, w, h, h_con = my_model(u_total, x, it)
            # Back-propagation
            loss = wpe_joint(count_i, pred_rat, w, h, h_con, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, split='val')
        val_ndcg_tot.append(val_ndcg)
        print('\nLoss: {l:6.6f} | Time: {t:5.3f} | NDCG: {n:5.3f}'.format(l=loss_ep, t=time_ep, n=val_ndcg),
              flush=True)

        # Save the model if it performs the best
        if val_ndcg > ndcg_opt:
            ndcg_opt = val_ndcg
            time_opt = time_tot
            torch.save(my_model.state_dict(), os.path.join(params['out_dir'], 'model.pt'))

    # Record the training log
    np.savez(os.path.join(params['out_dir'], 'training.npz'), loss=loss_tot, time=time_opt, val_ndcg=val_ndcg_tot)

    return


def train_val_mf_uni_relaxed_out(path_current, params, range_lW, range_lH):
    # Training with grid search on the hyperparameters
    for lW in range_lW:
        for lH in range_lH:
            print(lW, lH)
            params['lW'], params['lH'] = lW, lH
            params['out_dir'] = path_current + 'relaxed/lW_' + str(lW) + '/lH_' + str(lH) + '/'
            create_folder(params['out_dir'])
            train_mf_uni_out(params, variant='relaxed')
    get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])

    return


def train_val_mf_uni_strict_out(path_current, params, range_lW):
    # Training with grid search on the hyperparameters
    for lW in range_lW:
        print(lW)
        params['lW'], params['lH'] = lW, 0.
        params['out_dir'] = path_current + 'strict/lW_' + str(lW) + '/'
        create_folder(params['out_dir'])
        train_mf_uni_out(params, variant='strict')
    get_optimal_val_model_strict(path_current, range_lW, params['n_epochs'])

    return


def train_noval_mf_uni_out(path_current, variant, params, lW=1., lH=1.):

    params['lW'], params['lH'] = lW, lH
    params['out_dir'] = path_current + variant + '/'
    create_folder(params['out_dir'])
    train_mf_uni_out(params, variant=variant)

    return


def test_mf_uni_out(path_current, variant, params):

    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = int(0.7 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))
    my_model = ModelMFuni(n_users, n_songs_train, params['n_embeddings'], params['n_features_in'],
                          params['n_features_hidden'], variant, params['out_sigm'])
    my_model.load_state_dict(torch.load(path_current + variant + '/model.pt'))
    my_model.to(params['device'])
    print('Variant: ' + variant)
    print('NDCG: ' + str(evaluate_uni(params, my_model, split='test')))
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
              'out_sigm': False,
              'data_dir': 'data/out/',
              'device': device
              }

    path_current = '../outputs/out/mf_uni/'
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