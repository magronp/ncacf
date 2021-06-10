#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder, get_optimal_val_model_relaxed, get_optimal_val_model_strict
import os
import time
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributesRatings
from helpers.utils import wpe_joint
from helpers.eval import evaluate_uni
from torch.nn import Module, ModuleList, Linear, Sequential, ReLU, Embedding, Sigmoid, Identity
from matplotlib import pyplot as plt


class ModelNCACF(Module):

    def __init__(self, n_users, n_songs, n_features_in, n_features_hidden, n_embeddings, n_layers_di=2, inter='mult', variant='relaxed'):
        super(ModelNCACF, self).__init__()

        self.n_users = n_users
        self.variant = variant

        # Embeddings
        self.user_emb = Embedding(n_users, n_embeddings)
        self.item_emb = Embedding(n_songs, n_embeddings)
        self.user_emb.weight.data.data.normal_(0, 0.01)
        self.item_emb.weight.data.data.normal_(0, 0.01)

        # Item content extractor
        self.fnn_in = Sequential(Linear(n_features_in, n_features_hidden, bias=True), ReLU())
        self.fnn_hi1 = Sequential(Linear(n_features_hidden, n_features_hidden, bias=True), ReLU())
        self.fnn_out = Sequential(Linear(n_features_hidden, n_embeddings, bias=True))

        # Deep interaction and output layers
        self.inter = inter
        self.n_layers_di = n_layers_di
        self.n_features_di_in = n_embeddings * 2 ** (self.inter == 'conc')

        if not(self.n_layers_di == -1):
            if self.n_layers_di == 0:
                self.di = ModuleList([Identity()])
            else:
                self.di = ModuleList([Sequential(
                Linear(self.n_features_di_in // (2 ** q), self.n_features_di_in // (2 ** (q + 1)), bias=True),
                ReLU()) for q in range(self.n_layers_di)])
            # Output layer
            self.out_layer = Linear(self.n_features_di_in // (2 ** self.n_layers_di), 1, bias=False)
            self.out_layer.weight.data.fill_(1)
            self.out_act = Sigmoid()

    def forward(self, u, x, i):
        # Get the user factor
        w = self.user_emb(u)

        # Apply the content feature extractor
        h_con = self.fnn_in(x)
        h_con = self.fnn_hi1(h_con)
        h_con = self.fnn_out(h_con)

        # If strict model or for evaluation: no item embedding
        if all(i == -1):
            h = h_con
        else:
            # Distinct between strict, relaxed or 'sum' model
            if self.variant == 'strict':
                h = h_con
            else:
                h = self.item_emb(i)

        # Interaction model
        if self.inter == 'conc':
            emb = torch.cat((w.unsqueeze(1).expand(*[-1, h.shape[0], -1]),
                             h.unsqueeze(0).expand(*[w.shape[0], -1, -1])), dim=-1)
            emb = emb.view(-1, self.n_features_di_in)
        else:
            emb = w.unsqueeze(1) * h
            emb = emb.view(-1, emb.shape[-1])

        # Deep interaction model
        if self.n_layers_di == -1:
            pred_rat = emb.sum(dim=-1)
            pred_rat = pred_rat.view(self.n_users, -1)
        else:
            for nl in range(self.n_layers_di):
                emb = self.di[nl](emb)
            pred_rat = self.out_act(self.out_layer(emb))
            pred_rat = pred_rat.view(self.n_users, -1)

        return pred_rat, w, h, h_con


def train_ncacf(params, path_pretrain=None, n_layers_di=2, in_out='out', variant='relaxed', inter='mult'):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    if in_out == 'out':
        n_songs_train = int(0.7 * n_songs_train)

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    if in_out == 'out':
        path_features = os.path.join(params['data_dir'], 'train_feats.num.csv')
    else:
        path_features = os.path.join(params['data_dir'], 'feats.num.csv')

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelNCACF(n_users, n_songs_train, params['n_features_in'], params['n_features_hidden'],
                          params['n_embeddings'], n_layers_di, variant, inter)
    #my_model = ModelNCF(n_users, n_songs_train, params['n_embeddings'], n_layers_di, inter)
    #my_model = ModelMLP(n_users, n_songs_train, params['n_embeddings'])
    if not(path_pretrain is None):
        my_model.load_state_dict(torch.load(path_pretrain + 'model.pt'), strict=False)
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
            counts_tot = torch.transpose(data[1], 1, 0).to(params['device'])
            it = data[2].to(params['device'])
            # Forward pass
            pred_rat, w, h, h_con = my_model(u_total, x, it)
            # Back-propagation
            loss = wpe_joint(counts_tot, pred_rat, w, h, h_con, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())
        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, in_out=in_out, split='val')
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


def train_main_ncacf(in_out_list, variant_list, params, range_lW, range_lH, range_inter, range_nl_di, data_dir='data/'):

    val_b = not(len(range_lW) == 1 and len(range_lW) == 1)

    for in_out in in_out_list:
        params['data_dir'] = data_dir + in_out + '/'
        for inter in range_inter:
            for nl_di in range_nl_di:
                path_current = 'outputs/' + in_out + '/ncacf/' + inter + '/' + str(nl_di) + '/'

                for variant in variant_list:
                    if nl_di == -1:
                        path_pretrain = None
                    else:
                        path_pretrain = 'outputs/' + in_out + '/ncacf/' + inter + '/' + str(-1) + '/' + variant + '/'
                    # Grid search on the hyperparameters
                    if val_b:
                        if variant == 'relaxed':
                            for lW in range_lW:
                                for lH in range_lH:
                                    print('Task: ' + in_out + ' -  Inter: ' + inter + ' - N_layers: ' + str(nl_di) + ' - Variant: ' + variant)
                                    print('lambda_W=' + str(lW) + ' - lambda_H=' + str(lH))
                                    params['lW'], params['lH'] = lW, lH
                                    params['out_dir'] = path_current + 'relaxed/lW_' + str(lW) + '/lH_' + str(lH) + '/'
                                    create_folder(params['out_dir'])
                                    train_ncacf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, in_out=in_out,
                                                variant=variant, inter=inter)
                            get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])
                        else:
                            for lW in range_lW:
                                print('Task: ' + in_out + ' -  Inter: ' + inter + ' - N_layers: ' + str(nl_di) + ' - Variant: ' + variant)
                                print('lambda_W=' + str(lW))
                                params['lW'], params['lH'] = lW, 0.
                                params['out_dir'] = path_current + 'strict/lW_' + str(lW) + '/'
                                create_folder(params['out_dir'])
                                train_ncacf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, in_out=in_out,
                                            variant=variant, inter=inter)
                            get_optimal_val_model_strict(path_current, range_lW, params['n_epochs'])
                    else:
                        print('Task: ' + in_out + ' -  Inter: ' + inter + ' - N_layers: ' + str(nl_di) + ' - Variant: ' + variant)
                        params['lW'], params['lH'] = range_lW[0], range_lH[0]
                        params['out_dir'] = path_current + variant + '/'
                        create_folder(params['out_dir'])
                        train_ncacf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, in_out=in_out,
                                    variant=variant, inter=inter)
                        np.savez(path_current + 'hyperparams.npz', lW=params['lW'], lH=params['lH'])
    return


def test_main_ncacf(in_out_list, variant_list, range_inter, range_nl_di, params, data_dir='data/'):

    test_results_ncacf = np.zeros((2, 2, 2, 7, 2))
    for i_io, in_out in enumerate(in_out_list):
        params['data_dir'] = data_dir + in_out + '/'
        # Number of users and songs for the test
        n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
        n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
        if in_out == 'out':
            n_songs_train = int(0.7 * n_songs_total)
        else:
            n_songs_train = n_songs_total

        for ii, inter in enumerate(range_inter):
            for inl, nl_di in enumerate(range_nl_di):
                for iv, variant in enumerate(variant_list):
                    # Load model
                    path_current = 'outputs/' + in_out + '/ncacf/' + inter + '/' + str(nl_di) + '/' + variant + '/'
                    my_model = ModelNCACF(n_users, n_songs_train, params['n_features_in'], params['n_features_hidden'],
                                  params['n_embeddings'], nl_di, variant, inter)
                    my_model.load_state_dict(torch.load(path_current + '/model.pt'))
                    my_model.to(params['device'])
                    # Evaluate the model on the test set
                    ncacf_ndcg = evaluate_uni(params, my_model, in_out=in_out, split='test') * 100
                    ncacf_time = np.load(path_current + '/training.npz')['time']
                    # Display and store the results
                    print('Task: ' + in_out + ' -  Inter: ' + inter + ' - N_layers: ' + str(nl_di) + ' - Variant: ' + variant)
                    print('NDCG: ' + str(ncacf_ndcg) + 'Time: ' + str(ncacf_time))
                    test_results_ncacf[i_io, ii, iv, inl, 0] = ncacf_ndcg
                    test_results_ncacf[i_io, ii, iv, inl, 1] = ncacf_time
    # Record the results
    np.savez('outputs/test_results_ncacf.npz', test_results_ncacf=test_results_ncacf)

    return


def plot_test_ndcg():

    test_ndcg = np.load('outputs/test_results_ncacf.npz')['test_results_ncacf'][:, :, :, :-1, 0]

    plt.figure(0)
    plt.subplot(2, 2, 1)
    plt.title('Warm-start')
    plt.plot(test_ndcg[0, 0, :, :].T)
    plt.ylabel('NDCG (%)')
    plt.legend(['Relaxed', 'Strict'])
    plt.subplot(2, 2, 2)
    plt.title('Cold-start')
    plt.plot(test_ndcg[1, 0, :, :].T)
    plt.subplot(2, 2, 3)
    plt.plot(test_ndcg[0, 1, :, :].T)
    plt.ylabel('NDCG (%)')
    plt.xlabel('Q')
    plt.subplot(2, 2, 4)
    plt.plot(test_ndcg[1, 1, :, :].T)
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
    params = {'batch_size': 128,
              'n_embeddings': 128,
              'n_features_hidden': 1024,
              'n_features_in': 168,
              'n_epochs': 100,
              'lr': 1e-4,
              'device': device
              }

    data_dir = 'data/'
    # Training and validation for the hyperparameters
    in_out_list, variant_list, range_lW, range_lH  = ['in', 'out'], ['relaxed', 'strict'], [0.1], [0.1]
    range_inter, range_nl_di = ['mult', 'conc'], [-1, 0, 1, 2, 3, 4, 5]
    #train_main_ncacf(in_out_list, variant_list, params, range_lW, range_lH, range_inter, range_nl_di, data_dir='data/')
    #test_main_ncacf(in_out_list, variant_list, range_inter, range_nl_di, params, data_dir='data/')

    plot_test_ndcg()
# EOF
