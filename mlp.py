#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder
import os
import time
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributes, DatasetAttributesRatings
from helpers.utils import wpe_joint
from helpers.models import ModelMFuni
from helpers.eval import evaluate_uni
import torch
from torch.nn import Module, ModuleList, Linear, Sequential, ReLU, Embedding, Sigmoid, Identity


class ModelMLP(Module):

    def __init__(self, n_users, n_songs, n_features_in, n_features_hidden, n_embeddings, n_layers_di, variant='relaxed',
                 inter='mult'):

        super(ModelMLP, self).__init__()

        # Define the model variant (strict or relaxed) and interaction (multiplication or concatenation)
        self.n_users = n_users
        self.variant = variant
        self.inter = inter
        self.n_layers_di = n_layers_di

        # Item content extractor
        self.fnn_in = Sequential(Linear(n_features_in, n_features_hidden, bias=True), ReLU())
        self.fnn_hi1 = Sequential(Linear(n_features_hidden, n_features_hidden, bias=True), ReLU())
        self.fnn_out = Sequential(Linear(n_features_hidden, n_embeddings, bias=True))

        # User (and item for the relaxed variant) embedding, corresponding to the factorization part
        self.user_emb_mlp = Embedding(n_users, n_embeddings)
        self.user_emb_mlp.weight.data.data.normal_(0, 0.01)
        # Item embedding (for the relaxed models)
        if self.variant == 'relaxed':
            self.user_emb_mlp = Embedding(n_songs, n_embeddings)
            self.user_emb_mlp.weight.data.data.normal_(0, 0.01)

        # Deep interaction layers
        self.n_features_di_in = n_embeddings * 2 ** (self.inter == 'conc')
        if n_layers_di == 0:
            self.di = ModuleList([Identity()])
        else:
            self.di = ModuleList([Sequential(
                Linear(self.n_features_di_in // (2 ** q), self.n_features_di_in // (2 ** (q + 1)), bias=True),
                ReLU()) for q in range(self.n_layers_di)])

        # Output layers
        self.out_layer_mlp = Linear(self.n_features_di_in // (2 ** self.n_layers_di), 1, bias=False)
        self.out_act = Sigmoid()

    def forward(self, u, x, i):

        # Get the user factors
        w = self.user_emb(u)

        # Apply the content feature extractor
        h_con = self.fnn_in(x)
        h_con = self.fnn_hi1(h_con)
        h_con = self.fnn_out(h_con)

        # If strict model or for evaluation: no item embedding
        if all(i == -1):
            h = h_con
        else:
            # Distinct between strict and relaxed
            if self.variant == 'strict':
                h = h_con
            else:
                h = self.item_emb(i)

        # Interaction model: first do the combination of the embeddings
        if self.inter == 'mult':
            emb_mlp = w.unsqueeze(1) * h
        else:
            emb_mlp = torch.cat((w.unsqueeze(1).expand(*[-1, h.shape[0], -1]),
                                 h.unsqueeze(0).expand(*[self.n_users, -1, -1])), dim=-1)
        # Reshape/flatten as (n_users * batch_size, n_embeddings)
        emb_mlp = emb_mlp.view(-1, self.n_features_di_in)

        # Deep interaction model:
        for nl in range(self.n_layers_di):
            emb_mlp = self.di[nl](emb_mlp)

        # Concatenate embeddings and feed to the output layer
        pred_rat = self.out_act(self.out_layer_mlp(emb_mlp))
        # Reshape as (n_users, batch_size)
        pred_rat = pred_rat.view(self.n_users, -1)

        return pred_rat, w, h, h_con


def train_mlp(params, path_pretrain=None, in_out='out', variant='relaxed', inter='mult'):

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

    # Define and initialize the model
    my_model = ModelMLP(n_users, n_songs_train, params['n_features_in'], params['n_features_hidden'],
                          params['n_embeddings'], params['n_layers_di'], variant, inter)
    if path_pretrain is None:
        lW, lH = params['lW'], params['lH']
    else:
        my_model.load_state_dict(torch.load(path_pretrain + 'model.pt'), strict=False)
        lamb_load = np.load(os.path.join(path_pretrain, 'hyperparams.npz'))
        lW, lH = float(lamb_load['lW']), float(lamb_load['lH'])
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
            #loss = wpe_joint_ncacfnew(count_i, torch.transpose(pred_rat, 1, 0), w_gmf, h_gmf, w_mlp, h_mlp, h_con, lW, lH)
            loss = wpe_joint(count_i, torch.transpose(pred_rat, 1, 0), w, h, h_con, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())

        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, in_out, split='val')
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


def train_main_mlp(in_out_list, variant_list, inter_list, nl_list, params, data_dir='data/'):

    for in_out in in_out_list:
        params['data_dir'] = data_dir + in_out + '/'
        for variant in variant_list:
            print('Task: ' + in_out + ' -  Variant: ' + variant)
            path_pretrain = 'outputs/' + in_out + '/gmf/' + variant + '/'
            for inter in inter_list:
                for nl in nl_list:
                    params['n_layers_di'] = nl
                    params['out_dir'] = 'outputs/' + in_out + '/mlp/' + variant + '/' + inter + '/layers_di_' + str(nl) + '/'
                    create_folder(params['out_dir'])
                    train_mlp(params, path_pretrain=path_pretrain, in_out=in_out, variant=variant, inter=inter)
    return


def test_main_mlp(in_out_list, variant_list, inter_list, nl_list, params, data_dir='data/'):

    for in_out in in_out_list:
        # Define the dataset and output path depending on if it's in/out task
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
            for inter in inter_list:
                for nl in nl_list:
                    params['n_layers_di'] = nl
                    my_model = ModelMLP(n_users, n_songs_train, params['n_features_in'],
                                             params['n_features_hidden'],
                                             params['n_embeddings'], params['n_layers_di'], variant, inter)
                    path_current = 'outputs/' + in_out + '/mlp/' + variant + '/' + inter + '/layers_di_' + str(
                        nl) + '/'
                    my_model.load_state_dict(torch.load(path_current + '/model.pt'))
                    my_model.to(params['device'])
                    print('Task: ' + in_out + ' -  Variant: ' + variant)
                    print('NDCG: ' + str(evaluate_uni(params, my_model, in_out=in_out, split='test')))
                    print('Time: ' + str(np.load(path_current + '/training.npz')['time']))

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
              'device': device
              }
    data_dir = 'data/'

    # Training
    #in_out_list, variant_list, inter_list, nl_list = ['out', 'in'], ['relaxed', 'strict'], ['mult', 'conc'], [0, 1, 2, 3]
    in_out_list, variant_list, inter_list, nl_list = ['out'], ['relaxed'], ['mult'], [1]
    train_main_mlp(in_out_list, variant_list, inter_list, nl_list, params, data_dir)

    # Testing
    test_main_mlp(in_out_list, variant_list, inter_list, nl_list, params, data_dir)

# EOF
