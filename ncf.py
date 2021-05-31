#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from helpers.utils import create_folder, get_optimal_val_model_relaxed
from helpers.training import train_ncf_in
import os
import time
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from helpers.data_feeder import load_tp_data, DatasetAttributes, DatasetAttributesRatings, DatasetAttributesNegsamp
from helpers.utils import compute_factor_wmf_deep, wpe_hybrid_strict, wpe_joint, wpe_joint_ncf, wpe_joint_ncacfnew, wpe_joint_neg
from helpers.eval import evaluate_mf_hybrid, predict_attributes, evaluate_uni
from torch.nn import Module, ModuleList, Linear, Sequential, ReLU, Embedding, Sigmoid, Identity


class ModelNCF(Module):

    def __init__(self, n_users, n_songs, n_embeddings, n_layers_di=2, inter='mult'):
        super(ModelNCF, self).__init__()

        self.n_users = n_users
        # Same for the MLP part
        self.user_emb = Embedding(n_users, n_embeddings)
        self.item_emb = Embedding(n_songs, n_embeddings)
        self.user_emb.weight.data.data.normal_(0, 0.01)
        self.item_emb.weight.data.data.normal_(0, 0.01)

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
            self.out_layer = Linear(n_embeddings // (2 ** self.n_layers_di), 1, bias=False)
            self.out_layer.weight.data.fill_(1)

    def forward(self, u, x, i):
        # Get the user/item factors
        w = self.user_emb(u)
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
            pred_rat = self.out_layer(emb)
            pred_rat = pred_rat.view(self.n_users, -1)

        return pred_rat, w, h


def train_ncf(params, path_pretrain=None, n_layers_di=2, inter='mult'):

    # Get the hyperparameters
    lW, lH = params['lW'], params['lH']

    # Get the number of songs and users
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_train = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

    # Path for the TP training data, features and the WMF
    path_tp_train = params['data_dir'] + 'train_tp.num.csv'
    path_features = params['data_dir'] + 'feats.num.csv'

    # Get the playcount data, confidence, and precompute its transpose
    train_data, _, _, conf = load_tp_data(path_tp_train, shape=(n_users, n_songs_train))

    # Define and initialize the model, and get the hyperparameters
    my_model = ModelNCF(n_users, n_songs_train, params['n_embeddings'], n_layers_di, inter)
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
            # Load the user / item indices and (positive) counts
            counts_tot = torch.transpose(data[1], 1, 0).to(params['device'])
            it_batch = data[2].to(params['device'])
            # Forward pass
            pred_rat, w, h = my_model(u_total, None, it_batch)
            # Back-propagation
            loss = wpe_joint_ncf(counts_tot, pred_rat, w, h, lW, lH)
            loss.backward()
            clip_grad_norm_(my_model.parameters(), max_norm=1.)
            my_optimizer.step()
            epoch_losses.append(loss.item())
        # Overall stats for one epoch
        loss_ep = np.mean(epoch_losses)
        loss_tot.append(loss_ep)
        time_ep = time.time() - start_time_ep
        time_tot += time_ep
        val_ndcg = evaluate_uni(params, my_model, in_out='in', split='val')
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


def train_main_ncf(params, range_lW, range_lH, range_inter, range_nl_di, data_dir='data/'):

    val_b = not(len(range_lW) == 1 and len(range_lW) == 1)
    params['data_dir'] = data_dir + 'in/'
    n_ep_max = params['n_epochs']

    for inter in range_inter:
        for nl_di in range_nl_di:
            path_current = 'outputs/in/ncf/' + inter + '/' + str(nl_di) + '/'
            if nl_di == -1:
                path_pretrain = None
                params['n_epochs'] = n_ep_max
            else:
                path_pretrain = 'outputs/in/ncf/' + inter + '/' + str(-1) + '/'
                params['n_epochs'] = 30

            # Training with grid search on the hyperparameters
            if val_b:
                for lW in range_lW:
                    for lH in range_lH:
                        print(lW, lH)
                        params['lW'], params['lH'] = lW, lH
                        params['out_dir'] = path_current + 'lW_' + str(lW) + '/lH_' + str(lH) + '/'
                        create_folder(params['out_dir'])
                        train_ncf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, inter=inter)
                get_optimal_val_model_relaxed(path_current, range_lW, range_lH, params['n_epochs'])
            else:
                params['lW'], params['lH'] = range_lW[0], range_lH[0]
                params['out_dir'] = path_current
                create_folder(params['out_dir'])
                train_ncf(params, path_pretrain=path_pretrain, n_layers_di=nl_di, inter=inter)
    return


def test_main_ncf(range_inter, range_nl_di, params, data_dir='data/'):

    params['data_dir'] = data_dir + 'in/'

    for inter in range_inter:
        for nl_di in range_nl_di:
            path_current = 'outputs/in/ncf/' + inter + '/' + str(nl_di) + '/'
            # Number of users and songs for the test
            n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
            n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
            n_songs_train = n_songs_total
            my_model = ModelNCF(n_users, n_songs_train, params['n_embeddings'], nl_di, inter)
            my_model.load_state_dict(torch.load(path_current + '/model.pt'))
            my_model.to(params['device'])
            print('Inter: ' + inter + ' -  Layers DI: ' + str(nl_di))
            print('NDCG: ' + str(evaluate_uni(params, my_model, in_out='in', split='test')))
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
              'n_epochs': 100,
              'lr': 1e-4,
              'device': device
              }

    data_dir = 'data/'
    # Training and validation for the hyperparameters
    #range_lW, range_lH = [0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10]
    #range_lW, range_lH = [0.1], [0.1]
    #train_main_ncf(params, range_lW, range_lH, data_dir)

    range_lW, range_lH, range_inter, range_nl_di = [0.1], [0.1], ['mult'], [-1, 0, 1, 2]
    train_main_ncf(params, range_lW, range_lH, range_inter, range_nl_di, data_dir='data/')
    test_main_ncf(range_inter, range_nl_di, params, data_dir='data/')

# EOF
