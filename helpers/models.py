#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import torch
from torch.nn import Module, ModuleList, Linear, Sequential, ReLU, Embedding, Sigmoid, Identity


class ModelAttributes(Module):

    def __init__(self, n_features_in, n_features_hidden, n_embeddings):

        super(ModelAttributes, self).__init__()

        self.fnn_in = Sequential(Linear(n_features_in, n_features_hidden, bias=True), ReLU())
        self.fnn_hi1 = Sequential(Linear(n_features_hidden, n_features_hidden, bias=True), ReLU())
        self.fnn_out = Sequential(Linear(n_features_hidden, n_embeddings, bias=True))

    def forward(self, x):

        # Apply NN
        y = self.fnn_in(x)
        y = self.fnn_hi1(y)
        outputs = self.fnn_out(y)

        return outputs


class ModelMFuni(Module):

    def __init__(self, n_users, n_songs, n_embeddings, n_features_in, n_features_hidden, variant):

        super(ModelMFuni, self).__init__()

        # Define if the model variant is strict or relaxed
        self.variant = variant

        # Item content extractor
        self.fnn_in = Sequential(Linear(n_features_in, n_features_hidden, bias=True), ReLU())
        self.fnn_hi1 = Sequential(Linear(n_features_hidden, n_features_hidden, bias=True), ReLU())
        self.fnn_out = Linear(n_features_hidden, n_embeddings, bias=True)

        # embedding layers and initialization (uniform)
        self.user_emb = Embedding(n_users, n_embeddings)
        self.user_emb.weight.data.normal_(0, 0.01)
        if self.variant == 'relaxed':
            self.item_emb = Embedding(n_songs, n_embeddings)
            self.item_emb.weight.data.normal_(0, 0.01)

    def forward(self, u, x, i):

        # Apply the content feature extractor
        h_con = self.fnn_in(x)
        h_con = self.fnn_hi1(h_con)
        h_con = self.fnn_out(h_con)

        # Get the factors
        w = self.user_emb(u)

        # If the variant is strict, or if it's for evaluation: no item embedding
        if all(i == -1):
            h = h_con
        else:
            # Distinct between strict, or relaxed or model
            if self.variant == 'strict':
                h = h_con
            else:
                h = self.item_emb(i)

        # Interaction model
        pred_rat = torch.matmul(h, torch.transpose(w, 0, 1))
        pred_rat = torch.transpose(pred_rat, 1, 0)

        return pred_rat, w, h, h_con


class ModelNCF(Module):

    def __init__(self, n_users, n_songs, n_embeddings, n_layers_di=2, inter='mult'):
        super(ModelNCF, self).__init__()

        self.n_users = n_users
        # Embeddings
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
            self.out_layer = Linear(self.n_features_di_in // (2 ** self.n_layers_di), 1, bias=False)
            self.out_layer.weight.data.fill_(1)
            self.out_act = Sigmoid()

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
            pred_rat = self.out_act(self.out_layer(emb))
            pred_rat = pred_rat.view(self.n_users, -1)

        return pred_rat, w, h


class ModelNCACF(Module):

    def __init__(self, n_users, n_songs, n_features_in, n_features_hidden, n_embeddings, n_layers_di=2,
                 variant='relaxed', inter='mult'):
        super(ModelNCACF, self).__init__()

        self.n_users = n_users
        self.variant = variant
        self.n_songs = n_songs
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
            # Distinct between strict, relaxed
            if self.variant == 'strict':
                h = h_con
                print(h - h_con)
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

# EOF
