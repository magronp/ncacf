#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.nn import Module, ModuleList, Linear, Sequential, ReLU, Embedding, Sigmoid, Identity, Dropout


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

    def __init__(self, n_users, n_songs, n_embeddings, n_features_in, n_features_hidden, mod, out_sigm=False):

        super(ModelMFuni, self).__init__()

        # Define if the model variant is strict, relaxed, or sum
        self.mod = mod

        if out_sigm:
            self.func_act_out = Sigmoid()
        else:
            self.func_act_out = Identity()

        # Item content extractor
        self.fnn_in = Sequential(Linear(n_features_in, n_features_hidden, bias=True), ReLU())
        self.fnn_hi1 = Sequential(Linear(n_features_hidden, n_features_hidden, bias=True), ReLU())
        self.fnn_out = Linear(n_features_hidden, n_embeddings, bias=True)

        # embedding layers and initialization (uniform)
        self.user_emb = Embedding(n_users, n_embeddings)
        self.user_emb.weight.data.normal_(0, 0.01)
        if self.mod == 'relaxed':
            self.item_emb = Embedding(n_songs, n_embeddings)
            self.item_emb.weight.data.normal_(0, 0.01)

    def forward(self, u, x, i):

        # Apply the content feature extractor
        h_con = self.fnn_in(x)
        h_con = self.fnn_hi1(h_con)
        h_con = self.fnn_out(h_con)

        # Get the factors
        w = self.user_emb(u)

        # If strict model or for evaluation: no item embedding
        if all(i == -1):
            h = h_con
        else:
            # Distinct between strict, relaxed or 'sum' model
            if self.mod == 'strict':
                h = h_con
            else:
                h = self.item_emb(i)

        # Interaction model
        pred_rat = self.func_act_out(torch.matmul(h, torch.transpose(w, 0, 1)))

        return pred_rat, w, h, h_con


class ModelNCACF(Module):

    def __init__(self, n_users, n_songs, n_features_in, n_features_hidden, n_embeddings, n_layers_di, variant='relaxed',
                 inter='mult', out_act='sigmoid'):

        super(ModelNCACF, self).__init__()

        # Define the model variant (strict or relaxed) and interaction (multiplication or concatenation)
        self.n_users = n_users
        self.variant = variant
        self.inter = inter
        self.n_layers_di = n_layers_di

        if out_act == 'relu':
            self.func_out = ReLU()
        elif out_act == 'sigmoid':
            self.func_out = Sigmoid()
        else:
            self.func_out = Identity()

        # Item content extractor
        self.fnn_in = Sequential(Linear(n_features_in, n_features_hidden, bias=True), ReLU())
        self.fnn_hi1 = Sequential(Linear(n_features_hidden, n_features_hidden, bias=True), ReLU())
        self.fnn_out = Sequential(Linear(n_features_hidden, n_embeddings, bias=True))

        # User embedding
        self.user_emb = Embedding(n_users, n_embeddings)
        self.user_emb.weight.data.data.normal_(0, 0.01)
        # Item embedding (for the relaxed and sum models)
        if self.variant == 'relaxed':
            self.item_emb = Embedding(n_songs, n_embeddings)
            self.item_emb.weight.data.data.normal_(0, 0.01)

        # Deep interaction layers
        self.n_features_di_in = n_embeddings * 2**(self.inter == 'conc')

        # First create the intermediate layers
        self.di = ModuleList([Sequential(Linear(2 ** (n_layers_di - q + 2), 2 ** (n_layers_di - q+1), bias=True),
                                         ReLU()) for q in range(self.n_layers_di-1)])
        
        # Now, add the input and output layers depending on the total amount of layers
        if n_layers_di == 0:
            self.di_in = Sequential(Linear(self.n_features_di_in, 1, bias=False), self.func_out)
            #self.di.insert(0, Sequential(Linear(self.n_features_di_in, 1, bias=False), self.func_out))
        else:
            self.di_in = Sequential(Linear(self.n_features_di_in, 2 ** (n_layers_di+2), bias=True), ReLU())
            self.di.insert(n_layers_di - 1, Sequential(Linear(8, 1, bias=False), self.func_out))
            #self.di.insert(0, Sequential(Linear(self.n_features_di_in, 2 ** (n_layers_di+2), bias=True), ReLU()))
        # Finally add the input layers to the list
        self.di.insert(0, self.di_in)

        # Initialize the last interaction model layer with weights = 1 ( ~ matrix factorization)
        if (variant == 'relaxed') and (inter == 'mult'):
            self.di[-1][0].weight.data.fill_(1)
        # Other interaction layers to initialize
        for nl in range(self.n_layers_di):
             self.di[nl][0].weight.data.data.normal_(0, 0.01)
             self.di[nl][0].bias.data.data.normal_(0, 0.01)

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
            # Distinct between strict, relaxed or 'sum' model
            if self.variant == 'strict':
                h = h_con
            else:
                h = self.item_emb(i)

        # Interaction model: first do the combination of the embeddings
        if self.inter == 'mult':
            pred_rat = w.unsqueeze(1) * h
        else:
            pred_rat = torch.cat((w.unsqueeze(1).expand(*[-1, h.shape[0], -1]),
                                  h.unsqueeze(0).expand(*[self.n_users, -1, -1])), dim=-1)

        # Reshape/flatten as (n_users * batch_size, n_embeddings)
        pred_rat = pred_rat.view(-1, self.n_features_di_in)

        # Deep interaction model:
        for nl in range(self.n_layers_di+1):
            pred_rat = self.di[nl](pred_rat)

        # Reshape as (n_users, batch_size)
        pred_rat = pred_rat.view(self.n_users, -1)

        return pred_rat, w, h, h_con


# EOF
