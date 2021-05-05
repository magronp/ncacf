#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import pickle
from helpers.model_attributes import evaluate_mf
from helpers.model_joint import evaluate_joint
from helpers.functions import evaluate_random


# Run on GPU (if it's available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Process on: {}'.format(torch.cuda.get_device_name(device)))

# Set parameters
params = {'batch_size_wmf': 10000,
          'batch_size_deep': 1,
          'n_embeddings': 50,
          'n_epochs': 50,
          'n_iter_wmf': 20,
          'lr': 1e-4,
          'n_features_hidden': 1024,
          'n_features_in': 168,
          'data_dir': 'data/',
          'out_dir': '',
          'device': device
          }

# Lower bound: random predictions
score = {}
score['random'] = evaluate_random(params)

# Baseline - relaxed
my_model = torch.load('outputs/baseline_relaxed/model.pt')
W = np.load('outputs/baseline_relaxed/wmf.npz')['W']
score['baseline_relaxed'] = evaluate_mf(params, W, my_model, split='test')

# Baseline - strict
my_model = torch.load('outputs/baseline_strict/model.pt')
W = np.load('outputs/baseline_strict/wmf.npz')['W']
score['baseline_strict'] = evaluate_mf(params, W, my_model, split='test')

# Alternate updates - relaxed
my_model = torch.load('outputs/alt_MF_relaxed/model_1.pt')
W = np.load('outputs/alt_MF_relaxed/wmf_1.npz')['W']
score['alt_relaxed'] = evaluate_mf(params, W, my_model, split='test')

# Alternate updates - strict
my_model = torch.load('outputs/alt_MF_strict/model_1.pt')
W = np.load('outputs/alt_MF_strict/wmf_1.npz')['W']
score['alt_strict'] = evaluate_mf(params, W, my_model, split='test')

# Joint model - MF - relaxed
my_model = torch.load('outputs/joint_MF_relaxed/model.pt')
score['joint_MF_relaxed'] = evaluate_joint(params, my_model, split='test', mod='relaxed')

# Joint model - MF - strict
my_model = torch.load('outputs/joint_MF_strict/model.pt')
score['joint_MF_relaxed'] = evaluate_joint(params, my_model, split='test', mod='strict')

# Record
with open(os.path.join('outputs/score.npz'), 'wb') as f1:
    pickle.dump(score, f1)
print(score)

