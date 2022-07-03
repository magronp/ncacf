#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

from helpers.plotters import plot_val_ndcg_ncacf, plot_val_ndcg_lW_lH, plot_val_ndcg_lW
import numpy as np


# MF Hybrid validation results - impact of lambda
plot_val_ndcg_lW_lH('outputs/cold/mf_hybrid/relaxed/')
plot_val_ndcg_lW('outputs/cold/mf_hybrid/strict/')
plot_val_ndcg_lW_lH('outputs/warm/mf_hybrid/relaxed/')
plot_val_ndcg_lW('outputs/warm/mf_hybrid/strict/')

# MF Hybrid, influence of N_GD (and compare with the 2 stage approach)
ndcg_mf_gd = np.zeros((2, 2, 4))
ndcg_dcb = np.zeros((2, 2))
for ise, setting in enumerate(['warm', 'cold']):
    for iv, variant in enumerate(['relaxed', 'strict']):
        if not(setting == 'warm' and variant == 'relaxed'):
            ndcg_dcb[ise, iv] = round(np.load('outputs/' + setting + '/2stages/' + variant + '/training.npz')['val_ndcg'].max()*100, 1)
        ndcg_mf_gd[ise, iv, 0] = round(np.load('outputs/' + setting + '/mf_hybrid/' + variant + '/training.npz')['val_ndcg'].max()*100, 1)
        for ingd, ngd in enumerate([2, 5, 10]):
            ndcg_mf_gd[ise, iv, ingd+1] = round(np.load('outputs/' + setting + '/mf_hybrid/' + variant + '/gd_' + str(ngd) +
                                             '/training.npz')['val_ndcg'].max()*100, 1)
print('----Warm----')
print(' Relaxed: ', ndcg_mf_gd[0, 0, :])
print(' Strict: ', ndcg_dcb[0, 1], ndcg_mf_gd[0, 1, :])
print('----Cold----')
print(' Relaxed: ', ndcg_dcb[1, 0], ndcg_mf_gd[1, 0, :])
print(' Strict: ', ndcg_dcb[1, 1], ndcg_mf_gd[1, 1, :])

# MF Uni validation results - impact of lambda
plot_val_ndcg_lW_lH('outputs/cold/mf_uni/relaxed/')
plot_val_ndcg_lW('outputs/cold/mf_uni/strict/')
plot_val_ndcg_lW_lH('outputs/warm/mf_uni/relaxed/')
plot_val_ndcg_lW('outputs/warm/mf_uni/strict/')

# MF Hybrid and Uni : display best perf as well as computation time
perf_mf_hybrid = np.zeros((2, 2, 2))
perf_mf_uni = np.zeros((2, 2, 2))
for ise, setting in enumerate(['warm', 'cold']):
    for iv, variant in enumerate(['relaxed', 'strict']):
        loadmf = np.load('outputs/' + setting + '/mf_hybrid/' + variant + '/training.npz')
        perf_mf_hybrid[ise, iv, :] = round(loadmf['val_ndcg'].max()*100, 1), round(loadmf['time'].item())
        loadmf = np.load('outputs/' + setting + '/mf_uni/' + variant + '/training.npz')
        perf_mf_uni[ise, iv, :] = round(loadmf['val_ndcg'].max()*100, 1), round(loadmf['time'].item())

print('----Warm----')
print(' Relaxed - MF Hybrid - NDCG: ', perf_mf_hybrid[0, 0, 0], '%', ' Time: ', perf_mf_hybrid[0, 0, 1], 's')
print(' Relaxed - MF Uni    - NDCG: ', perf_mf_uni[0, 0, 0], '%', ' Time: ', perf_mf_uni[0, 0, 1], 's')
print(' Strict  - MF Hybrid - NDCG: ', perf_mf_hybrid[0, 1, 0], '%', ' Time: ', perf_mf_hybrid[0, 1, 1], 's')
print(' Strict  - MF Uni    - NDCG: ', perf_mf_uni[0, 1, 0], '%', ' Time: ', perf_mf_uni[0, 1, 1], 's')
print('----Cold----')
print(' Relaxed - MF Hybrid - NDCG: ', perf_mf_hybrid[1, 0, 0], '%', ' Time: ', perf_mf_hybrid[1, 0, 1], 's')
print(' Relaxed - MF Uni    - NDCG: ', perf_mf_uni[1, 0, 0], '%', ' Time: ', perf_mf_uni[1, 0, 1], 's')
print(' Strict  - MF Hybrid - NDCG: ', perf_mf_hybrid[1, 1, 0], '%', ' Time: ', perf_mf_hybrid[1, 1, 1], 's')
print(' Strict  - MF Uni    - NDCG: ', perf_mf_uni[1, 1, 0], '%', ' Time: ', perf_mf_uni[1, 1, 1], 's')

# NCACF validation results (impact of variant, interaction model, and number of layers)
plot_val_ndcg_ncacf()


# Test results


# EOF
