#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
#from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot_val_ndcg_mf_hybrid_relaxed(setting):

    path_ndcg = 'outputs/' + setting + '/mf_hybrid/relaxed/val_ndcg.npz'

    # Load the overall validation NDCG
    ndcg_loader = np.load(path_ndcg)
    val_ndcg, range_lW, range_lH = ndcg_loader['val_ndcg'], ndcg_loader['range_lW'], ndcg_loader['range_lH']
    Nw = len(range_lW)
    n_epochs = val_ndcg.shape[-1]

    # Plot the results
    plt.figure(figsize=(8, 5.4))
    for il, l in enumerate(range_lW):
        plt.subplot(2, Nw // 2, il + 1)
        plt.plot(np.arange(n_epochs) + 1, val_ndcg[il, :, :].T)
        plt.title(r'$\lambda_W$=' + str(l))
        if il > 2:
            plt.xlabel('Epochs')
        if il == 0 or il == 3:
            plt.ylabel('NDCG (%)')
    leg_lambda = [r'$\lambda_H$=' + str(lh) for lh in range_lH]
    plt.legend(leg_lambda)
    plt.show()
    plt.tight_layout()

    return


def plot_val_ndcg_mf_hybrid_strict(setting):

    path_ndcg = 'outputs/' + setting + '/mf_hybrid/strict/val_ndcg.npz'

    # Load the overall validation NDCG
    ndcg_loader = np.load(path_ndcg)
    val_ndcg, range_lW = ndcg_loader['val_ndcg'], ndcg_loader['range_lW']
    Nw = len(range_lW)
    n_epochs = val_ndcg.shape[-1]

    # Plot the results
    plt.figure(figsize=(8, 5.4))
    for il, l in enumerate(range_lW):
        plt.subplot(2, Nw // 2, il + 1)
        plt.plot(np.arange(n_epochs) + 1, val_ndcg[il, :].T)
        plt.title(r'$\lambda_W$=' + str(l))
        if il > 2:
            plt.xlabel('Epochs')
        if il == 0 or il == 3:
            plt.ylabel('NDCG (%)')
    plt.show()
    plt.tight_layout()

    return


def plot_val_ndcg_mf_uni_relaxed(setting):

    path_ndcg = 'outputs/' + setting + '/mf_uni/relaxed/val_ndcg.npz'

    # Load the overall validation NDCG
    ndcg_loader = np.load(path_ndcg)
    val_ndcg, range_lW, range_lH = ndcg_loader['val_ndcg'], ndcg_loader['range_lW'], ndcg_loader['range_lH']
    Nw = len(range_lW)
    n_epochs = val_ndcg.shape[-1]

    # Plot the results
    plt.figure(figsize=(9.1, 2.55))
    for il, l in enumerate(range_lW):
        plt.subplot(1, Nw, il + 1)
        plt.plot(np.arange(n_epochs) + 1, val_ndcg[il, :, :].T)
        plt.title(r'$\lambda_W$=' + str(l))
        plt.xlabel('Epochs')
        if il == 0:
            plt.ylabel('NDCG (%)')
            leg_lambda = [r'$\lambda_H$=' + str(lh) for lh in range_lH]
            plt.legend(leg_lambda)
    plt.show()
    plt.tight_layout()

    return


def plot_val_ndcg_mf_uni_strict(setting):

    path_ndcg = 'outputs/' + setting + '/mf_uni/strict/val_ndcg.npz'

    # Load the overall validation NDCG
    ndcg_loader = np.load(path_ndcg)
    val_ndcg, range_lW = ndcg_loader['val_ndcg'], ndcg_loader['range_lW']
    Nw = len(range_lW)
    n_epochs = val_ndcg.shape[-1]

    # Plot the results
    plt.figure(figsize=(9.1, 2.55))
    for il, l in enumerate(range_lW):
        plt.subplot(1, Nw, il + 1)
        plt.plot(np.arange(n_epochs) + 1, val_ndcg[il, :].T)
        plt.title(r'$\lambda_W$=' + str(l))
        plt.xlabel('Epochs')
        if il == 0:
            plt.ylabel('NDCG (%)')
    plt.show()
    plt.tight_layout()

    return


def plot_grad_flow(named_parameters):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.0001, top=0.002)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.draw()
    plt.pause(0.001)
    plt.cla()

    return


def plot_val_ndcg_ncf():

    val_ndcg = np.load('outputs/warm/ncf/val_results.npz')['val_ndcg'][:, :-1]

    plt.figure(0)
    plt.subplot(2, 1, 1)
    plt.plot(val_ndcg[0, :].T)
    plt.ylabel('NDCG (%)')
    plt.title('Multiplication')
    plt.subplot(2, 1, 2)
    plt.plot(val_ndcg[1, :].T)
    plt.ylabel('NDCG (%)')
    plt.title('Concatenation')
    plt.xlabel('Q')

    return


def plot_val_ndcg_ncacf():

    val_ndcg_warm = np.round(np.load('outputs/warm/ncacf/val_results.npz')['val_ndcg'][:, 1:, :]*100, 1)
    val_ndcg_cold = np.round(np.load('outputs/cold/ncacf/val_results.npz')['val_ndcg'][:, 1:, :]*100, 1)

    plt.figure(figsize=(6.75, 3.2))
    plt.subplot(1, 2, 1)
    plt.title('Warm-start')
    plt.plot(val_ndcg_warm[0, :, :])
    plt.ylabel('NDCG (%)')
    plt.legend(['Relaxed', 'Strict'])
    plt.xlabel('Q')
    plt.subplot(1, 2, 2)
    plt.title('Cold-start')
    plt.plot(val_ndcg_cold[0, :, :])
    plt.xlabel('Q')
    plt.show()
    plt.tight_layout()

    plt.figure(figsize=(6.75, 3.2))
    plt.subplot(1, 2, 1)
    plt.plot(val_ndcg_warm[1, :, :])
    plt.title('Warm-start')
    plt.ylabel('NDCG (%)')
    plt.xlabel('Q')
    plt.subplot(1, 2, 2)
    plt.plot(val_ndcg_cold[1, :, :])
    plt.title('Cold-start')
    plt.xlabel('Q')
    plt.show()
    plt.tight_layout()

    return


def plot_val_mf_hybrid_epiter(setting, variant, n_epochs, n_ep_it_list):

    # Load the validation NDCGs
    val_ndcg_epit = np.zeros((len(n_ep_it_list), n_epochs))
    for inep, n_ep_it in enumerate(n_ep_it_list):
        if n_ep_it == 1:
            path_ep = 'outputs/' + setting + '/mf_hybrid/' + variant + '/'
        else:
            path_ep = 'outputs/' + setting + '/mf_hybrid/' + variant + '/gd_' + str(n_ep_it) + '/'
        val_ndcg_epit[inep, :] = np.load(path_ep + 'training.npz')['val_ndcg'][:n_epochs] * 100

    # Plot the validation NDCG
    plt.figure()
    plt.plot(np.arange(n_epochs) + 1, val_ndcg_epit.T)
    plt.xlabel('Epochs')
    plt.ylabel('NDCG (%)')
    plt.legend(['$N_{gd}=1$', '$N_{gd}=2$', '$N_{gd}=5$'])

    return

# EOF
