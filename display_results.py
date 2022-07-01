#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

from helpers.plotters import plot_val_ndcg_ncacf, plot_val_ndcg_lW_lH, plot_val_ndcg_lW


# MF Hybrid validation results - impact of lambda
plot_val_ndcg_lW_lH('outputs/cold/mf_hybrid/relaxed/')
plot_val_ndcg_lW('outputs/cold/mf_hybrid/strict/')
plot_val_ndcg_lW_lH('outputs/warm/mf_hybrid/relaxed/')
plot_val_ndcg_lW('outputs/warm/mf_hybrid/strict/')

# MF Hybrid, influence of N_GD (and compare with the 2 stage approaches)




# MF Uni validation results - impact of lambda
plot_val_ndcg_lW_lH('outputs/cold/mf_uni/relaxed/')
plot_val_ndcg_lW('outputs/cold/mf_uni/strict/')
plot_val_ndcg_lW_lH('outputs/warm/mf_uni/relaxed/')
plot_val_ndcg_lW('outputs/warm/mf_uni/strict/')

# MF Hybrid and Uni : display best perf as well as computation time


# NCACF validation results (impact of variant, interaction model, and number of layers)
plot_val_ndcg_ncacf()


# Test results


# EOF
