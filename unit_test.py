#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import torch
from training.twostages import train_val_wmf_2stages, get_optimal_2stages, get_optimal_wmf
from training.mf_hybrid import train_val_mf_hybrid, check_NGD_mf_hybrid
from training.mf_uni import train_val_mf_uni
from training.ncf import train_val_ncf
from training.ncacf import train_val_ncacf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Process on: {}'.format(device))

