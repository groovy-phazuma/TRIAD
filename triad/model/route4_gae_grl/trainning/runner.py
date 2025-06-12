# -*- coding: utf-8 -*-
"""
Created on 2025-03-18 (Tue) 13:44:23

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'

import gc
import anndata
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.chdir(BASE_DIR+'/github/GSTMDec/dann_autoencoder_dec/model/route4_gae_grl')

from models.gae_grl_col import *

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch.nn.functional  as F
from torchviz import make_dot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




