# -*- coding: utf-8 -*-
"""
Created on 2025-02-13 (Thu) 14:29:50

Bug Survey >> 250213_bug_survey.py

@author: I.Azuma
"""
import os
import random
import numpy as np
import pandas as pd

from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional  as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


from model.utils import *

import warnings
warnings.filterwarnings('ignore')

import sys
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'
sys.path.append(BASE_DIR+'/github/GSTMDec')
from _utils import common_utils

# %%
class LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type='mse'):
        if rec_type == 'mse':
            if dropout_mask is None:
                loss = torch.mean((real - predicted).pow(2))
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(dropout_mask)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none').mean()
        else:
            raise Exception
        return loss

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(MLPBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=False),  # FIXME
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class BugFix(nn.Module):
    def __init__(self, n_celltype, n_gene, seed=42):
        super(BugFix, self).__init__()
        self.seed = seed
        cudnn.deterministic = True
        cudnn.benchmark = False

        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.n_gene = n_gene
        self.n_celltype = n_celltype

        self.encoder = nn.Sequential(
            MLPBlock(n_gene, 512, 0.2),
            MLPBlock(512, 256, 0.2),
            nn.Linear(256, 128),
            nn.Softmax(dim=1)
        )
        self.decoder = nn.Sequential(
            MLPBlock(128, 256, 0.2),
            MLPBlock(256, 512, 0.2),
            nn.Linear(512, n_gene)
        )
        self.predictor = nn.Sequential(
            MLPBlock(128, 32, 0.2),
            nn.Linear(32, n_celltype)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(1000000, 128),
            nn.LeakyReLU(0.2, inplace=False),  # FIXME
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.losses = LossFunctions()

        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, source_x, target_x, source_y, temperature=1.0):
        assert source_y.shape[1] == self.n_celltype

        source_feature = self.encoder(source_x)
        target_feature = self.encoder(target_x)

        source_rec = self.decoder(source_feature)
        target_rec = self.decoder(target_feature)

        source_pred = self.predictor(source_feature)
        target_pred = self.predictor(target_feature)

        # calculate loss
        loss_rec = self.losses.reconstruction_loss(target_x, target_rec, rec_type='mse')
        loss_pred = summarize_loss(source_pred, source_y)

        return loss_rec, loss_pred


def summarize_loss(theta_tensor, prop_tensor):
    # deconvolution loss
    assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
    deconv_loss_dic = common_utils.calc_deconv_loss(theta_tensor, prop_tensor)
    deconv_loss = deconv_loss_dic['cos_sim'] + 0.0*deconv_loss_dic['rmse']

    return deconv_loss

def prepare_dataloader(source_data, target_data, batch_size):
    g = torch.Generator()
    g.manual_seed(42)
    
    ### Prepare data loader for training ###
    # Source dataset
    source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
    source_data_x = source_data.X.astype(np.float32)
    source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
    
    tr_data = torch.FloatTensor(source_data_x)
    tr_labels = torch.FloatTensor(source_data_y)
    source_dataset = Data.TensorDataset(tr_data, tr_labels)
    train_source_loader = DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # Extract celltype and feature info
    labels = source_data.uns['cell_types']
    celltype_num = len(labels)
    used_features = list(source_data.var_names)

    # Target dataset
    target_data_x = target_data.X.astype(np.float32)
    target_data_y = np.random.rand(target_data.shape[0], celltype_num)

    te_data = torch.FloatTensor(target_data_x)
    te_labels = torch.FloatTensor(target_data_y)
    target_dataset = Data.TensorDataset(te_data, te_labels)

    train_target_loader = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    return train_source_loader, train_target_loader, test_target_loader, labels, used_features

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
