# -*- coding: utf-8 -*-
"""
Created on 2025-04-16 (Wed) 13:31:22

Labeled and unlabeled data

References:
- https://github.com/KaiyangZhou/Dassl.pytorch

@author: I.Azuma
"""
import os
import random
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional  as F
import torch.backends.cudnn as cudnn

from torchviz import make_dot
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

import warnings
warnings.filterwarnings('ignore')

from model.utils import *
from model.route3_dael.dael_utils import *

import sys
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'
sys.path.append(BASE_DIR+'/github/GSTMDec')
from _utils import common_utils

cudnn.deterministic = True

class LossFunctions:
    eps = 1e-8

    def rec_loss(self, real, predicted, dropout_mask=None, rec_type='mse'):
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
    
    def dag_rec_loss(self, real, predicted):
        loss = torch.square(torch.norm(real - predicted, p=2))

        n = real.shape[0]
        loss = (0.5/n) * loss

        return loss
    
    def summarize_loss(self, theta_tensor, prop_tensor):
        # deconvolution loss
        assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
        deconv_loss_dic = common_utils.calc_deconv_loss(theta_tensor, prop_tensor)
        deconv_loss = deconv_loss_dic['cos_sim'] + 0.0*deconv_loss_dic['rmse']

        return deconv_loss
    
    def L1_loss(self, preds, gt):
        loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
        return loss
    
    def compute_h(self, w_adj):
        d = w_adj.shape[0]
        h = torch.trace(torch.matrix_exp(w_adj * w_adj)) - d

        return h
    
    def dag_loss(self, rec_mse, w_adj, l1_penalty, alpha, rho):
        curr_h = self.compute_h(w_adj)
        loss = rec_mse + l1_penalty * torch.norm(w_adj, p=1) \
            + alpha * curr_h + 0.5 * rho * curr_h * curr_h
        
        return loss

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, layers, units, output_dim,
                 activation=None, device=None) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.layers = layers
        self.units = units
        self.output_dim = output_dim
        self.activation = activation
        self.device = device

        mlp = []
        for i in range(layers):
            input_size = units
            if i == 0:
                input_size = input_dim
            weight = nn.Linear(in_features=input_size,
                               out_features=self.units,
                               bias=True,
                               device=self.device)
            mlp.append(weight)
            if activation is not None:
                mlp.append(activation)
        out_layer = nn.Linear(in_features=self.units,
                              out_features=self.output_dim,
                              bias=True,
                              device=self.device)
        mlp.append(out_layer)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x) -> torch.Tensor:

        x_ = x.reshape(-1, self.input_dim)
        output = self.mlp(x_)

        return output.reshape(x.shape[0], -1, self.output_dim)

# %% Reconstruction models
class BaseModel(torch.nn.Module):
    def __init__(self, seed=42):
        super(BaseModel, self).__init__()
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
    
    def multi_aug_dataloaders(self, train_data, batch_size=128, weak_noise=0.1, strong_noise=1.0, target_cells=[]):
        g = torch.Generator()
        g.manual_seed(self.seed)

        source_ds = train_data.obs['ds'].unique().tolist()
        domain_dict = {ds: i for i, ds in enumerate(source_ds)}

        w_aug_data, s_aug_data, y_prop, domains = [], [], [], []
        for i, ds in enumerate(source_ds):
            data = train_data[train_data.obs['ds'] == ds]
            x = torch.tensor(data.X).float()
            y = torch.tensor(data.obs[target_cells].values).float()

            # Add Gaussian noise to the input tensor for data augmentation
            w_aug_data.append(add_noise(x, weak_noise))
            s_aug_data.append(add_noise(x, strong_noise))
            y_prop.append(y)
            domains.append(torch.full((len(x),), domain_dict.get(ds), dtype=torch.long))

        w_aug_data = torch.cat(w_aug_data)
        s_aug_data = torch.cat(s_aug_data)
        y_prop = torch.cat(y_prop)
        domains = torch.cat(domains)

        # Create DataLoader
        tr_data_w = torch.FloatTensor(w_aug_data)
        tr_data_s = torch.FloatTensor(s_aug_data)
        tr_labels = torch.FloatTensor(y_prop)
        tr_domains = torch.LongTensor(domains)
        dataset = Data.TensorDataset(tr_data_w, tr_data_s, tr_labels, tr_domains)
        aug_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

        return aug_loader



class AE(BaseModel):
    """ Reconstruction with autoencoder (AE) model. """
    def __init__(self, option_list, seed=42):
        super(AE, self).__init__()

        self.seed = seed
        self.batch_size = option_list['batch_size']
        self.feature_num = option_list['feature_num']
        self.latent_dim = option_list['latent_dim']

        self.num_epochs = option_list['epochs']
        self.lr = option_list['learning_rate']
        self.early_stop = option_list['early_stop']
        self.outdir = option_list['SaveResultsDir']

        self.losses = LossFunctions()
        self.activation = torch.nn.LeakyReLU(0.05)  # NOTE: default nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.encoder = nn.Sequential(EncoderBlock(self.feature_num, 512, 0), 
                                     EncoderBlock(512, self.latent_dim, 0.2))
        self.decoder = nn.Sequential(DecoderBlock(self.latent_dim, 512, 0.2),
                                     DecoderBlock(512, self.feature_num, 0))

    def forward(self, x):  # NOTE: x: (batch_size, feature_num)
        batch_size = x.size(0)

        # 1. Encoder
        out = self.encoder(x).view(batch_size, -1)

        # 2. Decoder
        rec = self.decoder(out)
        
        return rec, out

# %% main model
class Experts(nn.Module):
    def __init__(self, n_source, fdim1, fdim2, num_classes):
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fdim1, fdim2),
                    nn.ReLU(),
                    nn.Linear(fdim2, num_classes)
                )
                for _ in range(n_source)
            ]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, domain, x):
        """
        domain: (batch_size,) int tensor
        x: (batch_size, fdim1) float tensor
        """
        outputs = []
        for expert in self.linears:
            outputs.append(expert(x))  # pass to all experts（batch_size, num_classes）
        outputs = torch.stack(outputs, dim=1)  # (batch_size, n_source, num_classes)

        domain = domain.unsqueeze(1).unsqueeze(2).expand(-1, 1, outputs.size(2))  # (batch_size, 1, num_classes)
        out = outputs.gather(1, domain).squeeze(1)  # (batch_size, num_classes)

        out = self.softmax(out)
        return out

class DAEL(nn.Module):
    """
    Domain Adaptive Ensemble Learning.
    https://arxiv.org/abs/2003.07325.
    """

    def __init__(self, option_list, seed=42):
        super().__init__()
        #n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        #batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        n_domain = option_list['n_domain']
        batch_size = option_list['batch_size']

        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain
        self.fdim1 = option_list['feature_num']
        self.fdim2 = option_list['hidden_dim']
        self.celltype_num = option_list['celltype_num']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.E = Experts(self.n_domain, self.fdim1, self.fdim2, self.celltype_num)
        self.E.to(self.device)

class AE_DAEL(BaseModel):
    """ Reconstruction with autoencoder (AE) model. """
    def __init__(self, option_list, seed=42):
        super(AE_DAEL, self).__init__()

        self.weak_noise = option_list['weak_noise']
        self.strong_noise = option_list['strong_noise']
        self.n_domain = option_list['n_domain']

        self.seed = seed
        self.batch_size = option_list['batch_size']
        self.feature_num = option_list['feature_num']
        self.latent_dim = option_list['latent_dim']
        self.hidden_dim = option_list['hidden_dim']
        self.celltype_num = option_list['celltype_num']

        self.num_epochs = option_list['epochs']
        self.lr = option_list['learning_rate']
        self.early_stop = option_list['early_stop']
        self.outdir = option_list['SaveResultsDir']

        self.losses = LossFunctions()
        self.activation = torch.nn.LeakyReLU(0.05)  # NOTE: default nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # AutoEncoder
        self.encoder = nn.Sequential(
            EncoderBlock(self.feature_num, 512, 0),
            EncoderBlock(512, self.latent_dim, 0.2)
        )
        self.decoder = nn.Sequential(
            DecoderBlock(self.latent_dim, 512, 0.2),
            DecoderBlock(512, self.feature_num, 0)
        )

        # DAEL
        self.E = Experts(self.n_domain, self.latent_dim, self.hidden_dim, self.celltype_num)
        self.E.to(self.device)

        self.losses = LossFunctions()
        self.activation = torch.nn.LeakyReLU(0.05)


    def forward(self, x, domain_label=None):
        batch_size = x.size(0)

        # Encoder
        latent = self.encoder(x).view(batch_size, -1)

        # Decoder
        rec = self.decoder(latent)

        # Predictor
        if domain_label is not None:
            domain_preds = self.E(domain_label, x)  # shape: (B, celltype_num)
        else:
            domain_preds = None

        return rec, latent, domain_preds

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

