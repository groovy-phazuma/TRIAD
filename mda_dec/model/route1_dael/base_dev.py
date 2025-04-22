# -*- coding: utf-8 -*-
"""
Created on 2025-04-16 (Wed) 13:31:22

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
from model.route1_dael.dael_utils import *

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
    
    def aug_dataloader(self, train_data, batch_size, noise=0.1, target_cells=[]):
        """
        Add Gaussian noise to the input tensor for data augmentation.
        train_data: source + target
        """
        g = torch.Generator()
        g.manual_seed(self.seed)

        source_ds = train_data.obs['ds'].unique().tolist()
        domain_dict = {ds: i for i, ds in enumerate(source_ds)}

        aug_data, y_prop, domains = [], [], []
        for i, ds in enumerate(source_ds):
            data = train_data[train_data.obs['ds'] == ds]
            x = torch.tensor(data.X).float()
            y = torch.tensor(data.obs[target_cells].values).float()

            # Add Gaussian noise to the input tensor for data augmentation
            aug_data.append(add_noise(x, noise))
            y_prop.append(y)
            domains.append(torch.full((len(x),), domain_dict.get(ds), dtype=torch.long))

        aug_data = torch.cat(aug_data)
        y_prop = torch.cat(y_prop)
        domains = torch.cat(domains)

        self.data_x = aug_data
        self.data_y = y_prop
        self.domains = domains

        # Create DataLoader
        tr_data = torch.FloatTensor(self.data_x)
        tr_labels = torch.FloatTensor(self.data_y)
        dataset = Data.TensorDataset(tr_data, tr_labels)
        self.aug_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)


    def mix_dataloader(self, source_data, target_data, batch_size, target_cells=[]):
        """
        Combine source and target data for training.
        """
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Source dataset
        if len(target_cells) == 0:
            source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
            self.labels = source_data.uns['cell_types']
        else:
            source_ratios = [source_data.obs[ctype] for ctype in target_cells]
            self.labels = target_cells
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        self.target_data_y = np.random.rand(target_data.shape[0], len(self.labels))

        # Combine source and target data
        self.data_x = np.concatenate([self.source_data_x, self.target_data_x], axis=0)
        self.data_y = np.concatenate([self.source_data_y, self.target_data_y], axis=0)

        # Create DataLoader
        tr_data = torch.FloatTensor(self.data_x)
        tr_labels = torch.FloatTensor(self.data_y)
        dataset = Data.TensorDataset(tr_data, tr_labels)
        self.mix_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

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

class GAE(BaseModel):
    """ Reconstruction with graph autoencoder (GAE) model. """
    def __init__(self, option_list, seed=42):
        super(GAE, self).__init__()

        self.seed = seed
        self.batch_size = option_list['batch_size']
        self.feature_num = option_list['feature_num']
        self.hidden_layers = option_list['hidden_layers']
        self.hidden_dim = option_list['hidden_dim']

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

        self.encoder = MLP(input_dim=1,
                           layers=self.hidden_layers,
                           units=self.hidden_dim,
                           output_dim=self.hidden_dim,
                           activation=self.activation,
                           device=self.device)
        self.decoder = MLP(input_dim=self.hidden_dim,
                           layers=self.hidden_layers,
                           units=self.hidden_dim,
                           output_dim=1,
                           activation=self.activation,
                           device=self.device)

        w = torch.nn.init.uniform_(torch.empty(self.feature_num, self.feature_num),a=-0.1, b=0.1)
        self.w = torch.nn.Parameter(w.to(device=self.device))

    def forward(self, x, alpha=1.0):  # NOTE: x: (batch_size, feature_num)
        batch_size = x.size(0)

        # 1. Encoder
        x = x.reshape((batch_size, x.size(1), 1))  # x: (batch_size, feature_num, 1)
        out = self.encoder(x)  # out: (batch_size, feature_num, hidden_dim)

        # 2. Decoder
        self.w_adj = self._preprocess_graph(self.w)
        out2 = torch.einsum('ijk,jl->ilk', out, self.w_adj)  # emb2: (batch_size, feature_num, hidden_dim)
        rec = self.decoder(out2)
        
        return rec, out, out2
    
    def _preprocess_graph(self, w_adj):
        return (1. - torch.eye(w_adj.shape[0], device=self.device)) * w_adj

# %% main model
class Experts_Original(nn.Module):
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
        self.softmax = nn.Softmax()

    def forward(self, i, x):
        x = self.linears[i](x)
        x = self.softmax(x)
        return x

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

        print("Building E")
        self.E = Experts(self.n_domain, self.fdim1, self.fdim2, self.celltype_num)
        self.E.to(self.device)
    
    def train_tmp(self, batch):
        self.E.train()
        loss_x_epoch, loss_cr_epoch = 0, 0 
        for batch_idx, (feat, feat2, y_prop, domain) in enumerate(dael_loader):
            feat = feat.cuda()
            feat2 = feat2.cuda()
            y_prop = y_prop.cuda()
            domain = domain.cuda()

            loss_x = 0
            loss_cr = 0
            acc = 0

            for feat_i, feat2_i, label_i, i in zip(feat, feat2, y_prop, domain):
                cr_s = domain[domain != i]

                # Learning expert
                pred_i = self.E(i, feat_i)  # weak aug
                loss_x += ((label_i - pred_i) ** 2).mean()
                expert_pred_i = pred_i.detach()

                # Consistency regularization
                cr_pred = []
                for j in cr_s:
                    pred_j = self.E(j, feat2_i)  # strong aug
                    pred_j = pred_j.unsqueeze(1)
                    cr_pred.append(pred_j)
                cr_pred = torch.cat(cr_pred, 1)
                cr_pred = cr_pred.mean(1)
                loss_cr += ((cr_pred - expert_pred_i)**2).mean()

            loss_x /= self.n_domain
            loss_cr /= self.n_domain
            loss_x_epoch += loss_x.item()
            loss_cr_epoch += loss_cr.item()

            loss = 0
            loss += loss_x
            loss += loss_cr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def forward_backward(self, batch):
        parsed_data = self.parse_batch_train(batch)
        input, input2, label, domain = parsed_data

        input = torch.split(input, self.split_batch, 0)
        input2 = torch.split(input2, self.split_batch, 0)
        label = torch.split(label, self.split_batch, 0)
        domain = torch.split(domain, self.split_batch, 0)
        domain = [d[0].item() for d in domain]

        loss_x = 0
        loss_cr = 0
        acc = 0

        feat = [self.F(x) for x in input]
        feat2 = [self.F(x) for x in input2]

        for feat_i, feat2_i, label_i, i in zip(feat, feat2, label, domain):
            cr_s = [j for j in domain if j != i]

            # Learning expert
            pred_i = self.E(i, feat_i)
            loss_x += (-label_i * torch.log(pred_i + 1e-5)).sum(1).mean()
            expert_label_i = pred_i.detach()
            acc += compute_accuracy(pred_i.detach(),
                                    label_i.max(1)[1])[0].item()

            # Consistency regularization
            cr_pred = []
            for j in cr_s:
                pred_j = self.E(j, feat2_i)
                pred_j = pred_j.unsqueeze(1)
                cr_pred.append(pred_j)
            cr_pred = torch.cat(cr_pred, 1)
            cr_pred = cr_pred.mean(1)
            loss_cr += ((cr_pred - expert_label_i)**2).sum(1).mean()

        loss_x /= self.n_domain
        loss_cr /= self.n_domain
        acc /= self.n_domain

        loss = 0
        loss += loss_x
        loss += loss_cr
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss_x": loss_x.item(),
            "acc": acc,
            "loss_cr": loss_cr.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
