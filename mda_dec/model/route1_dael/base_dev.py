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
class Experts(nn.Module):
    def __init__(self, n_source, fdim, num_classes):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(fdim, num_classes) for _ in range(n_source)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, i, x):
        x = self.linears[i](x)
        x = self.softmax(x)
        return x

class DAEL(nn.Module):
    """
    Domain Adaptive Ensemble Learning.
    https://arxiv.org/abs/2003.07325.
    """

    def __init__(self, option_list, seed=42):
        super().__init__(cfg)
        #n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        #batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        n_domain = option_list['n_domain']
        batch_size = option_list['batch_size']
        if n_domain <= 0:
            n_domain = self.num_source_domains
        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain


    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.DAEL.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
    
    def build_model(self):
        cfg = self.cfg

        print("Building F")
        self.F = SimpleNet(cfg, cfg.MODEL, 0)
        self.F.to(self.device)
        print("# params: {:,}".format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model("F", self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print("Building E")
        self.E = Experts(self.num_source_domains, fdim, self.num_classes)
        self.E.to(self.device)
        print("# params: {:,}".format(count_num_param(self.E)))
        self.optim_E = build_optimizer(self.E, cfg.OPTIM)
        self.sched_E = build_lr_scheduler(self.optim_E, cfg.OPTIM)
        self.register_model("E", self.E, self.optim_E, self.sched_E)

    def forward_backward(self, batch_x, batch_u):
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, domain_x, input_u, input_u2 = parsed_data

        input_x = torch.split(input_x, self.split_batch, 0)
        input_x2 = torch.split(input_x2, self.split_batch, 0)
        label_x = torch.split(label_x, self.split_batch, 0)
        domain_x = torch.split(domain_x, self.split_batch, 0)
        domain_x = [d[0].item() for d in domain_x]

        # Generate pseudo label
        with torch.no_grad():
            feat_u = self.F(input_u)
            pred_u = []
            for k in range(self.num_source_domains):
                pred_uk = self.E(k, feat_u)
                pred_uk = pred_uk.unsqueeze(1)
                pred_u.append(pred_uk)
            pred_u = torch.cat(pred_u, 1)  # (B, K, C)
            # Get the highest probability and index (label) for each expert
            experts_max_p, experts_max_idx = pred_u.max(2)  # (B, K)
            # Get the most confident expert
            max_expert_p, max_expert_idx = experts_max_p.max(1)  # (B)
            pseudo_label_u = []
            for i, experts_label in zip(max_expert_idx, experts_max_idx):
                pseudo_label_u.append(experts_label[i])
            pseudo_label_u = torch.stack(pseudo_label_u, 0)
            pseudo_label_u = create_onehot(pseudo_label_u, self.num_classes)
            pseudo_label_u = pseudo_label_u.to(self.device)
            label_u_mask = (max_expert_p >= self.conf_thre).float()

        loss_x = 0
        loss_cr = 0
        acc_x = 0

        feat_x = [self.F(x) for x in input_x]
        feat_x2 = [self.F(x) for x in input_x2]
        feat_u2 = self.F(input_u2)

        for feat_xi, feat_x2i, label_xi, i in zip(
            feat_x, feat_x2, label_x, domain_x
        ):
            cr_s = [j for j in domain_x if j != i]

            # Learning expert
            pred_xi = self.E(i, feat_xi)
            loss_x += (-label_xi * torch.log(pred_xi + 1e-5)).sum(1).mean()
            expert_label_xi = pred_xi.detach()
            acc_x += compute_accuracy(pred_xi.detach(),
                                      label_xi.max(1)[1])[0].item()

            # Consistency regularization
            cr_pred = []
            for j in cr_s:
                pred_j = self.E(j, feat_x2i)
                pred_j = pred_j.unsqueeze(1)
                cr_pred.append(pred_j)
            cr_pred = torch.cat(cr_pred, 1)
            cr_pred = cr_pred.mean(1)
            loss_cr += ((cr_pred - expert_label_xi)**2).sum(1).mean()

        loss_x /= self.n_domain
        loss_cr /= self.n_domain
        acc_x /= self.n_domain

        # Unsupervised loss
        pred_u = []
        for k in range(self.num_source_domains):
            pred_uk = self.E(k, feat_u2)
            pred_uk = pred_uk.unsqueeze(1)
            pred_u.append(pred_uk)
        pred_u = torch.cat(pred_u, 1)
        pred_u = pred_u.mean(1)
        l_u = (-pseudo_label_u * torch.log(pred_u + 1e-5)).sum(1)
        loss_u = (l_u * label_u_mask).mean()

        loss = 0
        loss += loss_x
        loss += loss_cr
        loss += loss_u * self.weight_u
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss_x": loss_x.item(),
            "acc_x": acc_x,
            "loss_cr": loss_cr.item(),
            "loss_u": loss_u.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


def preprocess(trainingdatapath, source='data6k', target='sdy67', 
               priority_genes=[], target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], n_samples=None, n_vtop=None):
    assert target in ['sdy67', 'GSE65133', 'donorA', 'donorC', 'data6k', 'data8k']
    pbmc = sc.read_h5ad(trainingdatapath)
    test = pbmc[pbmc.obs['ds']==target]

    if n_samples is not None:
        np.random.seed(42)
        idx = np.random.choice(8000, n_samples, replace=False)
        donorA = pbmc[pbmc.obs['ds']=='donorA'][idx]
        donorC = pbmc[pbmc.obs['ds']=='donorC'][idx]
        data6k = pbmc[pbmc.obs['ds']=='data6k'][idx]
        data8k = pbmc[pbmc.obs['ds']=='data8k'][idx]
    
    else:    
        donorA = pbmc[pbmc.obs['ds']=='donorA']
        donorC = pbmc[pbmc.obs['ds']=='donorC']
        data6k = pbmc[pbmc.obs['ds']=='data6k']
        data8k = pbmc[pbmc.obs['ds']=='data8k']

    if source == 'all':
        train = anndata.concat([donorA, donorC, data6k, data8k])
    else:
        if n_samples is not None:
            train = pbmc[pbmc.obs['ds']==source][idx]
        else:
            train = pbmc[pbmc.obs['ds']==source]

    train_y = train.obs[target_cells]
    test_y = test.obs[target_cells]
    
    if n_vtop is None:
        #### variance cut off
        label = test.X.var(axis=0) > 0.1  # FIXME: mild cut-off
        label_idx = np.where(label)[0]
    else:
        #### top 1000 highly variable genes
        label_idx = np.argsort(-train.X.var(axis=0))[:n_vtop]
    
    # add priority genes
    priority_label = np.array([True if gene in priority_genes else False for gene in train.var_names])
    priority_idx = np.where(priority_label)[0]
    print(f"Priority genes: {np.sum(priority_label)}/{len(priority_genes)} genes")
    label_idx = np.unique(np.concatenate([label_idx, priority_idx]))
    gene_names = train.var_names[label_idx]
    
    train_data = train[:, label_idx]
    train_data.X = np.log2(train_data.X + 1)
    test_data = test[:, label_idx]
    if target != 'GSE65133':
        test_data.X = np.log2(test_data.X + 1)
    else:
        # GSE65133 is already log2 transformed
        test_data.X = test_data.X

    print("Train data shape: ", train_data.X.shape)
    print("Test data shape: ", test_data.X.shape)

    return train_data, test_data, train_y, test_y, gene_names

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
