#!/usr/bin/env python3
"""
Created on 2025-06-27 (Fri) 11:46:39

Tissue-adaptive Representation via Integrated graph Autoencoder for Deconvolution

@author: I.Azuma
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings('ignore')

from model.utils import *

import sys
BASE_DIR = '/workspace/cluster/HDD/azuma/TopicModel_Deconv'
sys.path.append(BASE_DIR + '/github/TRIAD')
from _utils import common_utils

cudnn.deterministic = True

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

    def dag_rec_loss(self, real, predicted):
        loss = torch.square(torch.norm(real - predicted, p=2))
        n = real.shape[0]
        loss = (0.5 / n) * loss
        return loss

    def summarize_loss(self, theta_tensor, prop_tensor):
        # deconvolution loss
        assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
        deconv_loss_dic = common_utils.calc_deconv_loss(theta_tensor, prop_tensor)
        deconv_loss = deconv_loss_dic['cos_sim'] + 0.0 * deconv_loss_dic['rmse']
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


class MLP(nn.Module):
    def __init__(self, input_dim, layers, units, output_dim, activation=None, device=None) -> None:
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

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(LinearBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.Dropout(p=do_rates, inplace=False),
                                   nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        out = self.layer(x)
        return out

# GRL (Gradient Reversal Layer)
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(context, x, constant):
        context.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(context, grad):
        return grad.neg() * context.constant, None

class TRIAD(nn.Module):
    def __init__(self, option_list, seed=42):
        super(TRIAD, self).__init__()

        self.seed = seed
        self.batch_size = option_list['batch_size']
        self.feature_num = option_list['feature_num']
        self.latent_dim = option_list['latent_dim']
        self.hidden_dim = option_list['hidden_dim']
        self.hidden_layers = option_list['hidden_layers']
        self.celltype_num = option_list['celltype_num']
        self.num_epochs = option_list['epochs']
        self.lr = option_list['learning_rate']
        self.early_stop = option_list['early_stop']
        self.outdir = option_list['SaveResultsDir']
        self.pred_loss_type = option_list['pred_loss_type']

        self.dag_w = option_list['dag_w']
        self.pred_w = option_list['pred_w']
        self.disc_w = option_list['disc_w']

        self.losses = LossFunctions()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = torch.nn.LeakyReLU(0.05)

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

        self.embedder = nn.Sequential(LinearBlock(self.feature_num, 512, 0), 
                                      LinearBlock(512, self.latent_dim, 0.2))

        self.predictor = nn.Sequential(nn.Linear(self.latent_dim, 64),
                                       nn.Dropout(p=0.2, inplace=False),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Linear(64, self.celltype_num),
                                       nn.Softmax(dim=1))
        
        self.discriminator = nn.Sequential(nn.Linear(self.latent_dim, 64),
                                           nn.BatchNorm1d(64),
                                           nn.Dropout(p=0.2, inplace=False),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.Linear(64, 1),
                                           nn.Sigmoid()) 

    
    def forward(self, x, alpha=1.0):  # NOTE: x: (batch_size, feature_num)
        batch_size = x.size(0)

        # 1. Encoder
        x = x.reshape((batch_size, x.size(1), 1))  # x: (batch_size, feature_num, 1)
        out = self.encoder(x)  # out: (batch_size, feature_num, hidden_dim)

        # 2. Decoder
        self.w_adj = self._preprocess_graph(self.w)
        out2 = torch.einsum('ijk,jl->ilk', out, self.w_adj)  # emb2: (batch_size, feature_num, hidden_dim)
        rec = self.decoder(out2)

        # 3. Mean embedding (batch_size, feature_num, hidden_dim) --> (batch_size, feature_num)
        out_mean = torch.mean(out, dim=2)
        emb = self.embedder(out_mean)  # (batch_size, latent_dim)

        # 3. Predictor
        pred = self.predictor(emb)

        # 4. Domain classifier with GRL
        domain_emb = GradientReversalLayer.apply(emb, alpha)
        domain = self.discriminator(domain_emb)

        return rec, pred, domain

    def _preprocess_graph(self, w_adj):
        return (1. - torch.eye(w_adj.shape[0], device=self.device)) * w_adj
