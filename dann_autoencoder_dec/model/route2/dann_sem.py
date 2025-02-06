# -*- coding: utf-8 -*-
"""
Created on 2025-02-04 (Tue) 13:43:07

Reference
- scpDeconv
- DeepSEM

@author: I.Azuma
"""
# %%
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
Tensor = torch.cuda.FloatTensor

def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


# We followed implement in https://github.com/jariasf/GMVAE/tree/master/pytorch
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

    def log_normal(self, x, mu, var):

        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).cuda()).sum(0) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))


class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, ):
        y = self.gumbel_softmax_sample(logits, temperature)
        return y

    def forward(self, x, temperature=1.0):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature)
        return logits, prob, y


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu.squeeze(2), logvar.squeeze(2)
    
class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(InferenceNet, self).__init__()
        self.inference_qyx = torch.nn.ModuleList([
            nn.Linear(n_gene, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            GumbelSoftmax(z_dim, y_dim)  # >> logits, prob, y
        ])
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(x_dim + y_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            Gaussian(z_dim, 1)  # >> mu, logvar
        ])

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std

        return z

    def qyx(self, x, temperature):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                x = layer(x, temperature)  # x: (batch_size, feature_dim, 1)
            else:
                x = layer(x)
        return x

    def qzxy(self, x, y):
        concat = torch.cat((x, y.unsqueeze(1).repeat(1, x.shape[1], 1)), dim=2)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, adj, temperature=1.0):
        logits, prob, y = self.qyx(x.squeeze(2), temperature)
        mu, logvar = self.qzxy(x, y)
        mu_ori = mu
        mu = torch.matmul(mu.float(), adj.float())  
        logvar = torch.matmul(logvar.float(), adj.float())
        var = torch.exp(logvar)
        z = self.reparameterize(mu, var)
        output = {'mean'  : mu, 'var': var, 'gaussian': z,
                  'logits': logits, 'prob_cat': prob, 'categorical': y, 'mu_ori': mu_ori}
        return output

class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(GenerativeNet, self).__init__()
        self.n_gene = n_gene
        self.y_mu = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))
        self.y_var = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))

        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(1, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_logvar = self.y_var(y)
        return y_mu, y_logvar

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y, adj):
        y_mu, y_logvar = self.pzy(y)
        y_mu = torch.matmul(y_mu.float(), adj.float())
        y_logvar = torch.matmul(y_logvar.float(), adj.float())
        y_var = torch.exp(y_logvar)
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        output = {'y_mean': y_mu.view(-1, self.n_gene), 'y_var': y_var.view(-1, self.n_gene), 'x_rec': x_rec}
        return output

class Predictor(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(Predictor, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    
    def forward(self, x):
        out = self.layer(x)
        return out

class DANN_SEM(nn.Module):
    def __init__(self, adj_A, x_dim, z_dim, y_dim, celltype_num, pred_loss_type='L1'):
        super(DANN_SEM, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True, name='adj_A'))
        self.n_gene = n_gene = len(adj_A)
        self.celltype_num = celltype_num
        self.pred_loss_type = pred_loss_type
        nonLinear = nn.Tanh()
        self.inference = InferenceNet(x_dim, z_dim, y_dim, n_gene, nonLinear)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim, n_gene, nonLinear)
        self.predictor = nn.Sequential(Predictor(y_dim, y_dim//2, 0.2), 
                                          nn.Linear(y_dim//2, celltype_num), 
                                          nn.Softmax(dim=1))
        self.losses = LossFunctions()
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def _one_minus_A_t(self, adj):
        adj_normalized = Tensor(np.eye(adj.shape[0])) - (adj.transpose(0, 1))
        return adj_normalized

    def forward(self, x, frac_true, dropout_mask, temperature=1.0):
        assert frac_true.shape[1] == self.celltype_num
        x_ori = x
        x = x.view(x.size(0), -1, 1)  # (batch_size, feature_dim, 1)
        mask = Variable(torch.from_numpy(np.ones(self.n_gene) - np.eye(self.n_gene)).float(), requires_grad=False).cuda()
        adj_A_t = self._one_minus_A_t(self.adj_A * mask)
        adj_A_t_inv = torch.inverse(adj_A_t)

        # encoder
        out_inf = self.inference(x, adj_A_t, temperature)
        z, y = out_inf['gaussian'], out_inf['categorical']
        z_inv = torch.matmul(z.float(), adj_A_t_inv.float())

        # decoder
        out_gen = self.generative(z_inv, y, adj_A_t)
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        dec = output['x_rec']
        loss_rec = self.losses.reconstruction_loss(x_ori, output['x_rec'], dropout_mask, 'mse')
        loss_gauss = self.losses.gaussian_loss(z, output['mean'], output['var'], output['y_mean'], output['y_var'])# * opt.beta
        loss_cat = (-self.losses.entropy(output['logits'], output['prob_cat']) - np.log(0.1))# * opt.beta

        # predictor
        frac_pred = self.predictor(output['prob_cat'])
        output['frac_pred'] = frac_pred

        # calculate loss 
        if self.pred_loss_type == 'L1':
            loss_pred = L1_loss(frac_pred, frac_true)
        elif self.pred_loss_type == 'custom':
            loss_pred = summarize_loss(frac_pred, frac_true)
        else:
            raise ValueError("Invalid prediction loss type.")

        # total loss
        loss_dict = {'loss_rec': loss_rec, 'loss_gauss': loss_gauss, 'loss_cat': loss_cat, 'loss_pred': loss_pred}

        return loss_dict, dec, y, output

# %%
def initialize_A(topic_nums=16):
    A = np.ones([topic_nums, topic_nums]) / (topic_nums - 1) + (
        np.random.rand(topic_nums * topic_nums) * 0.0002
    ).reshape([topic_nums, topic_nums])
    for i in range(topic_nums):
        A[i, i] = 0
    A = A.astype(np.float32)
    return A

def prepare_dataloader(source_data, target_data, batch_size):
    ### Prepare data loader for training ###
    # Source dataset
    source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
    source_data_x = source_data.X.astype(np.float32)
    source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
    
    tr_data = torch.FloatTensor(source_data_x)
    tr_labels = torch.FloatTensor(source_data_y)
    source_dataset = Data.TensorDataset(tr_data, tr_labels)
    train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

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
    train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
    test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    return train_source_loader, train_target_loader, test_target_loader, labels, used_features

def summarize_loss(theta_tensor, prop_tensor):
    # deconvolution loss
    assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
    deconv_loss_dic = common_utils.calc_deconv_loss(theta_tensor, prop_tensor)
    deconv_loss = deconv_loss_dic['cos_sim'] + 0.0*deconv_loss_dic['rmse']

    return deconv_loss
