# -*- coding: utf-8 -*-
"""
Created on 2025-02-28 (Fri) 08:21:25

- Gaussian Mixture Variational Autoencoder (GMVAE)
    - Based on NSEM-GMHTM (Simple decoder version)
- Domain adaptation with Gradient Reversal Layer (GRL)

@author: I.Azuma
"""
# %%
import os
import random
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
from model.route3_mysimple.customized_linear import CustomizedLinear

import sys
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'
sys.path.append(BASE_DIR+'/github/GSTMDec')
from _utils import common_utils

adj_flag = False
print("adj_flag: ",adj_flag)

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
    
    def log_normal(self, x, mu, var, eps=1e-8):
        """Logarithm of normal distribution with mean=mu and variance=var
            log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
            x: (array) corresponding array containing the input
            mu: (array) corresponding array containing the mean 
            var: (array) corresponding array containing the variance

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        if eps > 0.0:
            var = var + eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).cuda()).sum(0)
            + torch.log(var)
            + torch.pow(x - mu, 2) / var,
            dim=-1,
        )
    
    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):  
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.sum()
    
    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.sum(torch.sum(targets * log_q, dim=-1))
    
    def summarize_loss(self, theta_tensor, prop_tensor):
        # deconvolution loss
        assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
        deconv_loss_dic = common_utils.calc_deconv_loss(theta_tensor, prop_tensor)
        deconv_loss = deconv_loss_dic['cos_sim'] + 0.0*deconv_loss_dic['rmse']

        return deconv_loss
    
    def L1_loss(self, preds, gt):
        loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
        return loss


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

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 
  
  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y

class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu, logvar


# Encoder
class InferenceNet(nn.Module):
    def __init__(self,topic_num_1,topic_num_2,topic_num_3,hidden_num,y_dim=10,nonLinear=None):
        super(InferenceNet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(topic_num_1,topic_num_2), nn.BatchNorm1d(topic_num_2), nonLinear)
        self.encoder_2 = nn.Sequential(nn.Linear(topic_num_2,topic_num_3), nn.BatchNorm1d(topic_num_3), nonLinear)
        self.inference_qyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num_3, hidden_num),
                nn.BatchNorm1d(hidden_num),
                nonLinear,
                GumbelSoftmax(hidden_num, y_dim),
            ]
        )
        self.inference_qzyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num_3 + y_dim, hidden_num),
                nn.BatchNorm1d(hidden_num),
                nonLinear,
                Gaussian(hidden_num, topic_num_3),
            ]
        )

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    # q(y|x)
    def qyx3(self, x,temperature,hard):
        num_layers = len(self.inference_qyx3)
        for i, layer in enumerate(self.inference_qyx3):
            if i == num_layers - 1:
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x
    
    # q(z|x,y)
    def qzxy3(self, x, y):
        concat = torch.cat((x.squeeze(2), y), dim=1)
        for layer in self.inference_qzyx3:
            concat = layer(concat)
        return concat


    def forward(self, x, adj_1=None, adj_2=None, adj_3=None, temperature=1.0, hard=0):
        if adj_flag ==True:
            x_1 = torch.matmul(adj_1.to(torch.float32),x.squeeze(2).T).T
            x_2 = self.encoder(x_1)
            x_2 = torch.matmul(adj_2.to(torch.float32),x_2.T).T
            x_3 = self.encoder_2(x_2)
            x_3 = torch.matmul(adj_3.to(torch.float32),x_3.T).T
        else:
            x_1 = x.squeeze(2)
            x_2 = self.encoder(x_1)
            x_3 = self.encoder_2(x_2)                     
        logits_3, prob_3, y_3  = self.qyx3(x_3,temperature, hard=0)
        #print(y_3)
        mu_3, logvar_3 = self.qzxy3(x_3.view(x_3.size(0), -1, 1), y_3)
        var_3 = torch.exp(logvar_3)
        # reparameter: td1
        z_3 = self.reparameterize(mu_3, var_3)
        output_3 = {"mean": mu_3, "var": var_3, "gaussian": z_3, "categorical": y_3,'logits': logits_3, 'prob_cat': prob_3}
        return output_3 

# Decoder
class GenerativeNet(nn.Module):
    def __init__(self, topic_num_1,topic_num_2,topic_num_3, y_dim=10, nonLinear=None):
        super(GenerativeNet, self).__init__()
        self.y_mu_1 = nn.Sequential(nn.Linear(y_dim, topic_num_3))
        self.y_var_1 = nn.Sequential(nn.Linear(y_dim, topic_num_3))
        self.decoder = nn.Sequential(CustomizedLinear(torch.ones(topic_num_3,topic_num_2),bias=False), nn.BatchNorm1d(topic_num_2), nonLinear)
        self.decoder_2 = nn.Sequential(CustomizedLinear(torch.ones(topic_num_2,topic_num_1),bias=False), nn.BatchNorm1d(topic_num_1), nonLinear)

        if True:
            print('Constraining decoder to positive weights', flush=True)

            self.decoder[0].reset_params_pos()
            self.decoder[0].weight.data *= self.decoder[0].mask        
            self.decoder_2[0].reset_params_pos()    
            self.decoder_2[0].weight.data *= self.decoder_2[0].mask 

        self.generative_pxz = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_3),
                nonLinear,
            ]
        )
        self.generative_pxz_1 = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_2),
                nonLinear,
            ]
        )
        self.generative_pxz_2 = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_1),
                nonLinear,
            ]
        )

    def pzy1(self, y):
        y_mu = self.y_mu_1(y)
        y_logvar = self.y_var_1(y)
        return y_mu, y_logvar
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z
    def pxz_1(self, z):
        for layer in self.generative_pxz_1:
            z = layer(z)
        return z
    def pxz_2(self, z):
        for layer in self.generative_pxz_2:
            z = layer(z)
        return z

    def forward(
        self,
        z,
        y_3,
        adj_inv_1=None,
        adj_inv_2=None,
        adj_inv_3=None,
    ):
        y_mu_3, y_logvar_3 = self.pzy1(y_3)
        y_var_3 = torch.exp(y_logvar_3)

        if adj_flag ==True:
            z = torch.matmul(adj_inv_3.to(torch.float32), z.T).T
            out_1 = self.pxz(z)
            z_2 = self.decoder(z)
            z_2 = torch.matmul(adj_inv_2.to(torch.float32), z_2.T).T
            out_2 = self.pxz_1(z_2)
            z_3 = self.decoder_2(z_2)
            z_3 =  torch.matmul(adj_inv_1.to(torch.float32), z_3.T).T
            out_3 = self.pxz_2(z_3)
        else:
            out_1 = self.pxz(z)
            z_2 = self.decoder(z)
            out_2 = self.pxz_1(z_2)
            z_3 = self.decoder_2(z_2)
            out_3 = self.pxz_2(z_3)
        
        m0 = self.decoder[0].weight.data
        m1 = self.decoder_2[0].weight.data

        # torch.Size([batch_size, topic_n])

        output = {"x_rec1": out_1, "x_rec2": out_2, "x_rec3": out_3, "y_mean": y_mu_3, "y_var": y_var_3}

        return output


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   #nn.BatchNorm1d(out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
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


class MultiTaskAutoEncoder(nn.Module):
    def __init__(self, option_list, seed=42):
        super(MultiTaskAutoEncoder, self).__init__()
        self.seed = seed
        self.batch_size = option_list['batch_size']
        self.feature_num = option_list['feature_num']
        self.latent_dim = option_list['latent_dim']
        self.celltype_num = option_list['celltype_num']
        self.gene_num = option_list['gene_num']
        self.topic_num_1 = option_list['topic_num_1']
        self.topic_num_2 = option_list['topic_num_2']
        self.topic_num_3 = option_list['topic_num_3']

        self.num_epochs = option_list['epochs']
        self.lr = option_list['learning_rate']
        self.early_stop = option_list['early_stop']
        self.outdir = option_list['SaveResultsDir']
        self.pred_loss_type = option_list['pred_loss_type']
        self.loss_ref = option_list['loss_ref']
        assert self.loss_ref in ['pred_loss', 'total_loss'], "!! Invalid loss reference !!"

        self.rec_w = option_list['rec_w']
        self.pred_w = option_list['pred_w']
        self.disc_w = option_list['disc_w']

        self.losses = LossFunctions()

        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # embeddings
        xavier_init = torch.distributions.Uniform(-0.05,0.05)
        self.gene_embed = nn.Parameter(torch.rand(self.latent_dim, self.feature_num))  # phi_3
        self.topic_embed_1 = nn.Parameter(xavier_init.sample((self.topic_num_1, self.latent_dim)))
        self.topic_embed_2 = nn.Parameter(xavier_init.sample((self.topic_num_2, self.latent_dim)))
        self.topic_embed_3 = nn.Parameter(xavier_init.sample((self.topic_num_3, self.latent_dim)))

        # layers
        self.encoder = nn.Sequential(nn.Linear(self.feature_num, self.topic_num_1), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(self.topic_num_1, self.feature_num))

        y_dim = 10  # FIXME: hard-coded
        self.inference = InferenceNet(self.topic_num_1,self.topic_num_2,self.topic_num_3,self.latent_dim,y_dim,nn.Tanh())
        self.generative = GenerativeNet(self.topic_num_1,self.topic_num_2,self.topic_num_3,y_dim,nn.Tanh())


        self.predictor = nn.Sequential(EncoderBlock(self.topic_num_3, 64, 0.2),
                                       nn.Linear(64, self.celltype_num),
                                       nn.Softmax(dim=1))
        
        self.discriminator = nn.Sequential(nn.Linear(self.topic_num_3, 64),
                                           nn.BatchNorm1d(64),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.Dropout(p=0.2, inplace=False),
                                           nn.Linear(64, 1),
                                           nn.Sigmoid())
    
    def encode(self, x):
        p1 = self.encoder(x)
        return p1
    
    def decode(self, x):
        p_fin = self.decoder(x)
        return p_fin
    
    """
    def decode(self, x_ori, out_1, out_2, out_3):
        out_1 = torch.softmax(out_1, dim=1)
        out_2 = torch.softmax(out_2, dim=1)
        out_3 = torch.softmax(out_3, dim=1)

        self.theta_1 = out_1
        self.theta_2 = out_2
        self.theta_3 = out_3

        # legacy
        phi_1 = torch.softmax(self.topic_embed_1 @ self.gene_embed, dim=1)
        phi_2 = torch.softmax(self.topic_embed_2 @ self.gene_embed, dim=1)
        phi_3 = torch.softmax(self.topic_embed_3 @ self.gene_embed, dim=1)

        p1 = out_3 @ phi_1 
        p2 = out_2 @ phi_2 
        p3 = out_1 @ phi_3
        p_fin = (p1.T+p2.T+p3.T)/3.0

        return p_fin.T
    """

    
    def forward(self, x, temperature=1.0, alpha=1.0):
        x_ori = x
        x = self.encode(x)
        x = x.view(x.size(0), -1, 1)  # (batch_size, topic_num_1, 1)

        # inference
        out_inf = self.inference(
            x, adj_1=None, adj_2=None, adj_3=None, temperature=temperature, hard=x_ori.view(x.size(0), -1, 1))

        z_3, y_3 = out_inf["gaussian"], out_inf["categorical"]

        out_dec = self.generative(z_3, y_3)
        rec = self.decode(out_dec["x_rec3"])


        pred = self.predictor(z_3)
        domain_emb = GradientReversalLayer.apply(z_3, alpha)
        domain = self.discriminator(domain_emb)

        # calculate reconstruction loss
        rec_loss = self.losses.reconstruction_loss(x_ori, rec, rec_type='mse')
        loss_gauss = (
            self.losses.gaussian_loss(
                z_3,
                out_inf["mean"],
                out_inf["var"],
                out_dec["y_mean"],
                out_dec["y_var"],
            )* 1)
        loss_cat = (-self.losses.entropy(out_inf['logits'], out_inf['prob_cat']) - np.log(0.1))

        ae_loss = {"loss_rec": rec_loss, "loss_gauss": loss_gauss, "loss_cat": loss_cat}

        return ae_loss, pred, domain

    
    def prepare_dataloader(self, source_data, target_data, batch_size):
        ### Prepare data loader for training ###
        g = torch.Generator()
        g.manual_seed(42)

        # Source dataset
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)

        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)


def preprocess(trainingdatapath, source='data6k', target='sdy67', n_samples=None, n_vtop=None):
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

    train_y = train.obs.iloc[:,:-2]
    test_y = test.obs.iloc[:,:-2]

    
    if n_vtop is None:
        #### variance cut off
        label = test.X.var(axis=0) > 0.1  # FIXME: mild cut-off
    else:
        #### top 1000 highly variable genes
        label = np.argsort(-train.X.var(axis=0))[:n_vtop]
    
    train_data = train[:, label]
    train_data.X = np.log2(train_data.X + 1)
    test_data = test[:, label]
    test_data.X = np.log2(test_data.X + 1)

    print("Train data shape: ", train_data.X.shape)
    print("Test data shape: ", test_data.X.shape)

    return train_data, test_data, train_y, test_y

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



