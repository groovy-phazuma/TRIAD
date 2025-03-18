# -*- coding: utf-8 -*-
"""
Created on 2025-02-21 (Fri) 09:06:45

Domain adaptation with Gradient Reversal Layer (GRL)

@author: I.Azuma
"""
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

import sys
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'
sys.path.append(BASE_DIR+'/github/GSTMDec')
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

class MLP(nn.Module):
    """
    Feed-forward neural networks----MLP

    """

    def __init__(self, input_dim, layers, units, output_dim,
                 activation=None, device=None) -> None:
        super(MLP, self).__init__()
        # self.desc = desc
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

"""
class EmbeddingBlock(nn.Module):
    def __init__(self, feature_num, hidden_dim, latent_dim):
        super().__init__()
        # hidden_dim --> 1
        self.compress_hidden = nn.Linear(hidden_dim, 1)
        # feature_num --> latent_dim
        self.reduce_feature = nn.Linear(feature_num, latent_dim)
    
    def forward(self, x):
        # x: (batch_size, feature_num, hidden_dim)
        x = self.compress_hidden(x)  # (batch_size, feature_num, 1)
        x = x.squeeze(-1)  # (batch_size, feature_num)
        x = self.reduce_feature(x)  # (batch_size, latent_dim)
        return x
"""

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
        self.hidden_dim = option_list['hidden_dim']
        self.d = option_list['d']
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
        self.activation = torch.nn.LeakyReLU(0.05)  # NOTE: default nn.ReLU()

        W = torch.nn.init.uniform_(torch.empty(self.d, self.d,),a=-0.1, b=0.1)
        self.w = torch.nn.Parameter(W.to(device=self.device))

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
        
        w = torch.nn.init.uniform_(torch.empty(self.d, self.d,),
                                   a=-0.1, b=0.1)
        self.w = torch.nn.Parameter(w.to(device=self.device))

        self.embedder = nn.Sequential(EncoderBlock(self.feature_num, 512, 0), 
                                      EncoderBlock(512, self.latent_dim, 0.2))
        #self.embedder = EmbeddingBlock(self.feature_num, self.hidden_dim, self.latent_dim)

        self.predictor = nn.Sequential(EncoderBlock(self.latent_dim, 64, 0.2),
                                       nn.Linear(64, self.celltype_num),
                                       nn.Softmax(dim=1))
        
        self.discriminator = nn.Sequential(nn.Linear(self.latent_dim, 64),
                                           nn.BatchNorm1d(64),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.Dropout(p=0.2, inplace=False),
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
        """
        # 3. Embedding (batch_size, feature_num, hidden_dim) --> (batch_size, feature_num)
        emb = self.embedder(out)
        """
        
        # 3. Predictor
        pred = self.predictor(emb)

        # 4. Domain classifier with GRL
        domain_emb = GradientReversalLayer.apply(emb, alpha)
        domain = self.discriminator(domain_emb)

        return rec, pred, domain
    
    def forward_stable(self, data):

        self.w_adj = self._preprocess_graph(self.w)

        x = torch.from_numpy(data).to(self.device)
        self.n, self.d = x.shape[:2]
        x = x.reshape((self.n, self.d, 1))


        out = self.encoder(x)
        print("x:", x.shape, "out:", out.shape, "w_adj:", self.w_adj.shape)
        out = torch.einsum('ijk,jl->ilk', out, self.w_adj)
        x_est = self.decoder(out)

        mse_loss = torch.square(torch.norm(x - x_est, p=2))


        return mse_loss, self.w_adj
    
    def _preprocess_graph(self, w_adj):
        return (1. - torch.eye(w_adj.shape[0], device=self.device)) * w_adj


    def load_checkpoint(self, model_path):
        self.model_da.load_state_dict(torch.load(model_path))
        self.model_da.eval()

    def prediction(self, test_target_loader=None):
        if test_target_loader is None:
            test_target_loader = self.test_target_loader
            
        self.model_da.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(test_target_loader):
            logits = self.predictor(self.encoder(x.cuda())).detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)

        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)  # random ratio is output if "real"
        return target_preds, ground_truth

    
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

