# -*- coding: utf-8 -*-
"""
Created on 2025-04-07 (Mon) 16:48:37

Model
- Reconstruction
    - without estimating gene regulatory network (included in route6)
- Prediction (deconvolution)
- Domain Adaptation

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
    
    def summarize_loss(self, theta_tensor, prop_tensor):
        # deconvolution loss
        assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
        deconv_loss_dic = common_utils.calc_deconv_loss(theta_tensor, prop_tensor)
        deconv_loss = deconv_loss_dic['cos_sim'] + 0.0*deconv_loss_dic['rmse']

        return deconv_loss
    
    def L1_loss(self, preds, gt):
        loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
        return loss

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
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
        self.d = option_list['d']
        self.celltype_num = option_list['celltype_num']
        self.num_epochs = option_list['epochs']
        self.lr = option_list['learning_rate']
        self.early_stop = option_list['early_stop']
        self.outdir = option_list['SaveResultsDir']
        self.pred_loss_type = option_list['pred_loss_type']

        self.rec_w = option_list['rec_w']
        self.pred_w = option_list['pred_w']
        self.disc_w = option_list['disc_w']

        self.losses = LossFunctions()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = torch.nn.LeakyReLU(0.05)  # NOTE: default nn.ReLU()

        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.encoder = nn.Sequential(EncoderBlock(self.feature_num, 512, 0), 
                                     EncoderBlock(512, self.latent_dim, 0.2))
                                     
        self.decoder = nn.Sequential(DecoderBlock(self.latent_dim, 512, 0.2),
                                     DecoderBlock(512, self.feature_num, 0))
        
        self.embedder = nn.Sequential(EncoderBlock(self.latent_dim, 64, 0),
                                      EncoderBlock(64, 32, 0.2))

        self.predictor = nn.Sequential(EncoderBlock(32, 16, 0.2),
                                       nn.Linear(16, self.celltype_num),
                                       nn.Softmax(dim=1))
        
        self.discriminator = nn.Sequential(EncoderBlock(32, 16, 0.2),
                                           nn.Linear(16, 1),
                                           nn.Sigmoid()) 

    
    def forward(self, x, alpha=1.0):  # NOTE: x: (batch_size, feature_num)
        batch_size = x.size(0)

        # 1. Encoder
        emb = self.encoder(x).view(batch_size, -1)

        # 2. Decoder
        rec = self.decoder(emb)

        # embedder
        emb2 = self.embedder(emb).view(batch_size, -1)
        
        # 3. Predictor
        pred = self.predictor(emb2)

        # 4. Domain classifier with GRL
        domain_emb = GradientReversalLayer.apply(emb2, alpha)
        domain = self.discriminator(domain_emb)

        return rec, pred, domain


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

    
    def prepare_dataloader(self, source_data, target_data, batch_size, target_cells=[]):
        ### Prepare data loader for training ###
        g = torch.Generator()
        g.manual_seed(42)

        # Source dataset
        if len(target_cells) == 0:
            source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
            self.labels = source_data.uns['cell_types']
        else:
            source_ratios = [source_data.obs[ctype] for ctype in target_cells]
            self.labels = target_cells
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
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
