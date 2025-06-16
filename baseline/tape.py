# -*- coding: utf-8 -*-
"""
Created on 2025-06-16 (Mon) 21:29:16

References
- https://github.com/poseidonchan/TAPE/blob/main/Experiments/TAPE_realbulk.ipynb

@author: I.Azuma
"""
# %%
import random
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from anndata import read_h5ad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'
sys.path.append(BASE_DIR+'/github/TRIAD')
from _utils.dataset import *

class simdatset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y


def prep4tape(h5ad_path, target='sdy67', test_ratio=0.2, target_path=None):
    pbmc = read_h5ad(h5ad_path)
    pbmc1 = pbmc[pbmc.obs['ds']=='sdy67']
    microarray = pbmc[pbmc.obs['ds']=='GSE65133']
    
    donorA = pbmc[pbmc.obs['ds']=='donorA']
    donorC = pbmc[pbmc.obs['ds']=='donorC']
    data6k = pbmc[pbmc.obs['ds']=='data6k']
    data8k = pbmc[pbmc.obs['ds']=='data8k']
    
    if target == 'sdy67':
        test = pbmc1
        train = anndata.concat([donorA,donorC,data6k,data8k])
    elif target == 'GSE65133':
        test = microarray
        train = anndata.concat([donorA,donorC,data6k,data8k,pbmc1])
    else:
        if target_path is None:
            raise ValueError("Please provide target_path for inference mode.")
        print(f"Target domain: {target}")
        target_df = pd.read_csv(target_path, index_col=0)

        # Match gene names
        target_df.index = target_df.index.str.upper()
        target_genes = target_df.index
        source_genes = pbmc.var_names.str.upper()
        common_genes = target_genes.intersection(source_genes)
        target_df_filtered = target_df.loc[common_genes]

        target_adata = AnnData(X=target_df_filtered.T.values)
        target_adata.var_names = target_df_filtered.index
        target_adata.obs_names = target_df_filtered.columns
        target_adata.obs['ds'] = target

        print("Target data shape: ", target_adata.X.shape)
        if len(target_adata.obs_names) == 0:
            raise ValueError("No cells found in target data. Please check the input data.")

        # add obs columns from source_pbmc to target_adata
        for col in pbmc.obs.columns:
            if col not in target_adata.obs:
                target_adata.obs[col] = target if col == "ds" else (2 if col == "batch" else np.nan)
        target_adata.obs = target_adata.obs[pbmc.obs.columns]

        # train and test definition
        gene_map = dict(zip(pbmc.var_names.str.upper(), pbmc.var_names)) 
        ordered_genes_in_pbmc = [gene_map[gene] for gene in target_df_filtered.index if gene in gene_map]
        source_domains = ['donorA', 'donorC', 'data6k', 'data8k']
        train = pbmc[pbmc.obs['ds'].isin(source_domains), ordered_genes_in_pbmc]
        test = target_adata
        
    train_y = train.obs.iloc[:,:-2].values
    test_y = test.obs.iloc[:,:-2].values

    print(train.obs.head())
    print(test.obs.head())

    #### variance cut off
    if target == 'sdy67':
        label = test.X.var(axis=0) > 0.1  # NOTE: not train
    elif target == 'GSE65133':
        label = test.X.var(axis=0) > 0.01
    else:
        label = test.X.var(axis=0) > 0.1
    test_x = np.zeros((test.X.shape[0],np.sum(label)))
    train_x = np.zeros((train.X.shape[0],np.sum(label)))

    test_x = test.X[:,label]
    train_x = train.X[:,label]
    
    train_x = np.log2(train_x + 1)
    test_x = np.log2(test_x + 1)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=test_ratio)

    print("Start scaling data...")
    mms = MinMaxScaler()
    test_x = mms.fit_transform(test_x.T)
    test_x = test_x.T
    train_x = mms.fit_transform(train_x.T)
    train_x = train_x.T
    val_x = mms.fit_transform(val_x.T)
    val_x = val_x.T


    print(f"Train data shape: {train_x.shape}")
    print(f"Test data shape: {test_x.shape}")

    return train_x, val_x, test_x, train_y, val_y, test_y
    
class AutoEncoder(nn.Module):
    def __init__(self, cfg, seed=42):
        super().__init__()
        self.name = 'ae'
        self.state = 'train' # or 'test'
        self.cfg = cfg
        self.lr  = cfg.tape.learning_rate
        self.batch_size = cfg.tape.batch_size
        self.iterations= cfg.tape.max_iter
        self.inputdim = None
        self.outputdim = None
        self.seed = seed
        set_random_seed(self.seed)
    
        self.encoder = nn.Sequential(nn.Dropout(),
                                     nn.Linear(self.inputdim, 512),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 256),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(256, 128),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),
                                     nn.Linear(64, self.outputdim),
                                     )

        self.decoder = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.inputdim, bias=False))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self,x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x/x_sum

    
    def sigmatrix(self):
        w0 = self.decoder[0].weight.T
        w1 = self.decoder[1].weight.T
        w2 = self.decoder[2].weight.T
        w3 = self.decoder[3].weight.T
        w4 = self.decoder[4].weight.T
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = F.hardtanh(torch.mm(w03, w4),0,1)
        return w04

    def forward(self, x):
        sigmatrix = self.sigmatrix()
        z = self.encode(x)
        #z = self.own_softmax(z)
        if self.state == 'train':
            z = F.hardtanh(z,0,1)

        elif self.state == 'test':
            
            z = F.hardtanh(z,0,1)
            z = self.refraction(z)
        x_recon = torch.mm(z, sigmatrix)
        return x_recon, z, sigmatrix
    

class tape():
    def __init__(self, cfg, seed=42):
        self.cfg = cfg
        self.lr  = cfg.tape.learning_rate
        self.batch_size = cfg.tape.batch_size
        self.iterations= cfg.tape.max_iter
        self.inputdim = None
        self.outputdim = None
        self.seed = seed
        set_random_seed(self.seed)

    def set4benchmark(self):
        train_x, val_x, test_x, train_y, val_y, test_y = prep4tape(h5ad_path=self.cfg.paths.h5ad_path, 
                                                                     target=self.cfg.common.target_domain,
                                                                     test_ratio=self.cfg.tape.test_ratio,
                                                                     target_path=self.cfg.paths.target_path)
        
        self.train_source_loader = Data.DataLoader(simdatset(train_x, train_y), batch_size=self.batch_size, shuffle=True)
        self.val_source_loader = Data.DataLoader(simdatset(val_x, val_y), batch_size=self.batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(simdatset(test_x, test_y), batch_size=len(test_x), shuffle=False)

        self.inputdim = self.train_source_loader.dataset.X.shape[1]
        self.outputdim = self.train_source_loader.dataset.Y.shape[1]

    def train(self):
        self.model = AutoEncoder(cfg=self.cfg).to(device)

        # Train
        optimizer = Adam(self.model.parameters(), lr=self.cfg.tape.learning_rate)
        self.model, loss, reconloss = training_stage(self.model, train_loader, optimizer, epochs=int(self.iterations /(len(train_x)/self.batch_size)), device=device, verbose=True)
        print('prediction loss is:')
        showloss(loss)
        print('reconstruction loss is:')
        showloss(reconloss)
    
    def predict(self):
        self.model = train_model(train_x, train_y, seed=seed)
        self.model.eval()
        self.model.state = 'test'
        data = torch.from_numpy(test_x).float().to(device)
        _, pred, _ = model(data)
        pred = pred.cpu().detach().numpy()

        return pred.cpu().detach().numpy()

def L1error(pred, true):
    return np.mean(np.abs(pred - true))


def CCCscore(y_pred, y_true):
    # pred: shape{n sample, m cell}
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # print(r)
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        # print(ccc)
        ccc_value += ccc
    return ccc_value / y_pred.shape[1]

def score(pred, label):
    new_pred = pred.reshape(pred.shape[0]*pred.shape[1],1)
    new_label = label.reshape(label.shape[0]*label.shape[1],1)
    distance = L1error(new_pred, new_label)
    ccc = CCCscore(new_pred, new_label)
    return distance, ccc
    
def showloss(loss):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()           


def training_stage(model, train_loader, optimizer, epochs=10):
    model.train()
    model.state = 'train'
    loss = []
    recon_loss = []
    for i in tqdm(range(epochs)):
        for k, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)
            batch_loss = F.l1_loss(cell_prop, label) + F.l1_loss(x_recon, data)
            batch_loss.backward()
            optimizer.step()
            loss.append(F.l1_loss(cell_prop, label).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

    return model, loss, recon_loss


        
def test(train_x, train_y, test_x, test_y, seed=0):
    reproducibility(seed)
    model = train_model(train_x, train_y, seed=seed)
    model.eval()
    model.state = 'test'
    data = torch.from_numpy(test_x).float().to(device)
    _, pred, _ = model(data)
    pred = pred.cpu().detach().numpy()
    a,b = score(pred,test_y)
    print(a,b)
    return a, b


