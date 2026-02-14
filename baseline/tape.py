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
from baseline.baseline_utils import prep4pbmc, prep4tissue

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
    
class AutoEncoder(nn.Module):
    def __init__(self, cfg, inputdim, outputdim, seed=42):
        super().__init__()
        self.name = 'ae'
        self.state = 'train' # or 'test'
        self.cfg = cfg
        self.lr  = cfg.tape.learning_rate
        self.batch_size = cfg.tape.batch_size
        self.inputdim = inputdim
        self.outputdim = outputdim
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
        self.max_iter = cfg.tape.max_iter
        self.epochs = cfg.tape.epochs
        self.inputdim = None
        self.outputdim = None
        self.seed = seed
        set_random_seed(self.seed)

    def set4benchmark(self):
        train_x, val_x, test_x, train_y, val_y, test_y = prep4pbmc(h5ad_path=self.cfg.paths.h5ad_path, 
                                                                   target=self.cfg.common.target_domain,
                                                                   test_ratio=self.cfg.tape.test_ratio,
                                                                   target_path=self.cfg.paths.target_path)
        
        self.train_source_loader = Data.DataLoader(simdatset(train_x, train_y), batch_size=self.batch_size, shuffle=True)
        self.val_source_loader = Data.DataLoader(simdatset(val_x, val_y), batch_size=self.batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(simdatset(test_x, test_y), batch_size=len(test_x), shuffle=False)

        self.inputdim = self.train_source_loader.dataset.X.shape[1]
        self.outputdim = self.train_source_loader.dataset.Y.shape[1]
    
    def set4tissue_application(self):
        train_x, val_x, test_x, train_y, val_y, test_y = prep4tissue(h5ad_path=self.cfg.paths.h5ad_path, 
                                                                     target=self.cfg.common.target_domain,
                                                                     test_ratio=self.cfg.tape.test_ratio,
                                                                     target_path=self.cfg.paths.target_path)
        
        self.train_source_loader = Data.DataLoader(simdatset(train_x, train_y), batch_size=self.batch_size, shuffle=True)
        self.val_source_loader = Data.DataLoader(simdatset(val_x, val_y), batch_size=self.batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(simdatset(test_x, test_y), batch_size=len(test_x), shuffle=False)

        self.inputdim = self.train_source_loader.dataset.X.shape[1]
        self.outputdim = self.train_source_loader.dataset.Y.shape[1]

    def training_stage(self, model, optimizer):
        model.train()
        model.state = 'train'
        loss = []
        recon_loss = []
        best_loss = 1e10
        num_iter = 0
        for i in tqdm(range(self.epochs)):
            loss_epoch = 0
            for k, (data, label) in enumerate(self.train_source_loader):
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                x_recon, cell_prop, sigm = model(data)
                batch_loss = F.l1_loss(cell_prop, label) + F.l1_loss(x_recon, data)
                batch_loss.backward()
                optimizer.step()
                recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())
                l1_loss = F.l1_loss(cell_prop, label).cpu().detach().numpy()
                loss.append(l1_loss)
                loss_epoch += l1_loss

                num_iter += 1
                if num_iter == self.max_iter:
                    print(f"Maximum iterations {self.max_iter} reached")
                    return model, loss, recon_loss

            if i > self.cfg.tape.early_stop:
                if loss_epoch < best_loss:
                    best_loss = loss_epoch
                    self.update_flag = 0
                    #torch.save(model.state_dict(), './model.pth')
                else:
                    self.update_flag += 1
                    if self.update_flag == self.cfg.tape.early_stop:
                        print(f"Early stopping at epoch {i+1}")
                        return model, loss, recon_loss

        return model, loss, recon_loss

    def train(self):
        self.model = AutoEncoder(cfg=self.cfg, inputdim=self.inputdim, outputdim=self.outputdim).to(device)

        # Train
        optimizer = Adam(self.model.parameters(), lr=self.cfg.tape.learning_rate)
        self.model, loss, reconloss = self.training_stage(self.model, optimizer)
        print('prediction loss is:')
        showloss(loss)
        print('reconstruction loss is:')
        showloss(reconloss)
    
    def save(self):
        save_path = self.cfg.paths.save_path
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, f'tape_{self.seed}.pth'))
    
    def load_model(self):
        save_path = self.cfg.paths.save_path
        model_path = os.path.join(save_path, f'tape_{self.seed}.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = AutoEncoder(cfg=self.cfg, inputdim=self.inputdim, outputdim=self.outputdim).to(device)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self):
        self.model.eval()
        self.model.state = 'test'
        test_x, test_y = self.test_target_loader.dataset.X, self.test_target_loader.dataset.Y
        data = torch.from_numpy(test_x).float().to(device)
        _, pred, _ = self.model(data)
        pred = pred.cpu().detach().numpy()

        a,b = score(pred, test_y)
        print(f"Prediction L1 error: {a}, CCC score: {b}")

        return pred

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
