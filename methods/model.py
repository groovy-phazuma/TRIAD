# -*- coding: utf-8 -*-
"""
Created on 2025-06-12 (Thu) 23:46:39

References
- https://github.com/poseidonchan/TAPE/blob/main/Experiments/pytorch_scaden_PBMConly.ipynb
- https://github.com/poseidonchan/TAPE/blob/main/TAPE/model.py

@author: I.Azuma
"""
import random
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from anndata import read_h5ad
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'
sys.path.append(BASE_DIR+'/github/TRIAD')
from _utils.dataset import *


class simdatset(Data.Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rates):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._mlp()

    def forward(self,x):
        # x: (n sample, m gene)
        # output: (n sample, k cell proportions)
        return self.model(x)

    def _mlp(self):
        mlp = nn.Sequential(nn.Linear(self.input_dim,self.hidden_units[0]),
                            nn.Dropout(self.dropout_rates[0]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
                            nn.Dropout(self.dropout_rates[1]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
                            nn.Dropout(self.dropout_rates[2]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units[2], self.hidden_units[3]),
                            nn.Dropout(self.dropout_rates[3]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units[3], self.output_dim),
                            nn.Softmax(dim=1))
        return mlp


def initialize_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data,0)
        
class scaden():
    def __init__(self, cfg, architectures, traindata, seed=42):
        self.cfg = cfg
        self.architectures = architectures
        self.model512 = None
        self.model256 = None
        self.model1024 = None
        self.testmodel = None
        self.lr = cfg.scaden.learning_rate
        self.batch_size = cfg.scaden.batch_size
        self.epochs = cfg.scaden.epochs
        self.inputdim = None
        self.outputdim = None
        self.seed = seed

        self.target_cells = cfg.common.target_cells

        self.set_data()
        self.build_dataloader(batch_size=self.batch_size)
    
    def set_data(self):
        train_data, test_data, train_y, test_y, gene_names = prep4benchmark(h5ad_path=self.cfg.paths.h5ad_path,
                                                                            source_list=self.cfg.common.source_domain,
                                                                            target=self.cfg.common.target_domain,
                                                                            target_cells=self.target_cells,
                                                                            n_samples=self.cfg.common.n_samples, 
                                                                            n_vtop=self.cfg.common.n_vtop,
                                                                            seed=self.seed)
        self.source_data = train_data
        self.target_data = test_data
        self.target_y = test_y

    def _loss_func(self, pred, target):
        l1loss = nn.L1Loss()
        kldloss = nn.KLDivLoss()
        l2loss = nn.MSELoss()
        return l1loss(pred, target)  # +l2loss(pred,target)#+kldloss(pred,target)

    def _subtrain(self, model, optimizer):
        model.train()
        i = 0
        loss = []
        best_loss = 1e10
        for i in tqdm(range(self.epochs)):
            loss_epoch = 0
            for data, label in self.train_source_loader:
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                batch_loss = self._loss_func(model(data), label)
                batch_loss.backward()
                optimizer.step()
                loss_epoch += batch_loss.item()
            
            loss.append(loss_epoch)

            if i > self.cfg.scaden.early_stop:
                if loss_epoch < best_loss:
                    best_loss = loss_epoch
                    self.update_flag = 0
                    #torch.save(model.state_dict(), './model.pth')
                else:
                    self.update_flag += 1
                    if self.update_flag == self.cfg.scaden.early_stop:
                        print(f"Early stopping at epoch {i+1}")
                        return model, loss

        return model, loss

    def score(self, model1, model2=None, model3=None, mode='val'):
        model1.eval()
        if model2 is not None and model3 is not None:
            model2.eval()
            model3.eval()
        CCC = 0
        RMSE = 0

        if mode == 'val':
            loader = self.val_source_loader  # NOTE: not implemented yet
        elif mode == 'test':
            loader = self.test_target_loader

        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            if model2 is not None and model3 is not None:
                pred = (model1(data) + model2(data) + model3(data)) / 3
            else:
                pred = model1(data)
            new_pred = pred.reshape(pred.shape[0]*pred.shape[1],1)
            new_label = label.reshape(label.shape[0]*label.shape[1],1)
            RMSE += self.RMSE(new_pred.cpu().detach().numpy(), new_label.cpu().detach().numpy())
            CCC += self.CCCscore(new_pred.cpu().detach().numpy(), new_label.cpu().detach().numpy())
        return CCC / len(loader), RMSE / len(loader)
    
    def RMSE(self, pred, true):
        return np.sqrt(np.mean(np.square(pred - true)))
    
    def CCC(self,pred,true):
        r = np.corrcoef(pred, true)[0, 1]
        mean_true = np.mean(true)
        mean_pred = np.mean(pred)
        # Variance
        var_true = np.var(true)
        var_pred = np.var(pred)
        # Standard deviation
        sd_true = np.std(true)
        sd_pred = np.std(pred)
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        return ccc
    
    def CCCscore(self, y_pred, y_true):
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

    def showloss(self, loss):
        plt.figure()
        plt.plot(loss)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        #plt.title()
        #plt.savefig(str(z)+'_'+str(beta)+'_'+str(gamma)+'.png')
        plt.show()
        
    def train(self, mode='all'):
        if mode=='all':
            ##### train
            self.build_model()
            optimizer = torch.optim.Adam(self.model256.parameters(), lr=self.lr, eps=1e-07)
            print('train model256 now')
            self.model256, loss = self._subtrain(self.model256, optimizer)
            self.showloss(loss)
            print(self.score(self.model256, mode='test'))
            optimizer = torch.optim.Adam(self.model512.parameters(), lr=self.lr, eps=1e-07)
            print('train model512 now')
            self.model512, loss = self._subtrain(self.model512, optimizer)
            self.showloss(loss)
            print(self.score(self.model512, mode='test'))
            optimizer = torch.optim.Adam(self.model1024.parameters(), lr=self.lr, eps=1e-07)
            print('train model1024 now')
            self.model1024, loss = self._subtrain(self.model1024, optimizer)
            self.showloss(loss)
            print(self.score(self.model1024, mode='test'))
            ##### evalutaion on val_set
            #print(self.score(self.model256, self.model512, self.model1024, mode='val'))
            ##### evaluation on test_set
            print(self.score(self.model256, self.model512, self.model1024, mode='test'))
        elif mode=='single':
            self.build_model(mode=mode)
            optimizer = torch.optim.Adam(self.testmodel.parameters(), lr=self.lr)
            print('train test model now')
            self.testmodel, loss = self._subtrain(self.testmodel,optimizer)
            self.showloss(loss)
            ##### evaluation on test_set
            print(self.score(self.testmodel, mode='test'))


    def build_dataloader(self, batch_size):
        ### Prepare data loader for training ###
        g = torch.Generator()
        g.manual_seed(self.seed)

        source_data = self.source_data
        target_data = self.target_data

        # 1. Source dataset
        if self.target_cells is None:
            source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
            self.target_cells = source_data.uns['cell_types']
        else:
            source_ratios = [source_data.obs[ctype] for ctype in self.target_cells]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        self.inputdim = tr_data.shape[1]
        self.outputdim = tr_labels.shape[1]


        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
        self.celltype_num = len(self.target_cells)
        self.used_features = list(source_data.var_names)

        # 2. Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)

        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    def build_model(self,mode='all'):
        if mode=='all':
            self.model256 = MLP(self.inputdim, self.outputdim, self.architectures['m256'][0], self.architectures['m256'][1])
            self.model512 = MLP(self.inputdim, self.outputdim, self.architectures['m512'][0], self.architectures['m512'][1])
            self.model1024 = MLP(self.inputdim, self.outputdim, self.architectures['m1024'][0],self.architectures['m1024'][1])
            self.model1024 = self.model1024.to(device)
            self.model512 = self.model512.to(device)
            self.model256 = self.model256.to(device)
            self.model256.apply(initialize_weight)
            self.model512.apply(initialize_weight)
            self.model1024.apply(initialize_weight)
        elif mode=='single':
            self.testmodel = MLP(self.inputdim, self.outputdim, self.architectures['m512'][0], self.architectures['m512'][1])
            self.testmodel = self.testmodel.to(device)
            
    def predict(self,mode='all'):
        if mode == 'all':
            self.model256.eval()
            self.model512.eval()
            self.model1024.eval()
        elif mode == 'single':
            self.testmodel.eval()
        for data, label in self.test_target_loader:
            data = data.to(device)
            label = label.to(device)
            if mode == 'all':
                pred = (self.model256(data) + self.model512(data) + self.model1024(data)) / 3
            elif mode == 'single':
                pred = self.testmodel(data)
        return pred.cpu().detach().numpy()

    def save(self):
        save_path = self.cfg.paths.save_path
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model256.state_dict(), os.path.join(save_path, 'model256.pth'))
        torch.save(self.model512.state_dict(), os.path.join(save_path, 'model512.pth'))
        torch.save(self.model1024.state_dict(), os.path.join(save_path, 'model1024.pth'))


    def load(self, model256, model512, model1024):
        self.build_model()
        self.model256.load_state_dict(torch.load(model256))
        self.model512.load_state_dict(torch.load(model512))
        self.model1024.load_state_dict(torch.load(model1024))


