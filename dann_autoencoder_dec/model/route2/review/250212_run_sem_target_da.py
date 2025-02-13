# -*- coding: utf-8 -*-
"""
Created on 2025-02-12 (Wed) 10:55:55

Add domain adaptation to the model.

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'

import anndata
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(BASE_DIR+'/github/GSTMDec/dann_autoencoder_dec')
import argparse
import options
from model.route2.sem_target_da import *

from tqdm import tqdm
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% preprocessing in TAPE
def preprocess(trainingdatapath, source='data6k', target='sdy67', n_samples=None):
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

    #### top 1000 highly variable genes
    label = np.argsort(-train.X.var(axis=0))[:1000]
    
    train_data = train[:, label]
    train_data.X = np.log2(train_data.X + 1)
    test_data = test[:, label]
    test_data.X = np.log2(test_data.X + 1)

    return train_data, test_data, train_y, test_y

#  Load dataset
batch_size = 128
trainingdatapath = BASE_DIR+"/datasource/scRNASeq/Scaden/pbmc_data.h5ad"
train_data, test_data, train_y, test_y = preprocess(trainingdatapath,source='data6k',target='sdy67', n_samples=1024)
train_source_loader, train_target_loader, test_target_loader, labels, used_features = prepare_dataloader(train_data, test_data, batch_size=batch_size)


# %% Parameters
num_genes = train_data.shape[1]
K = 1  # 'Number of Gaussian kernel in GMM, default =1'
K1 = 1
K2 = 2
n_hidden = 128
lr = 5e-3
n_epochs = 200
alpha = 1 # sparsity
beta = 1  # KL
gamma = 100 # prediction
da_weight = 1

# %% Train
model = DANN_SEM_AD(x_dim=1, y_dim=n_hidden, z_dim=K, n_celltype=6, n_gene=num_genes, pred_loss_type='custom').to(device)

optimizer = optim.Adam([{'params': model.inference_t.parameters()},
                        {'params': model.generative_t.parameters()},
                        {'params': model.predictor.parameters()}
                        ], lr=lr)


#optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer_adj = optim.Adam([model.adj_A], lr=lr*0.2)
"""
optimizer_da1 = optim.Adam([{'params': model.inference_t.parameters()},
                            {'params': model.discriminator.parameters()},
                            {'params': model.predictor.parameters()}], lr=lr*0.2)
optimizer_da2 = optim.Adam([{'params': model.inference_t.parameters()},
                            {'params': model.discriminator.parameters()}], lr=lr*0.2)
"""


scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adj, step_size=1, gamma=0.95)
"""
scheduler_da1 = torch.optim.lr_scheduler.StepLR(optimizer_da1, step_size=10, gamma=0.9)
scheduler_da2 = torch.optim.lr_scheduler.StepLR(optimizer_da2, step_size=10, gamma=0.9)
"""

criterion_da = nn.BCELoss().cuda()
source_label = torch.ones(batch_size).unsqueeze(1).cuda()   # source domain: 1
target_label = torch.zeros(batch_size).unsqueeze(1).cuda()  # target domain: 0

# Training phase
metric_logger = defaultdict(list) 
best_loss = 1e10
for epoch in tqdm(range(n_epochs+1)):
    loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse, loss_deconv, loss_disc, loss_disc_da = [], [], [], [], [], [], [], [], []
    if epoch % (K1+K2) < K1:
        model.adj_A.requires_grad = False
    else:
        model.adj_A.requires_grad = True

    
    model.train()
    for batch_idx, (source_x, source_y) in enumerate(train_source_loader):
        target_x = next(iter(test_target_loader))[0]  # NOTE: without shuffle
        #target_x = next(iter(train_target_loader))[0]   # NOTE: with shuffle

        source_x = Variable(source_x.to(device))
        source_y = Variable(source_y.to(device))
        target_x = Variable(target_x.to(device))

        temperature = max(0.95 ** epoch, 0.5)
        loss_dict, output = model(source_x=source_x, target_x=target_x, source_y=source_y, dropout_mask=None, temperature=temperature)

        """
        # domain discriminator 1 (s<-->s, t<-->t)
        domain_source = output['domain_source']
        domain_target = output['domain_target']
        
        disc_loss1 = criterion_da(domain_source, source_label[0:domain_source.shape[0],])\
             + criterion_da(domain_target, target_label[0:domain_target.shape[0],])"""

        loss_rec = 0.1*loss_dict['loss_rec']
        sparse_loss = alpha * torch.mean(torch.abs(model.adj_A))
        loss_gauss = beta*loss_dict['loss_gauss']
        loss_cat = beta*loss_dict['loss_cat']
        loss_pred = gamma*loss_dict['loss_pred']
        #disc_loss1 = da_weight*disc_loss1

        # update weights 1
        optimizer.zero_grad()
        optimizer_adj.zero_grad()

        loss = loss_rec + loss_gauss + loss_cat + sparse_loss + loss_pred  # NOTE + loss_pred is important
        #loss.backward(retain_graph=True)
        loss.backward()

        if epoch % (K1+K2) < K1:
            optimizer.step()
        else:
            optimizer_adj.step()
        """
        # update weights 2
        optimizer_da1.zero_grad()
        loss2 = loss_pred + disc_loss1
        loss2.backward(retain_graph=True)
        if epoch % (K1+K2) < K1:
            optimizer_da1.step()
        
        # domain discriminator 2 (s<-->t, t<-->s)
        domain_source = output['domain_source']
        domain_target = output['domain_target']
        disc_loss2 = criterion_da(domain_source, target_label[0:domain_source.shape[0],])\
             + criterion_da(domain_target, source_label[0:domain_target.shape[0],])
        disc_loss2 = da_weight*disc_loss2

        # update weights 3
        optimizer_da2.zero_grad()
        disc_loss2.backward(retain_graph=True)
        if epoch % (K1+K2) < K1:
            optimizer_da2.step()
        """
        
        loss_all.append(loss.item()) #+ disc_loss1.item() + disc_loss2.item())
        mse_rec.append(loss_rec.item())
        loss_kl.append(loss_gauss.item() + loss_cat.item())
        loss_deconv.append(loss_pred.item())
        loss_sparse.append(sparse_loss.item())
        #loss_disc.append(disc_loss1.item())
        #loss_disc_da.append(disc_loss2.item())
    
    if epoch % 10 == 0:
        #print(f"Epoch: {epoch}, Loss: {np.mean(loss_all):.4f}, rec: {np.mean(mse_rec):.4f}, kl: {np.mean(loss_kl):.4f}, sparse: {np.mean(loss_sparse):.4f}, pred: {np.mean(loss_deconv):.4f}, disc: {np.mean(loss_disc):.4f}, disc_da: {np.mean(loss_disc_da):.4f}")
        print(f"Epoch: {epoch}, Loss: {np.mean(loss_all):.4f}, rec: {np.mean(mse_rec):.4f}, kl: {np.mean(loss_kl):.4f}, sparse: {np.mean(loss_sparse):.4f}, pred: {np.mean(loss_deconv):.4f}")

        # save best model
        if np.mean(loss_deconv) < best_loss:
            best_loss = np.mean(loss_deconv)
            metric_logger['best_epoch'] = epoch
            torch.save(model.state_dict(), BASE_DIR+'/workspace/240816_model_trial/250115_scpDeconv/250204_dann_sem_dev/results/250212_sem_da/best_model.pth')
            print("Save model at epoch %d" % (epoch))

    scheduler.step()
    #scheduler_da1.step()
    #scheduler_da2.step()

    metric_logger['loss_all'].append(np.mean(loss_all))
    metric_logger['loss_rec'].append(np.mean(mse_rec))
    metric_logger['loss_kl'].append(np.mean(loss_kl))
    metric_logger['loss_sparse'].append(np.mean(loss_sparse))
    metric_logger['loss_pred'].append(np.mean(loss_deconv))
    metric_logger['loss_disc'].append(np.mean(loss_disc))
    metric_logger['loss_disc_da'].append(np.mean(loss_disc_da))

pd.to_pickle(metric_logger, BASE_DIR+'/workspace/240816_model_trial/250115_scpDeconv/250204_dann_sem_dev/results/250212_sem_da/metric_logger.pkl')

