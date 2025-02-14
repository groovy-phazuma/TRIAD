
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
da_weight = 0

# %% Train
da_option = False
model = DANN_SEM_AD(x_dim=1, y_dim=n_hidden, z_dim=K, n_celltype=6, n_gene=num_genes, pred_loss_type='custom',da_option=da_option).to(device)

optimizer = optim.Adam([{'params': model.inference_t.parameters()},
                        {'params': model.generative_t.parameters()},
                        {'params': model.predictor.parameters()}
                        ], lr=lr)

optimizer_adj = optim.Adam([model.adj_A], lr=lr*0.2)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adj, step_size=1, gamma=0.95)
if da_option:
    optimizer_da1 = optim.Adam([{'params': model.inference_t.parameters()},
                                {'params': model.discriminator.parameters()},  # NOTE: add
                                {'params': model.generative_t.parameters()},
                                {'params': model.predictor.parameters()}], lr=lr)
    optimizer_da2 = optim.Adam([{'params': model.inference_t.parameters()},
                                {'params': model.generative_t.parameters()},  # NOTE: add
                                {'params': model.discriminator.parameters()}], lr=lr)
    scheduler_da1 = torch.optim.lr_scheduler.StepLR(optimizer_da1, step_size=10, gamma=0.9)
    scheduler_da2 = torch.optim.lr_scheduler.StepLR(optimizer_da2, step_size=10, gamma=0.9)


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


        # domain discriminator 1 (s<-->s, t<-->t)
        if da_option:
            domain_source = output['domain_source']
            domain_target = output['domain_target']
            
            disc_loss1 = criterion_da(domain_source, source_label[0:domain_source.shape[0],])\
                + criterion_da(domain_target, target_label[0:domain_target.shape[0],])
            disc_loss1 = da_weight*disc_loss1

        loss_rec = 0.1*loss_dict['loss_rec']
        sparse_loss = alpha * torch.mean(torch.abs(model.adj_A))
        loss_gauss = beta*loss_dict['loss_gauss']
        loss_cat = beta*loss_dict['loss_cat']
        loss_pred = gamma*loss_dict['loss_pred']

        # update weights 1
        optimizer.zero_grad()
        optimizer_adj.zero_grad()

        loss = loss_rec + loss_gauss + loss_cat + sparse_loss + loss_pred
        loss.backward(retain_graph=True)

        if epoch % (K1+K2) < K1:

            optimizer.step()
        else:
            optimizer_adj.step()

          
        if da_option:
            # update weights 2
            optimizer_da1.zero_grad()
            loss2 = loss_rec + loss_pred + disc_loss1  # FIXME add loss_rec
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
            loss3 = loss_rec + disc_loss2  # FIXME add loss_rec
            model.predictor.requires_grad = False  # FIXME is it correct ?
            loss3.backward(retain_graph=True)
            if epoch % (K1+K2) < K1:
                optimizer_da2.step()

        
            loss_all.append(loss.item() + disc_loss1.item() + disc_loss2.item())
            loss_disc.append(disc_loss1.item())
            loss_disc_da.append(disc_loss2.item())
        else:
            loss_all.append(loss.item())

        mse_rec.append(loss_rec.item())
        loss_kl.append(loss_gauss.item() + loss_cat.item())
        loss_deconv.append(loss_pred.item())
        loss_sparse.append(sparse_loss.item())
        
    
    if epoch % 10 == 0:
        if da_option:
            print(f"Epoch:{epoch}, Loss:{np.mean(loss_all):.3f}, rec:{np.mean(mse_rec):.3f}, kl:{np.mean(loss_kl):.3f}, sparse:{np.mean(loss_sparse):.3f}, pred:{np.mean(loss_deconv):.3f}, disc:{np.mean(loss_disc):.3f}, disc_da:{np.mean(loss_disc_da):.3f}")
            metric_logger['loss_disc'].append(np.mean(loss_disc))
            metric_logger['loss_disc_da'].append(np.mean(loss_disc_da))
        else:
            print(f"Epoch: {epoch}, Loss: {np.mean(loss_all):.4f}, rec: {np.mean(mse_rec):.4f}, kl: {np.mean(loss_kl):.4f}, sparse: {np.mean(loss_sparse):.4f}, pred: {np.mean(loss_deconv):.4f}")

        # save best model
        if np.mean(loss_deconv) < best_loss:
            best_loss = np.mean(loss_deconv)
            metric_logger['best_epoch'] = epoch
            torch.save(model.state_dict(), BASE_DIR+'/workspace/240816_model_trial/250115_scpDeconv/250204_dann_sem_dev/results/250212_sem_da/best_model.pth')
            print("Save model at epoch %d" % (epoch))

    scheduler.step()
    if da_option:
        scheduler_da1.step()
        scheduler_da2.step()

    metric_logger['loss_all'].append(np.mean(loss_all))
    metric_logger['loss_rec'].append(np.mean(mse_rec))
    metric_logger['loss_kl'].append(np.mean(loss_kl))
    metric_logger['loss_sparse'].append(np.mean(loss_sparse))
    metric_logger['loss_pred'].append(np.mean(loss_deconv))

pd.to_pickle(metric_logger, BASE_DIR+'/workspace/240816_model_trial/250115_scpDeconv/250204_dann_sem_dev/results/250212_sem_da/metric_logger.pkl')

# %%
metric_logger = pd.read_pickle(BASE_DIR+'/workspace/240816_model_trial/250115_scpDeconv/250204_dann_sem_dev/results/250212_sem_da/metric_logger.pkl')
# summarize loss history
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
loss_type = ['loss_all', 'loss_rec', 'loss_kl', 'loss_sparse', 'loss_pred', 'loss_disc', 'loss_disc_da']
for i in range(len(loss_type)):
    axes[i//4, i%4].plot(metric_logger[loss_type[i]])
    axes[i//4, i%4].set_title(loss_type[i], x=0.5, y=0.5)
    axes[i//4, i%4].set_xlabel('Epoch')
    axes[i//4, i%4].set_ylabel('Loss')
    axes[i//4, i%4].grid(True)
plt.tight_layout()
plt.show()

# %%
source_x = Variable(torch.tensor(train_data.X).to(device))
target_x = Variable(torch.tensor(test_data.X).to(device))

ens_pred_s = []
ens_pred_t = []
ens_adj_t = []
for ens_i in tqdm(range(100)):
    model = DANN_SEM_AD(x_dim=1, y_dim=n_hidden, z_dim=K, n_celltype=6, n_gene=num_genes, pred_loss_type='custom',seed=ens_i,da_option=da_option).to(device)
    model.load_state_dict(torch.load(BASE_DIR+'/workspace/240816_model_trial/250115_scpDeconv/250204_dann_sem_dev/results/250212_sem_da/best_model.pth'))

    model.eval()
    source_y = Variable(torch.tensor(train_y.values).to(device))
    target_y = Variable(torch.tensor(test_y.values).to(device))

    _, output = model(source_x, target_x, source_y, dropout_mask=None,temperature=1.0)

    pred_s = output['source_pred'].detach().cpu().numpy()
    pred_t = output['target_pred'].detach().cpu().numpy()
    pred_s_df = pd.DataFrame(pred_s, columns=train_data.uns['cell_types'])
    pred_t_df = pd.DataFrame(pred_t, columns=test_data.uns['cell_types'])
    s_y_df = train_data.obs[train_data.uns['cell_types']]
    t_y_df = test_data.obs[test_data.uns['cell_types']]

    # ensemble
    ens_pred_s.append(pred_s)
    ens_pred_t.append(pred_t)

    adj_A = model.adj_A.detach().cpu().numpy()
    ens_adj_t.append(adj_A)


ens_adj_t = np.mean(ens_adj_t, axis=0)
ens_s_summary = np.mean(ens_pred_s, axis=0)
ens_s_summary_df = pd.DataFrame(ens_s_summary, columns=train_data.uns['cell_types'])
ens_t_summary = np.mean(ens_pred_t, axis=0)
ens_t_summary_df = pd.DataFrame(ens_t_summary, columns=test_data.uns['cell_types'])

#  Evaluation
sys.path.append(BASE_DIR+'/github/deconv-utils')
from src import evaluation as ev

dec_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
val_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
res = ev.eval_deconv(dec_name_list=dec_name_list, val_name_list=val_name_list, deconv_df=ens_t_summary_df, y_df=t_y_df)
res2 = ev.eval_deconv(dec_name_list=dec_name_list, val_name_list=val_name_list, deconv_df=ens_s_summary_df, y_df=train_y)

# summarize
r_list = []
mae_list = []
ccc_list = []
for i in range(len(dec_name_list)):
    tmp_res = res[i][0]
    r, mae, ccc = tmp_res['R'], tmp_res['MAE'], tmp_res['CCC']
    r_list.append(r)
    mae_list.append(mae)
    ccc_list.append(ccc)
summary_df = pd.DataFrame({'R':r_list, 'CCC':ccc_list, 'MAE':mae_list})
summary_df.index = [t[0] for t in val_name_list]
display(summary_df)

# %%
