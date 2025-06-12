# -*- coding: utf-8 -*-
"""
Created on 2025-03-18 (Tue) 13:38:28

R: 0.599
CCC: 0.420

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'

import gc
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
from model.route4_gae_grl.gae_grl_col import *

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch.nn.functional  as F
from torchviz import make_dot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
#  Load dataset
trainingdatapath = BASE_DIR+"/datasource/scRNASeq/Scaden/pbmc_data.h5ad"
train_data, test_data, train_y, test_y = preprocess(trainingdatapath,source='data6k',target='sdy67', n_samples=1024, n_vtop=1000)

#  Parameter settings
option_list = defaultdict(list)
option_list['type_list']=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells']
option_list['ref_sample_num'] = train_data.shape[0]
option_list['target_sample_num'] = test_data.shape[0]
option_list['batch_size'] = 32
option_list['epochs'] = 1000
option_list['pred_loss_type'] = 'custom'


option_list['feature_num'] = train_data.shape[1]
option_list['latent_dim'] = 256
option_list['hidden_dim'] = 16  # FIXME
option_list['d'] = train_data.shape[1]
option_list['celltype_num'] = train_y.shape[1]
option_list['hidden_layers'] = 1
option_list['learning_rate'] = 1e-3
option_list['early_stop'] = 10

option_list['dag_w'] = 0.1
option_list['pred_w'] = 100
option_list['disc_w'] = 0.1

# Output parameters
option_list['SaveResultsDir'] = BASE_DIR+"/workspace/240816_model_trial/250115_scpDeconv/250303_gae_dev/results/250303_gae_initial_dev/"

# %% run
# prepare model structure
model = MultiTaskAutoEncoder(option_list).cuda()
model.prepare_dataloader(train_data, test_data, model.batch_size)

# setup optimizer
optimizer1 = torch.optim.Adam([{'params': model.encoder.parameters()},
                               {'params': model.decoder.parameters()},
                               {'params': model.w}],
                               lr=model.lr)

optimizer2 = torch.optim.Adam([#{'params': model.encoder.parameters()},  # FIXME
                               {'params': model.embedder.parameters()},
                               {'params': model.predictor.parameters()},
                               {'params': model.discriminator.parameters()},],
                               lr=model.lr)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.8)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.8)

criterion_da = nn.BCELoss().cuda()
source_label = torch.ones(model.batch_size).unsqueeze(1).cuda()   # source domain label as 1
target_label = torch.zeros(model.batch_size).unsqueeze(1).cuda()  # target domain label as 0

model.metric_logger = defaultdict(list) 
best_loss = 1e10  
update_flag = 0  

l1_penalty = 0.0
alpha, beta, rho = 0.0, 2.0, 1.0
gamma = 0.25
rho_thresh = 1e30
h_thresh = 1e-8
pre_h = np.inf
prev_w_est, prev_mse = None, np.inf
accumulation_steps = 8  # 16*8 = 128
w_stop_flag = False


for epoch in range(model.num_epochs+1):
    model.train()

    train_target_iterator = iter(model.train_target_loader)
    dag_loss_epoch, pred_loss_epoch, disc_loss_da_epoch = 0., 0., 0.
    all_preds = []
    all_labels = []
    for batch_idx, (source_x, source_y) in enumerate(model.train_source_loader):
        target_x = next(iter(model.train_target_loader))[0]   # NOTE: shuffle

        total_steps = model.num_epochs * len(model.train_source_loader)
        p = float(batch_idx + epoch * len(model.train_source_loader)) / total_steps
        a = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        source_x, source_y, target_x = source_x.cuda(), source_y.cuda(), target_x.cuda()
        rec_s, _, _= model(source_x, a)
        rec_t, _, _ = model(target_x, a)

        #### 1. DAG-related loss
        w_adj = model.w_adj
        curr_h = model.losses.compute_h(w_adj)
        curr_mse_s = model.losses.dag_rec_loss(target_x.reshape((target_x.size(0), target_x.size(1), 1)), rec_t)  # NOTE target_x: (12, n_genes) --> (12, n_genes, 1)
        curr_mse_t = model.losses.dag_rec_loss(source_x.reshape((source_x.size(0), source_x.size(1), 1)), rec_s)
        curr_mse = curr_mse_s + curr_mse_t
        dag_loss = (curr_mse
                    + l1_penalty * torch.norm(w_adj, p=1)
                    + alpha * curr_h + 0.5 * rho * curr_h * curr_h)

        dag_loss_epoch += dag_loss.data.item()
        dag_loss = model.dag_w * dag_loss

        if w_stop_flag:
            pass
        else:
            optimizer1.zero_grad()
            dag_loss.backward(retain_graph=True)
            optimizer1.step()

        # NOTE: re-obtain the prediction and domain classification
        _, pred_s, domain_s = model(source_x, a)
        _, pred_t, domain_t = model(target_x, a)

        #### 2. prediction
        if model.pred_loss_type == 'L1':
            pred_loss = model.losses.L1_loss(pred_s, source_y.cuda())
        elif model.pred_loss_type == 'custom':
            pred_loss = model.losses.summarize_loss(pred_s, source_y.cuda())
        else:
            raise ValueError("Invalid prediction loss type.")
        pred_loss_epoch += pred_loss.data.item()

        #### 3. domain classification
        all_preds.extend(domain_s.cpu().detach().numpy().flatten())  # Source domain predictions
        all_preds.extend(domain_t.cpu().detach().numpy().flatten())  # Target domain predictions
        all_labels.extend([1]*domain_s.shape[0])  # Source domain labels
        all_labels.extend([0]*domain_t.shape[0])  # Target domain labels

        disc_loss_da = criterion_da(domain_s, source_label[0:domain_s.shape[0],]) + criterion_da(domain_t, target_label[0:domain_t.shape[0],])
        disc_loss_da_epoch += disc_loss_da.data.item()

        #### 4. pred_loss + disc_loss_da
        loss = model.pred_w * pred_loss + model.disc_w * disc_loss_da
        optimizer2.zero_grad()
        loss.backward(retain_graph=True)
        optimizer2.step()

    # update alpha and rho
    if (epoch+1) % 10 == 0:
        if w_stop_flag:
            pass
        else:
            if curr_h > gamma * pre_h:
                rho *= beta
            else:
                pass
            alpha += rho * curr_h.detach().cpu()
            pre_h = curr_h
    
    # summarize loss
    dag_loss_epoch = model.dag_w * dag_loss_epoch / len(model.train_source_loader)
    pred_loss_epoch = model.pred_w * pred_loss_epoch / len(model.train_source_loader)
    disc_loss_da_epoch = model.disc_w * disc_loss_da_epoch / len(model.train_source_loader)
    loss_all = dag_loss_epoch + pred_loss_epoch + disc_loss_da_epoch

    auc_score = roc_auc_score(all_labels, all_preds)

    model.metric_logger['dag_loss'].append(dag_loss_epoch)
    model.metric_logger['pred_loss'].append(pred_loss_epoch)
    model.metric_logger['disc_loss_DA'].append(disc_loss_da_epoch)
    model.metric_logger['total_loss'].append(loss_all)
    model.metric_logger['disc_auc'].append(auc_score)

    if epoch % 10 == 0:
        print(f"Epoch:{epoch}, Loss:{loss_all:.3f}, dag:{dag_loss_epoch:.3f}, pred:{pred_loss_epoch:.3f}, disc_da:{disc_loss_da_epoch:.3f}, disc_auc:{auc_score:.3f}")    

        # save best model
        target_loss = dag_loss_epoch + pred_loss_epoch
        if target_loss < best_loss:
            update_flag = 0
            best_loss = target_loss
            model.metric_logger['best_epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(model.outdir, 'best_model.pth'))
        else:
            update_flag += 1
            if update_flag == model.early_stop:
                print("Early stopping at epoch %d" % (epoch+1))
                break


# %%
# summarize loss history
metric_logger = model.metric_logger
best_epoch = metric_logger['best_epoch']
print(f"Best epoch: {best_epoch}")
loss_type = ['total_loss','dag_loss','pred_loss', 'disc_loss_DA', 'disc_auc']
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(len(loss_type)):
    axes[i].plot(metric_logger[loss_type[i]][10::])
    axes[i].set_title(loss_type[i], x=0.5, y=0.5)
    axes[i].axvline(best_epoch, color='r', linestyle='--')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Loss')
    axes[i].grid(True)
plt.tight_layout()
plt.show()

# %% Inference & Evaluation
dat = MultiTaskAutoEncoder(option_list).cuda()
dat.prepare_dataloader(train_data, test_data, option_list['batch_size'])
dat.load_state_dict(torch.load(os.path.join(dat.outdir, 'best_model.pth')))
dat.eval()

preds, gt = None, None
for batch_idx, (x, y) in enumerate(dat.test_target_loader):
    rec, logits, domain = dat(x.cuda(), alpha=1.0)
    logits = logits.detach().cpu().numpy()
    frac = y.detach().cpu().numpy()
    preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
    gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
final_preds_target = pd.DataFrame(preds, columns=dat.labels)

preds, gt = None, None
for batch_idx, (x, y) in enumerate(dat.train_source_loader):
    rec, logits, domain = dat(x.cuda(), alpha=1.0)
    logits = logits.detach().cpu().numpy()
    frac = y.detach().cpu().numpy()
    preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
    gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
final_preds_source = pd.DataFrame(preds, columns=dat.labels)
final_gt_source = pd.DataFrame(gt, columns=dat.labels)


sys.path.append(BASE_DIR+'/github/deconv-utils')
from src import evaluation as ev

dec_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
val_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
res = ev.eval_deconv(dec_name_list=dec_name_list, val_name_list=val_name_list, deconv_df=final_preds_target, y_df=test_y)
#res2 = ev.eval_deconv(dec_name_list=dec_name_list, val_name_list=val_name_list, deconv_df=final_preds_source, y_df=final_gt_source)

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
summary_df.loc['mean'] = summary_df.mean()
display(summary_df)
