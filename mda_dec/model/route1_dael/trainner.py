# -*- coding: utf-8 -*-
"""
Created on 2025-04-30 (Wed) 11:05:55

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'
import os
os.chdir(BASE_DIR)

import gc
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

import torch

import sys
sys.path.append(BASE_DIR+'/github/GSTMDec/mda_dec')
from model.route1_dael import dael_da
from model.route1_dael import dael_dg
from model.route1_dael import dael_utils

sys.path.append(BASE_DIR+'/github/deconv-utils')
from src import evaluation as ev

sys.path.append(BASE_DIR+'/github/wandb-util')  
from wandbutil import WandbLogger



class SimpleTrainer():
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells

        self.data_preprocessing()
        print("--- Complete Data Preprocessing ---")
        
        self.feats0, self.domains, self.data_y = self.run_rec(noise=0.0)  # without augmentation
        self.feats1, _, _ = self.run_rec(noise=0.1)  # weak augmentation
        self.feats2, _, _ = self.run_rec(noise=1.0)  # strong augmentation
        print("--- Complete Feature Extraction ---")
        
        self.build_data_loader()
        print("--- Complete Build Data Loader ---")
        
    
    def data_preprocessing(self):
        cfg = self.cfg

        total_list = cfg.common.source_list + cfg.common.target_list
        self.target_indices = [total_list.index(item) for item in cfg.common.target_list]
        print(f"Target indices: {self.target_indices}")

        train_data, gene_names = dael_utils.prep_daeldg(h5ad_path=cfg.paths.h5ad_path, 
                                                        source_list=total_list, 
                                                        n_samples=cfg.common.n_samples, 
                                                        n_vtop=cfg.common.n_vtop)
        self.train_data = train_data
        self.gene_names = gene_names

    
    def run_rec(self, noise=0.0):
        cfg = self.cfg
        # preparation of option list
        option_list = defaultdict(list)
        for key, value in vars(cfg.rec).items():
            option_list[key] = value
        option_list['feature_num'] = self.train_data.shape[1]

        rec_model = dael_da.AE(option_list, seed=cfg.rec.seed).to(self.device)
        rec_model.aug_dataloader(self.train_data, batch_size=rec_model.batch_size, noise=noise, target_cells=self.target_cells)
        rec_model.load_state_dict(torch.load(os.path.join(cfg.paths.rec_model_path, f'ae_rec_{noise}.pth')))
        rec_model.eval()

        data_x = torch.tensor(rec_model.data_x).to(self.device)
        data_y = torch.tensor(rec_model.data_y).to(self.device)
        rec_x, feats = rec_model(data_x)
        domains = rec_model.domains.to(self.device)

        return feats, domains, data_y
    
    def build_data_loader(self):
        cfg = self.cfg

        feats1 = self.feats1.cpu().detach()
        feats2 = self.feats2.cpu().detach()
        label_loader, unlabel_loader = dael_utils.build_daelda_loader(self.train_data, feats1, feats2, batch_size=cfg.dael.batch_size, source_domains=cfg.common.source_list, target_domains=cfg.common.target_list, shuffle=True, target_cells=self.target_cells)

        self.label_loader = label_loader
        self.unlabel_loader = unlabel_loader

    def train_dael(self):
        cfg = self.cfg

        # preparation of option list
        option_list = defaultdict(list)
        for key, value in vars(cfg.dael).items():
            option_list[key] = value
        
        assert self.feats0.shape[1] == self.feats1.shape[1] == self.feats2.shape[1], "Feature dimension mismatch!"
        option_list['feature_num'] = self.feats0.shape[1]
        option_list['celltype_num'] = len(self.target_cells)
        option_list['n_domain'] = len(cfg.common.source_list)

        self.expert_selection = cfg.dael.expert_selection
        self.weight_u = cfg.dael.weight_u

        dael_model = dael_da.DAEL(option_list, seed=42).to(self.device)
        optimizer = torch.optim.Adam(dael_model.parameters(), lr=cfg.dael.learning_rate)

        # WandB logger settings
        logger = WandbLogger(
            entity=cfg.wandb.entity,  
            project=cfg.wandb.project,  
            group=cfg.wandb.group, 
            name=cfg.wandb.name,
            config=option_list,
        )

        best_loss = 1e10
        for epoch in range(cfg.dael.epochs):
            dael_model.train()
            loss_x_epoch, loss_cr_epoch, loss_u_epoch = 0, 0, 0

            for _, batch_x in enumerate(self.label_loader):
                batch_u = next(iter(self.unlabel_loader))
                feat_x, feat_x2, prop_x, domain_x, feat_u, feat_u2 = dael_utils.parse_batch_train(batch_x, batch_u, self.device)

                # --- Generate pseudo label for unlabeled (target) data ---
                with torch.no_grad():
                    u_preds = []
                    for j in range(dael_model.n_domain):
                        # Pass all the feat (weak augumented) to the jth Expert
                        domain_j = torch.full((feat_u.size(0),), j, device=feat_u.device)
                        pred_j = dael_model.E(domain_j, feat_u)  # (batch_size, num_classes)
                        u_preds.append(pred_j)
                    stacked = torch.stack(u_preds, dim=-1)  # (B, n_class, n_domain)
                    if self.expert_selection == 'mean':
                        pseudo_prop = stacked.mean(dim=-1)
                    elif self.expert_selection == 'max':
                        max_idx = stacked.max(dim=1).values.max(dim=1).indices
                        max_idx = max_idx.view(-1, 1)
                        max_idx = max_idx.expand(-1, stacked.size(1))
                        pseudo_prop = stacked.gather(dim=-1, index=max_idx.unsqueeze(-1)).squeeze(-1)
                    else:
                        raise ValueError(f"Unknown expert selection method: {self.expert_selection}")

                # --- Expert prediction (weak augmentation) ---
                pred = dael_model.E(domain_x, feat_x)  # (batch_size, num_classes)
                loss_x = ((prop_x - pred) ** 2).mean()

                expert_pred = pred.detach()

                # --- Consistency regularization (strong augmentation) ---
                loss_cr = 0
                cr_preds = []
                for j in range(dael_model.n_domain):
                    mask = (domain_x != j)  # (batch_size,) bool tensor
                    mask_counts = mask.sum()  # number of samples not in domain j
                    # Pass all the feat2 to the jth Expert
                    domain_j = torch.full_like(domain_x, j)
                    pred_j = dael_model.E(domain_j, feat_x2)  # (batch_size, num_classes)
                    pred_j = pred_j * mask.unsqueeze(1).float()
                    cr_preds.append(pred_j)  # use masked area
                
                stacked = torch.stack(cr_preds, dim=-1)  # (stacked_size, num_classes, num_domains)
                mask = (stacked != 0)
                sum_valid = (stacked * mask).sum(dim=-1)
                count_valid = mask.sum(dim=-1)
                assert (count_valid == dael_model.n_domain-1).all(), "Not all elements are K-1!"
                cr_preds_m = sum_valid / (count_valid + 1e-8)  # (batch, num_classes)
                
                loss_cr = ((cr_preds_m - expert_pred) ** 2).mean()

                # --- Unsupervised loss ---
                u_preds = []
                for j in range(dael_model.n_domain):
                    # Pass all the feat (strong augumented) to the jth Expert
                    domain_j = torch.full((feat_u2.size(0),), j, device=feat_u2.device)
                    pred_j = dael_model.E(domain_j, feat_u2)  # (batch_size, num_classes)
                    u_preds.append(pred_j)
                stacked = torch.stack(u_preds, dim=-1)  # (B, n_class, n_domain)
                pred_u = stacked.mean(dim=-1)

                loss_u = ((pseudo_prop - pred_u) ** 2).mean()

                # --- Backprop ---
                loss = 0
                loss += loss_x
                loss += loss_cr
                loss += loss_u * self.weight_u

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_x_epoch += loss_x.item()
                loss_cr_epoch += loss_cr.item()
                loss_u_epoch += loss_u.item() * self.weight_u
            
            # save best model and early stopping
            target_loss = loss_x_epoch + loss_cr_epoch + loss_u_epoch
            if target_loss < best_loss:
                update_flag = 0
                best_loss = target_loss
                torch.save(dael_model.state_dict(), os.path.join(cfg.paths.dael_model_path, f'dael_da_best.pth'))
            else:
                update_flag += 1
                if update_flag == option_list['early_stop']:
                    print("Early stopping at epoch %d" % (epoch+1))
                    break
            
            # inference
            summary_df = self.target_inference(dael_model, self.domains, target_domain=self.target_indices)
            
            # logging
            logger(
                epoch=epoch + 1,
                loss=loss_x_epoch + loss_cr_epoch + loss_u_epoch,
                loss_x=loss_x_epoch,
                loss_cr=loss_cr_epoch,
                loss_u=loss_u_epoch,
                R=summary_df.loc['mean']['R'],
                CCC=summary_df.loc['mean']['CCC'],
                MAE=summary_df.loc['mean']['MAE'],
            )
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss: {loss_x_epoch + loss_cr_epoch + loss_u_epoch:.4f}, Loss_x: {loss_x_epoch:.4f}, Loss_cr: {loss_cr_epoch:.4f}, Loss_u: {loss_u_epoch:.4f}")
            

    def target_inference(self, dael_model, domains=torch.tensor([0,1,2,3]), target_domain=[4,5]):
        target_idx = torch.isin(domains, torch.tensor(target_domain, device=self.device))
        t_feats = self.feats0[target_idx,:]
        data_y = self.data_y[target_idx,:]

        # inference target
        dael_model.eval()
        u_preds = []
        for j in range(dael_model.n_domain):
            # Pass all the feat (weak augumented) to the jth Expert
            domain_j = torch.full((t_feats.shape[0],), j, device=self.device)
            pred_j = dael_model.E(domain_j, t_feats)  # (batch_size, num_classes)
            u_preds.append(pred_j)
        stacked = torch.stack(u_preds, dim=-1)  # (B, n_class, n_domain)
        p_k_t = stacked.mean(dim=-1)

        dec_df = pd.DataFrame(p_k_t.cpu().detach().numpy(), columns=self.target_cells)
        y_df = pd.DataFrame(data_y.cpu().detach().numpy(), columns=self.target_cells)

        # Evaluation
        dec_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
        val_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]

        res = ev.eval_deconv(dec_name_list=dec_name_list, val_name_list=val_name_list, deconv_df=dec_df, y_df=y_df, do_plot=False)

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


        return summary_df
    
    def overall_inference(self, domains, target_domain=[], plot_target=True):
        cfg = self.cfg

        # preparation of option list
        option_list = defaultdict(list)
        for key, value in vars(cfg.dael).items():
            option_list[key] = value
        
        assert self.feats0.shape[1] == self.feats1.shape[1] == self.feats2.shape[1], "Feature dimension mismatch!"
        option_list['feature_num'] = self.feats0.shape[1]
        option_list['celltype_num'] = len(self.target_cells)
        option_list['n_domain'] = len(cfg.common.source_list)

        self.expert_selection = cfg.dael.expert_selection
        self.weight_u = cfg.dael.weight_u

        if target_domain == []:
            target_domain = self.target_indices
        target_idx = torch.isin(domains, torch.tensor(target_domain, device=domains.device))
        s_domains = domains[~target_idx]
        t_domains = domains[target_idx]

        s_feats = self.feats0[~target_idx,:]
        t_feats = self.feats0[target_idx,:]

        # load best model
        dael_model = dael_da.DAEL(option_list, seed=42).to(self.device)
        dael_model.load_state_dict(torch.load(os.path.join(cfg.paths.dael_model_path, f'dael_da_best.pth')))

        # inference source
        dael_model.eval()
        p_k_s = dael_model.E(s_domains, s_feats)

        # inference target
        dael_model.eval()
        u_preds = []
        for j in range(dael_model.n_domain):
            # Pass all the feat (weak augumented) to the jth Expert
            domain_j = torch.full((t_feats.shape[0],), j, device=t_feats.device)
            pred_j = dael_model.E(domain_j, t_feats)  # (batch_size, num_classes)
            u_preds.append(pred_j)
        stacked = torch.stack(u_preds, dim=-1)  # (B, n_class, n_domain)
        p_k_t = stacked.mean(dim=-1)

        # concat
        p_k = torch.cat((p_k_s, p_k_t), dim=0)


        dec_df = pd.DataFrame(p_k.cpu().detach().numpy(), columns=self.target_cells)
        y_df = pd.DataFrame(self.data_y.cpu().detach().numpy(), columns=self.target_cells)
        dec_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
        val_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]

        for d in range(dael_model.n_domain+1):
            # select the domain index
            d_idx = (domains.cpu().detach().numpy() == d).nonzero()[0]
            d_dec_df = dec_df.iloc[d_idx, :]
            d_y_df = y_df.iloc[d_idx, :]

            do_plot = plot_target and d in target_domain
            res = ev.eval_deconv(dec_name_list=dec_name_list, val_name_list=val_name_list,
                     deconv_df=d_dec_df, y_df=d_y_df, do_plot=do_plot)
                
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
