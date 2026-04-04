# -*- coding: utf-8 -*-
"""
Created on 2025-04-30 (Wed) 11:05:55

@author: I.Azuma
"""
# %%
import os
import gc
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import cycle

from scipy import sparse
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

import torch

import sys
current_file = Path(__file__).resolve()
triad_root = current_file.parents[3]
project_root = triad_root.parent

if str(triad_root) not in sys.path:
    sys.path.append(str(triad_root))

from da_models.dael.route2 import dael_da
from da_models.dael.route2 import dael_utils
from da_models.dael.route2.reconstruction import DenoisingVAE, DualLatentVAE

utils_path = project_root / "deconv-utils"
if str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))
from src import evaluation as ev

wandb_utils_path = project_root / "wandb-util"
if str(wandb_utils_path) not in sys.path:
    sys.path.append(str(wandb_utils_path))
from wandbutil import WandbLogger



class SimpleTrainer():
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.noise_std_list = [0.0, 0.1, 0.5, 1.0]
        self.drop_prob_list = [0.0, 0.1, 0.2, 0.3]

        # Reconstruction backbone switch (backward compatible: default AE)
        self.rec_model_type = str(getattr(cfg.rec, 'model', 'ae')).lower()
        self.rec_use_stable_only = bool(getattr(cfg.rec, 'use_stable_only', False))
        self.rec_feature_cache = bool(getattr(cfg.rec, 'feature_cache', True))

        # Stabilize pseudo-label training while preserving original DAEL behavior.
        self.weight_u = float(getattr(cfg.dael, 'weight_u', 1.0))
        self.weight_u_min = float(getattr(cfg.dael, 'weight_u_min', 0.1))
        self.rampup_u_epochs = max(1, int(getattr(cfg.dael, 'rampup_u_epochs', max(10, cfg.dael.epochs // 5))))
        self.pseudo_conf_threshold = float(getattr(cfg.dael, 'pseudo_conf_threshold', 0.45))
        self.pseudo_conf_power = float(getattr(cfg.dael, 'pseudo_conf_power', 2.0))

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

        train_data, gene_names = dael_utils.prep_daeldg(h5ad_path=cfg.paths.h5ad_path, 
                                                        source_list=cfg.common.source_list + cfg.common.target_list, 
                                                        n_samples=cfg.common.n_samples, 
                                                        n_vtop=cfg.common.n_vtop)
        self.train_data = train_data
        self.gene_names = gene_names

        # Domain ids must match the order used in dael_da.BaseModel.aug_dataloader.
        all_domains = self.train_data.obs['ds'].unique().tolist()
        domain_dict = {ds: i for i, ds in enumerate(all_domains)}
        self.target_indices = [domain_dict[item] for item in cfg.common.target_list]
        print(f"Target indices: {self.target_indices}")

    
    def run_rec(self, noise=0.0):
        cfg = self.cfg
        # preparation of option list
        option_list = defaultdict(list)
        for key, value in vars(cfg.rec).items():
            option_list[key] = value
        option_list['feature_num'] = self.train_data.shape[1]

        rec_model = self._build_rec_model(option_list, seed=cfg.rec.seed).to(self.device)
        rec_model.aug_dataloader(self.train_data, batch_size=rec_model.batch_size, noise=noise, target_cells=self.target_cells)
        rec_path = self._resolve_rec_checkpoint(noise)
        rec_model.load_state_dict(torch.load(rec_path, map_location=self.device))
        rec_model.eval()

        cache_path = os.path.join(cfg.paths.rec_model_path, f'{self.rec_model_type}_noise{noise}_features.pt')
        if self.rec_feature_cache and os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location=self.device)
            print(f"Use cached features: {cache_path}")
            return cached['feats'], cached['domains'], cached['data_y']

        with torch.no_grad():
            data_x = torch.tensor(rec_model.data_x).to(self.device)
            data_y = torch.tensor(rec_model.data_y).to(self.device)
            feats = self._extract_feats(rec_model, data_x)
            domains = rec_model.domains.to(self.device)

        if self.rec_feature_cache:
            torch.save(
                {
                    'feats': feats.detach().cpu(),
                    'domains': domains.detach().cpu(),
                    'data_y': data_y.detach().cpu(),
                },
                cache_path,
            )
            print(f"Saved feature cache: {cache_path}")

        return feats, domains, data_y

    def _build_rec_model(self, option_list, seed=42):
        if self.rec_model_type == 'ae':
            return dael_da.AE(option_list, seed=seed)
        if self.rec_model_type == 'vae':
            return DenoisingVAE(option_list, seed=seed)
        if self.rec_model_type == 'dualvae':
            return DualLatentVAE(option_list, seed=seed)
        raise ValueError(f"Unknown reconstruction model: {self.rec_model_type}")

    def _extract_feats(self, rec_model, data_x):
        if self.rec_model_type == 'ae':
            _, feats = rec_model(data_x)
            return feats

        if self.rec_model_type == 'vae':
            _, feats, _, _ = rec_model(data_x)
            return feats

        # dualvae: keep both stable/noise-aware latents by default
        _, z_s, z_n, _, _, _, _, _ = rec_model(data_x)
        if self.rec_use_stable_only:
            return z_s
        return torch.cat((z_s, z_n), dim=1)

    def _resolve_rec_checkpoint(self, noise):
        cfg = self.cfg
        rec_dir = cfg.paths.rec_model_path

        prefix_map = {
            'ae': 'ae_rec',
            'vae': 'vae_rec',
            'dualvae': 'dualvae_rec',
        }
        prefix = prefix_map.get(self.rec_model_type)
        if prefix is None:
            raise ValueError(f"Unknown reconstruction model: {self.rec_model_type}")

        # Preferred naming: <prefix>_<noise>.pth
        candidates = [
            os.path.join(rec_dir, f'{prefix}_{noise}.pth'),
            os.path.join(rec_dir, f'{prefix}_{float(noise):.1f}.pth'),
            os.path.join(rec_dir, f'{prefix}_{str(noise).rstrip("0").rstrip(".")}.pth'),
        ]

        # Backward-compatible names for old AE checkpoints.
        if self.rec_model_type == 'ae':
            if abs(float(noise) - 0.0) < 1e-8:
                candidates.append(os.path.join(rec_dir, 'ae_rec_base.pth'))
            elif abs(float(noise) - 0.1) < 1e-8:
                candidates.append(os.path.join(rec_dir, 'ae_rec_weak.pth'))
            elif abs(float(noise) - 1.0) < 1e-8:
                candidates.append(os.path.join(rec_dir, 'ae_rec_strong.pth'))

        for path in candidates:
            if os.path.exists(path):
                print(f"Use reconstruction checkpoint: {path}")
                return path

        raise FileNotFoundError(
            f"Reconstruction checkpoint not found for model={self.rec_model_type}, noise={noise}. "
            f"Checked: {candidates}"
        )
    
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
        dael_model = dael_da.DAEL(option_list, seed=42).to(self.device)
        optimizer = torch.optim.Adam(dael_model.parameters(), lr=cfg.dael.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.dael.epochs)
        update_flag = 0
        unlabeled_iter = cycle(self.unlabel_loader)

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
            u_ramp = min(1.0, float(epoch + 1) / float(self.rampup_u_epochs))
            u_w = self.weight_u_min + (self.weight_u - self.weight_u_min) * u_ramp

            for _, batch_x in enumerate(self.label_loader):
                batch_u = next(unlabeled_iter)
                feat_x, feat_x2, prop_x, domain_x, feat_u, feat_u2 = dael_utils.parse_batch_train(batch_x, batch_u, self.device)

                # --- Generate pseudo label for unlabeled (target) data ---
                with torch.no_grad():
                    if self.expert_selection == 'conformal':
                        u_vars = []
                        for j in range(dael_model.n_domain):
                            domain_j = torch.full((feat_u.size(0),), j, device=feat_u.device)
                            aug_pool = []
                            for noise_v in self.noise_std_list:
                                for drop_v in self.drop_prob_list:
                                    # augmentation (adding noise x random masking)
                                    feat_u_aug = dael_utils.add_noise(feat_u, noise_std=noise_v)
                                    feat_u_aug = dael_utils.random_masking(feat_u_aug, mask_prob=drop_v)
                                    pred_j = dael_model.E(domain_j, feat_u_aug)  # (batch_size, num_classes)
                                    aug_pool.append(pred_j)
                            aug_stacked = torch.stack(aug_pool, dim=-1)  # (B, n_class, n_augmentation)
                            # calc var
                            var_mat = aug_stacked.std(dim=-1)  # (B, n_class)
                            #var_sample = var_mat.mean(dim=1)  # (B,)
                            var_sample = var_mat.max(dim=1).values  # (B,)
                            u_vars.append(var_sample)
                        #print(u_vars)
                        stacked = torch.stack(u_vars, dim=1)  # (B, n_domain)
                        min_idx = stacked.min(dim=1).indices  # (B, )  e.g. [3, 1, 3, 0, 3, 3, 1, 2, 3, 1, 1, 1]
                        pseudo_prop = dael_model.E(min_idx, feat_u)  # (batch_size, num_classes)

                    else:
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

                # Confidence-weighted pseudo-label loss to reduce noisy unlabeled supervision.
                conf = pseudo_prop.max(dim=1).values.detach()
                conf = torch.clamp((conf - self.pseudo_conf_threshold) / (1.0 - self.pseudo_conf_threshold + 1e-8), min=0.0, max=1.0)
                conf = conf.pow(self.pseudo_conf_power)
                loss_u = (((pseudo_prop - pred_u) ** 2).mean(dim=1) * conf).mean()

                # --- Backprop ---
                loss = 0
                loss += loss_x
                loss += loss_cr
                loss += loss_u * u_w

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_x_epoch += loss_x.item()
                loss_cr_epoch += loss_cr.item()
                loss_u_epoch += loss_u.item() * u_w

            n_batches = max(len(self.label_loader), 1)
            loss_x_epoch /= n_batches
            loss_cr_epoch /= n_batches
            loss_u_epoch /= n_batches
            
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
                print(
                    f"Epoch {epoch+1}: Loss: {loss_x_epoch + loss_cr_epoch + loss_u_epoch:.4f}, "
                    f"Loss_x: {loss_x_epoch:.4f}, Loss_cr: {loss_cr_epoch:.4f}, Loss_u: {loss_u_epoch:.4f}, "
                    f"u_w: {u_w:.3f}"
                )

            scheduler.step()
            

    def target_inference(self, dael_model, domains=torch.tensor([0,1,2,3]), target_domain=[4,5]):
        target_domain_tensor = torch.tensor(target_domain, device=domains.device)
        target_idx = torch.isin(domains, target_domain_tensor)
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
        dec_name_list = [[cell] for cell in self.target_cells]
        val_name_list = [[cell] for cell in self.target_cells]

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
        dael_model.load_state_dict(torch.load(os.path.join(cfg.paths.dael_model_path, f'dael_da_best.pth'), map_location=self.device))

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
        dec_name_list = [[cell] for cell in self.target_cells]
        val_name_list = [[cell] for cell in self.target_cells]

        for d in torch.unique(domains).tolist():
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

            print(summary_df)


def main(cfg):
    trainer = SimpleTrainer(cfg=cfg)
    trainer.train_dael()


if __name__ == '__main__':
    raise SystemExit(
        'This module is intended to be imported with a cfg object. '
        'Use the project-level runner or call main(cfg) from a config script.'
    )
