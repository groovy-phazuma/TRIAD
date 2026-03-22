#!/usr/bin/env python3
"""
Created on 2026-03-09 (Sun) 11:57:24

MMD Trainer for Deconvolution Tasks

@author: I.Azuma
"""

import os
import gc
import numpy as np
import pandas as pd

from collections import defaultdict

import torch
import torch.utils.data as Data

# Import DARE-GRAM model and dataset utilities
from da_models.mmd.mmd_model import *
from _utils.dataset import *

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
model_root = current_file.parents[2]
utils_path = model_root.parent / "deconv-utils"

if str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))

from src import evaluation as ev


class BaseTrainer:
    """
    Base class for MMD training. Provides methods for building dataloaders,
    setting options, and the main training loop with MMD-based domain adaptation.
    """
    def __init__(self, cfg, seed=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.seed = seed

    def build_dataloader(self, batch_size):
        """
        Build dataloaders for training and testing.
        """
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

        # sum to 1 amoung target_cells  NOTE: 250818
        #self.source_data_y = self.source_data_y / self.source_data_y.sum(axis=1, keepdims=True)

        den = self.source_data_y.sum(axis=1, keepdims=True)
        valid = den.squeeze() > 0
        self.source_data_x = self.source_data_x[valid]
        self.source_data_y = self.source_data_y[valid]
        den = den[valid]
        self.source_data_y = self.source_data_y / den

        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Extract celltype and feature info
        self.celltype_num = len(self.target_cells)
        self.used_features = list(source_data.var_names)

        # 2. Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)

        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g, drop_last=False)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    def set_options(self):
        """
        Create option list for the model and initialize parameters.
        """
        option_list = defaultdict(list)
        model_cfg = getattr(self.cfg, 'mmd', None)
        if model_cfg is None:
            model_cfg = getattr(self.cfg, 'daregram')
        for key, value in vars(model_cfg).items():
            option_list[key] = value
        option_list['feature_num'] = self.source_data.shape[1]
        option_list['celltype_num'] = len(self.target_cells)
        option_list['seed'] = self.seed

        self.option_list = option_list

        # parameter initialization
        self.best_loss = 1e10
        self.update_flag = 0

    def train_model(self, logger, eval_mode=False):
        model = MMD_Deconv(self.option_list).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model.num_epochs)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=model.lr, total_steps=model.num_epochs, pct_start=0.3, anneal_strategy='cos')
        
        # No domain discriminator criterion needed for MMD

        self.best_loss = float('inf')
        for epoch in range(model.num_epochs):
            loss_dict = self.run_epoch(model, epoch, optimizer)

            if eval_mode is not None:
                summary_df = self.eval_target(model)
                loss_dict.update({
                    'R': summary_df.loc['mean']['R'],
                    'CCC': summary_df.loc['mean']['CCC'],
                    'MAE': summary_df.loc['mean']['MAE'],
                })

            logger(epoch=epoch, **loss_dict)

            # Early stopping & Model Save
            if loss_dict['pred_loss'] < self.best_loss:
                self.update_flag = 0
                self.best_loss = loss_dict['pred_loss']
                torch.save(model.state_dict(), os.path.join(self.cfg.paths.model_path, f'best_model_{self.seed}.pth'))
            else:
                self.update_flag += 1
                if self.update_flag >= model.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch:{epoch}, Loss:{loss_dict['total_loss']:.3f}, pred:{loss_dict['pred_loss']:.3f}, mmd:{loss_dict['mmd_loss']:.3f}")
            
            scheduler.step()
            gc.collect()
        
        torch.save(model.state_dict(), os.path.join(self.cfg.paths.model_path, f'last_model_{self.seed}.pth'))

    def run_epoch(self, model, epoch, optimizer):
        model.train()
        total_pred_loss = 0.
        total_mmd_loss = 0.
        total_weighted_loss = 0.

        n_batches = len(self.train_source_loader)
        target_iter = iter(self.train_target_loader)
        
        for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
            # 1. data loading
            try:
                target_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(self.train_target_loader)
                target_x, _ = next(target_iter)
            source_x, source_y = source_x.to(self.device), source_y.to(self.device)
            target_x = target_x.to(self.device)

            # 2. Forward Pass
            pred_s, features_s = model(source_x)
            pred_t, features_t = model(target_x)

            # 3. Loss
            # Prediction Loss (only for source domain)
            if model.pred_loss_type == 'L1':
                pred_loss = F.l1_loss(pred_s, source_y)
            else:
                pred_loss = model.losses.custom_loss(pred_s, source_y)

            # MMD Loss (aligning source and target feature distributions)
            mmd_loss = model.compute_mmd_loss(features_s, features_t, device=self.device)

            # Total Loss
            total_loss = model.pred_w * pred_loss + model.mmd_w * mmd_loss

            # 4. Backward & Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 5. Logging
            total_pred_loss += pred_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_weighted_loss += total_loss.item()

        # Epoch summary
        avg_pred = total_pred_loss / n_batches
        avg_mmd = total_mmd_loss / n_batches
        avg_total = total_weighted_loss / n_batches

        return {
            'pred_loss': avg_pred,
            'mmd_loss': avg_mmd,
            'total_loss': avg_total,
        }

    def predict(self, model=None):
        """
        Make predictions using the trained model.
        """
        if model is None:
            model_path = os.path.join(self.cfg.paths.model_path, f'best_model_{self.seed}.pth')
            model = MMD_Deconv(self.option_list).cuda()
            model.load_state_dict(torch.load(model_path))
            print("Model loaded from %s" % model_path)

        model.eval()
        preds = None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            logits, features = model(x.cuda())
            logits = logits.detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
        final_preds_target = pd.DataFrame(preds, columns=self.target_cells)

        return final_preds_target

    def eval_target(self, model):
        pred_df = self.predict(model)

        dec_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
        val_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]

        res = ev.eval_deconv(
            dec_name_list=dec_name_list,
            val_name_list=val_name_list,
            deconv_df=pred_df,
            y_df=self.target_y,
            do_plot=False
        )

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
        summary_df = pd.DataFrame({'R': r_list, 'CCC': ccc_list, 'MAE': mae_list})
        summary_df.index = [t[0] for t in val_name_list]
        summary_df.loc['mean'] = summary_df.mean()
        
        return summary_df


class BenchmarkTrainer(BaseTrainer):
    """
    Trainer for benchmarking. Includes dataset preparation and evaluation metric computation.
    """
    def __init__(self, cfg, seed=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.seed = seed

        self.set_data()
        trainer_cfg = getattr(self.cfg, 'mmd', None)
        if trainer_cfg is None:
            trainer_cfg = getattr(self.cfg, 'daregram')
        self.build_dataloader(batch_size=trainer_cfg.batch_size)
        self.set_options()

    def set_data(self):
        train_data, test_data, train_y, test_y, gene_names = prep4benchmark(
            h5ad_path=self.cfg.paths.h5ad_path,
            source_list=self.cfg.common.source_domain,
            target=self.cfg.common.target_domain,
            target_cells=self.target_cells,
            priority_genes=self.cfg.common.marker_genes,
            n_samples=self.cfg.common.n_samples,
            n_vtop=self.cfg.common.n_vtop,
            seed=self.seed,
            vtop_mode=self.cfg.common.vtop_mode,
        )
        self.source_data = train_data
        self.target_data = test_data
        self.target_y = test_y
        self.gene_names = gene_names

    def train_model(self, logger):
        super().train_model(logger=logger)
    
    def target_inference(self, model=None):
        if model is None:
            model_path = os.path.join(self.cfg.paths.model_path, f'best_model_{self.seed}.pth')
            model = MMD_Deconv(self.option_list).cuda()
            model.load_state_dict(torch.load(model_path))
            print("Model loaded from %s" % model_path)

        model.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            logits, features = model(x.cuda())
            logits = logits.detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
        final_preds_target = pd.DataFrame(preds, columns=self.target_cells)

        return final_preds_target


class InferenceTrainer(BaseTrainer):
    """
    Trainer for inference only. Prepares the inference dataset.
    """
    def __init__(self, cfg, seed=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.seed = seed

        self.set_data()
        trainer_cfg = getattr(self.cfg, 'mmd', None)
        if trainer_cfg is None:
            trainer_cfg = getattr(self.cfg, 'daregram')
        self.build_dataloader(batch_size=trainer_cfg.batch_size)
        self.set_options()

    def set_data(self):
        train_data, test_data, train_y, gene_names = prep4inference(
            h5ad_path=self.cfg.paths.h5ad_path,
            target_path=self.cfg.paths.target_path,
            source_list=self.cfg.common.source_domain,
            target=self.cfg.common.target_domain,
            priority_genes=self.cfg.common.marker_genes,
            target_cells=self.target_cells,
            n_samples=self.cfg.common.n_samples,
            n_vtop=self.cfg.common.n_vtop,
            target_log_conv=self.cfg.common.target_log_conv,
            mm_scale=self.cfg.common.mm_scale,
            seed=self.seed,
            vtop_mode=self.cfg.common.vtop_mode,
        )
        self.source_data = train_data
        self.target_data = test_data
        self.gene_names = gene_names

    def train_model(self):
        super().train_model()
