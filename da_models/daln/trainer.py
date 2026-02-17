#!/usr/bin/env python3
"""
Created on 2026-02-15 (Sun) 11:57:24

Trainer for DALN

@author: I.Azuma
"""

import os
import gc
import sys
import numpy as np
import pandas as pd
from itertools import cycle
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

# Import DALN-based model and dataset utilities
# Ensure these paths are correct in your environment
current_file = Path(__file__).resolve()
model_root = current_file.parents[2]
utils_path = model_root.parent / "deconv-utils"

if str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))

from da_models.daln.daln_model import DALN_Deconv
from _utils.dataset import prep4benchmark, seed_worker
from src import evaluation as ev


class BaseTrainer:
    """
    Base class for training. Provides methods for building dataloaders, setting options, and the main training loop.
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

    def set_options(self):
        """
        Create option list for the model and initialize parameters.
        """
        option_list = defaultdict(list)
        for key, value in vars(self.cfg.daln).items():
            option_list[key] = value
        option_list['feature_num'] = self.source_data.shape[1]
        option_list['celltype_num'] = len(self.target_cells)
        option_list['seed'] = self.seed
        
        # Ensure latent_dim is int
        if 'latent_dim' in option_list:
             option_list['latent_dim'] = int(option_list['latent_dim'])

        self.option_list = option_list

        # parameter initialization
        self.best_loss = 1e10
        self.update_flag = 0

    def train_model(self, logger, eval_mode=False):
        model = DALN_Deconv(self.option_list).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=1e-5)
        
        # Scheduler configuration
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=model.lr, total_steps=model.num_epochs, 
            pct_start=0.3, anneal_strategy='cos'
        )
        
        self.best_loss = float('inf')
        target_iter = cycle(self.train_target_loader)
        
        for epoch in range(model.num_epochs):
            loss_dict = self.run_epoch(model, epoch, optimizer, target_iter)

            # Evaluation phase
            if eval_mode is not None:
                summary_df = self.eval_target(model)
                loss_dict.update({
                    'R': summary_df.loc['mean']['R'],
                    'CCC': summary_df.loc['mean']['CCC'],
                    'MAE': summary_df.loc['mean']['MAE'],
                })

            # Logging
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
                print(f"Epoch:{epoch}, Total:{loss_dict['total_loss']:.3f}, Pred:{loss_dict['pred_loss']:.3f}, Disc(NWD):{loss_dict['nwd_loss']:.3f}")
            
            scheduler.step()
            gc.collect()
        
        torch.save(model.state_dict(), os.path.join(self.cfg.paths.model_path, f'last_model_{self.seed}.pth'))

    def run_epoch(self, model, epoch, optimizer, target_iter):
        model.train()
        total_pred_loss = 0.
        total_nwd_loss = 0.
        
        n_batches = len(self.train_source_loader)
        
        for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
            # 1. Data loading
            target_x, _ = next(target_iter)
            source_x, source_y = source_x.to(self.device), source_y.to(self.device)
            target_x = target_x.to(self.device)

            # 2. Forward Pass
            pred_s, pred_t, pred_loss, nwd_loss = model(x_s = source_x,
                                                        y_s = source_y,
                                                        x_t = target_x)

            # Total Loss
            total_loss = model.pred_w * pred_loss + model.nwd_w * nwd_loss

            # 4. Backward & Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 5. Accumulate metrics
            total_pred_loss += pred_loss.item()
            total_nwd_loss += nwd_loss.item()

        # Epoch summary
        # Avoid division by zero if all batches were skipped
        if n_batches > 0:
            avg_pred = total_pred_loss / n_batches
            avg_nwd = total_nwd_loss / n_batches
        else:
            avg_pred = 0.
            avg_nwd = 0.

        return {
            'pred_loss': avg_pred,
            'nwd_loss': avg_nwd,
            'total_loss': avg_pred + avg_nwd,
            'disc_auc': 0.0 # AUC is not applicable for NWD (Wasserstein distance)
        }

    def predict(self, model=None):
        """
        Make predictions using the model.
        """
        if model is None:
            model_path = os.path.join(self.cfg.paths.model_path, f'best_model_{self.seed}.pth')
            model = DALN_Deconv(self.option_list).cuda()
            model.load_state_dict(torch.load(model_path))
            print("Model loaded from %s" % model_path)
        
        model.eval()
        preds = None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            logits, _, _, _ = model(x_s=x.to(self.device), y_s=None, x_t=None)
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
    Trainer for benchmarking. Includes dataset preparation.
    """
    def __init__(self, cfg, seed=42):
        super().__init__(cfg, seed)
        
        # Initialize data and options
        self.set_data()
        self.set_options() # Must be called after set_data to get feature_num
        self.build_dataloader(batch_size=int(cfg.daln.batch_size))

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
        # Call the base class training method
        # We pass eval_mode=True to perform evaluation during training if desired
        super().train_model(logger=logger, eval_mode=True)
