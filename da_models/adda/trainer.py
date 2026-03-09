#!/usr/bin/env python3
"""
Created on 2026-02-15 (Sun) 11:57:24

@author: I.Azuma
"""

import os
import gc
import random
import numpy as np
import pandas as pd

from itertools import cycle
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

# Import ADDA-based model and dataset utilities
from da_models.adda.adda_model import *
from _utils.dataset import *

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
model_root = current_file.parents[2]
utils_path = model_root.parent / "deconv-utils"

if str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))

from src import evaluation as ev


def set_seed(seed):
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch, and CUDA.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Enable deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


class BaseTrainer:
    """
    Base class for training. Provides methods for building dataloaders, setting options, and the main training loop.
    """
    def __init__(self, cfg, seed=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.seed = seed
        
        # Set random seeds for reproducibility
        set_seed(seed)

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
        for key, value in vars(self.cfg.adda).items():
            option_list[key] = value
        option_list['feature_num'] = self.source_data.shape[1]
        option_list['celltype_num'] = len(self.target_cells)
        option_list['seed'] = self.seed

        self.option_list = option_list

        # parameter initialization
        self.best_loss = 1e10
        self.update_flag = 0

    def train_model(self, logger, eval_mode=False):
        model = ADDA_Deconv(self.option_list).to(self.device)
        
        # ===================================
        # Phase 1: Pre-train Source Model
        # ===================================
        print("\n=== Phase 1: Pre-training Source Model ===")
        source_params = list(model.source_feature_extractor.parameters()) + \
                       list(model.deconv_predictor.parameters())
        source_optimizer = torch.optim.Adam(source_params, lr=model.lr, weight_decay=1e-5)
        source_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            source_optimizer, max_lr=model.lr, 
            total_steps=model.num_epochs, pct_start=0.3, anneal_strategy='cos'
        )
        
        self.best_loss = float('inf')
        for epoch in range(model.num_epochs):
            loss_dict = self.pretrain_source_epoch(model, epoch, source_optimizer)
            
            if eval_mode:
                summary_df = self.eval_target_source(model)
                loss_dict.update({
                    'R': summary_df.loc['mean']['R'],
                    'CCC': summary_df.loc['mean']['CCC'],
                    'MAE': summary_df.loc['mean']['MAE'],
                })
            
            logger(epoch=epoch, phase='pretrain', **loss_dict)
            
            # Early stopping & Model Save
            if loss_dict['pred_loss'] < self.best_loss:
                print(f'Save model at {epoch}')
                self.update_flag = 0
                self.best_loss = loss_dict['pred_loss']
                torch.save(model.state_dict(), os.path.join(self.cfg.paths.model_path, f'pretrain_best_{self.seed}.pth'))
            else:
                self.update_flag += 1
                if self.update_flag >= model.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Pretrain Epoch:{epoch}, Loss:{loss_dict['pred_loss']:.4f}")
            
            source_scheduler.step()
            gc.collect()
        
        # Initialize target from source
        model.init_target_from_source()
        print("Target feature extractor initialized from source.")
        
        # ===================================
        # Phase 2: Adversarial Adaptation
        # ===================================
        print("\n=== Phase 2: Adversarial Adaptation ===")
        
        # Freeze source and predictor
        set_requires_grad(model.source_feature_extractor, requires_grad=False)
        set_requires_grad(model.deconv_predictor, requires_grad=False)
        
        # Setup optimizers for adversarial phase
        discriminator_optim = torch.optim.Adam(model.discriminator.parameters(), lr=model.lr*0.1, weight_decay=1e-5)
        target_optim = torch.optim.Adam(model.target_feature_extractor.parameters(), lr=model.lr*0.01, weight_decay=1e-5)
        
        criterion_bce = nn.BCEWithLogitsLoss().to(self.device)
        
        # Reset best loss for adaptation phase
        self.best_loss = float('inf')
        self.update_flag = 0
        
        adapt_epochs = model.num_epochs  # Same number of epochs for adaptation
        # Reset epoch counter for Phase II logging
        for adapt_epoch in range(adapt_epochs):
            loss_dict = self.adversarial_adaptation_epoch(
                model, adapt_epoch, discriminator_optim, target_optim, criterion_bce
            )
            
            if eval_mode:
                summary_df = self.eval_target(model)
                loss_dict.update({
                    'R': summary_df.loc['mean']['R'],
                    'CCC': summary_df.loc['mean']['CCC'],
                    'MAE': summary_df.loc['mean']['MAE'],
                })
            
            logger(epoch=adapt_epoch, phase='adaptation', **loss_dict)
            

            auc_distance = abs(loss_dict['disc_auc'] - 0.5)
            
            if auc_distance < self.best_loss:
                self.update_flag = 0
                self.best_loss = auc_distance
                print(f'New best model found at: {adapt_epoch} with disc_auc: {loss_dict["disc_auc"]:.4f}')
                torch.save(model.state_dict(), os.path.join(self.cfg.paths.model_path, f'best_model_{self.seed}.pth'))
            else:
                self.update_flag += 1
                if self.update_flag >= model.early_stop:
                    print(f"Early stopping at epoch {adapt_epoch}")
                    break
            
            if adapt_epoch % 10 == 0:
                
                print(f"Adapt Epoch:{adapt_epoch}, Disc AUC: {loss_dict['disc_auc']:.4f}")
            
            gc.collect()
        
        torch.save(model.state_dict(), os.path.join(self.cfg.paths.model_path, f'last_model_{self.seed}.pth'))

    def pretrain_source_epoch(self, model, epoch, optimizer):
        """Phase 1: Pre-train source model on source domain"""
        model.train()
        total_pred_loss = 0.
        n_batches = len(self.train_source_loader)
        
        for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
            source_x, source_y = source_x.to(self.device), source_y.to(self.device)
            
            # Forward pass through source model
            pred_s, _ = model.forward_source(source_x)
            
            # Prediction Loss
            if model.pred_loss_type == 'L1':
                pred_loss = F.l1_loss(pred_s, source_y)
            else:
                pred_loss = model.losses.custom_loss(pred_s, source_y)
            
            # Backward & Optimization
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()
            
            total_pred_loss += pred_loss.item()
        
        avg_pred = total_pred_loss / n_batches
        return {'pred_loss': avg_pred}
    
    def adversarial_adaptation_epoch(self, model, epoch, discriminator_optim, target_optim, criterion_bce):
        model.train()
        
        total_disc_loss = 0.
        total_target_loss = 0.
        all_domain_preds = []
        all_domain_labels = []
        
        # Create iterators
        source_iter = iter(cycle(self.train_source_loader))
        target_iter = iter(cycle(self.train_target_loader))
        
        # Use target loader length as base for iterations
        n_iterations = len(self.train_target_loader)
        
        for iteration in range(n_iterations):
            # ============================================
            # Step 1: Train Discriminator (k_disc times)
            # Goal: Classify source(1) vs target(0) with HIGH accuracy
            # HIGH accuracy = domains are still different = adaptation needed
            # ============================================
            set_requires_grad(model.target_feature_extractor, requires_grad=False)
            set_requires_grad(model.discriminator, requires_grad=True)
            
            disc_iteration_loss = 0.
            disc_iteration_acc = 0.
            
            for _ in range(model.k_disc):
                source_x, source_y = next(source_iter)
                target_x, _ = next(target_iter)
                source_x = source_x.to(self.device)
                target_x = target_x.to(self.device)
                
                # Extract features (source is frozen, no gradient)
                with torch.no_grad():
                    _, source_features = model.forward_source(source_x)
                _, target_features = model.forward_target(target_x)
                
                # Combine features and labels
                combined_features = torch.cat([source_features, target_features])
                domain_labels = torch.cat([
                    torch.ones(source_x.shape[0], device=self.device),
                    torch.zeros(target_x.shape[0], device=self.device)
                ])
                
                # Discriminator prediction
                domain_preds = model.discriminator(combined_features).squeeze()
                disc_loss = criterion_bce(domain_preds, domain_labels)
                
                # Update discriminator
                discriminator_optim.zero_grad()
                disc_loss.backward()
                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                discriminator_optim.step()
                
                disc_iteration_loss += disc_loss.item()
                
                # Track predictions for accuracy
                with torch.no_grad():
                    # Accuracy: whether sigmoid(pred) correctly classifies 0/1
                    disc_preds_binary = (torch.sigmoid(domain_preds) > 0.5).float()
                    disc_accuracy = (disc_preds_binary == domain_labels).float().mean().item()
                    disc_iteration_acc += disc_accuracy
                    
                    all_domain_preds.extend(torch.sigmoid(domain_preds).detach().cpu().numpy())
                    all_domain_labels.extend(domain_labels.detach().cpu().numpy())
                
                total_disc_loss += disc_loss.item()
            
            # ============================================
            # Step 2: Train Target Feature Extractor (k_target times)
            # Goal: Fool discriminator by generating target features classified as source(1)
            # This should DECREASE discriminator accuracy
            # ============================================
            set_requires_grad(model.target_feature_extractor, requires_grad=True)
            set_requires_grad(model.discriminator, requires_grad=False)
            
            for _ in range(model.k_target):
                target_x, _ = next(target_iter)
                target_x = target_x.to(self.device)
                
                # Extract target features
                _, target_features = model.forward_target(target_x)
                
                # Flipped labels: want discriminator to classify as source (1)
                flipped_labels = torch.ones(target_x.shape[0], device=self.device)
                
                # Discriminator prediction on target features
                domain_preds = model.discriminator(target_features).squeeze()
                target_loss = criterion_bce(domain_preds, flipped_labels)
                
                # Update target feature extractor
                target_optim.zero_grad()
                target_loss.backward()
                target_optim.step()
                
                total_target_loss += target_loss.item()
        
        # Epoch summary
        avg_disc_loss = total_disc_loss / (n_iterations * model.k_disc)
        avg_disc_acc = roc_auc_score(all_domain_labels, all_domain_preds)
        avg_target_loss = total_target_loss / (n_iterations * model.k_target)
        
        return {
            'disc_loss': avg_disc_loss,
            'disc_auc': avg_disc_acc,
            'target_loss': avg_target_loss
        }

    def predict(self, model=None):
        """
        Make predictions using the trained target model.
        """
        if model is None:
            model_path = os.path.join(self.cfg.paths.model_path, f'best_model_{self.seed}.pth')
            model = ADDA_Deconv(self.option_list).cuda()
            model.load_state_dict(torch.load(model_path))
            print("Model loaded from %s" % model_path)

        model.eval()
        preds = None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            # Use target model for predictions
            logits, _ = model.forward_target(x.cuda())
            logits = logits.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
        final_preds_target = pd.DataFrame(preds, columns=self.target_cells)

        return final_preds_target
    
    def eval_target_source(self, model):
        """Evaluate using source model (during pre-training phase)"""
        model.eval()
        preds = None
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_target_loader):
                logits, _ = model.forward_source(x.cuda())
                logits = logits.detach().cpu().numpy()
                preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
        pred_df = pd.DataFrame(preds, columns=self.target_cells)

        dec_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
        val_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]

        res = ev.eval_deconv(
            dec_name_list=dec_name_list,
            val_name_list=val_name_list,
            deconv_df=pred_df,
            y_df=self.target_y,
            do_plot=False
        )

        r_list, mae_list, ccc_list = [], [], []
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
        self.build_dataloader(batch_size=cfg.adda.batch_size)
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
            model = ADDA_Deconv(self.option_list).cuda()
            model.load_state_dict(torch.load(model_path))
            print("Model loaded from %s" % model_path)

        model.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            # Use target model for predictions
            logits, _ = model.forward_target(x.cuda())
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
        self.build_dataloader(batch_size=cfg.adda.batch_size)
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
