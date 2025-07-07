#!/usr/bin/env python3
"""
Created on 2025-06-27 (Fri) 12:08:14

TRIAD (Tissue-adaptive Representation via Integrated graph Autoencoder for Deconvolution) Trainer

This module provides training classes for the TRIAD model, which is designed for cell type deconvolution
using domain adaptation techniques. The module includes:

1. BaseTrainer: Base class providing core functionality for model training, including:
   - Data loader construction for source and target domains
   - Training loop with DAG (Directed Acyclic Graph) constraints
   - Domain adaptation using gradient reversal layers
   - Early stopping and model checkpointing

2. BenchmarkTrainer: Extends BaseTrainer for benchmarking purposes, including:
   - Automatic evaluation metrics calculation (R, CCC, MAE)
   - Integration with evaluation utilities for performance assessment

3. InferenceTrainer: Extends BaseTrainer for inference-only tasks, providing:
   - Streamlined setup for making predictions on new data
   - Simplified workflow without evaluation metrics

@author: I.Azuma
"""
import os
import gc
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
import torch.utils.data as Data

# Set base directory and change working directory
BASE_DIR = '/workspace/cluster/HDD/azuma/TopicModel_Deconv'
os.chdir(BASE_DIR)

# Import TRIAD model and utilities
sys.path.append(BASE_DIR + '/github/TRIAD/triad')
from model.route9.triad_model import *
from _utils.dataset import *

# Import WandB logger
sys.path.append(BASE_DIR + '/github/wandb-util')
from wandbutil import WandbLogger  # type: ignore

# Import evaluation utilities
sys.path.append(BASE_DIR + '/github/deconv-utils')
from src import evaluation as ev  # type: ignore

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
        for key, value in vars(self.cfg.triad).items():
            option_list[key] = value
        option_list['feature_num'] = self.source_data.shape[1]
        option_list['celltype_num'] = len(self.target_cells)

        self.option_list = option_list

        # parameter initialization
        self.best_loss = 1e10
        self.update_flag = 0
        self.w_stop_flag = False
        self.alpha, self.beta, self.rho = 0.0, 2.0, 1.0
        self.gamma = 0.25
        self.l1_penalty = 0.0
        self.h_thresh = 1e-4
        self.pre_h = np.inf

    def train_model(self, inference_fn=None):
        """
        Main training loop for the TRIAD model.
        """
        model = TRIAD(self.option_list, seed=self.seed).to(self.device)
        optimizer1 = torch.optim.Adam([
            {'params': model.encoder.parameters()},
            {'params': model.decoder.parameters()},
            {'params': model.w}
        ], lr=model.lr)

        optimizer2 = torch.optim.Adam([
            {'params': model.embedder.parameters()},
            {'params': model.predictor.parameters()},
            {'params': model.discriminator.parameters()}
        ], lr=model.lr)

        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.8)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.8)
        criterion_da = nn.BCELoss().to(self.device)

        source_label = torch.ones(model.batch_size).unsqueeze(1).to(self.device)
        target_label = torch.zeros(10000).unsqueeze(1).to(self.device)

        # WandB logger settings
        logger = WandbLogger(
            entity=self.cfg.wandb.entity,
            project=self.cfg.wandb.project,
            group=self.cfg.wandb.group,
            name=self.cfg.wandb.name + f"_seed{self.seed}",
            config=self.option_list,
        )

        for epoch in range(model.num_epochs + 1):
            loss_dict, curr_h = self.run_epoch(model, epoch, optimizer1, optimizer2, criterion_da, source_label, target_label)

            # update dag restricion
            if (epoch + 1) % 10 == 0 and not self.w_stop_flag:
                if curr_h > self.gamma * self.pre_h:
                    self.rho *= self.beta
                self.alpha += self.rho * curr_h.detach().cpu()
                self.pre_h = curr_h
                if curr_h <= self.h_thresh and epoch > 100:
                    print(f"Stopped updating W at epoch {epoch+1}")
                    self.w_stop_flag = True

            # Inference
            if inference_fn is not None:
                summary_df = inference_fn(model)
                loss_dict.update({
                    'R': summary_df.loc['mean']['R'],
                    'CCC': summary_df.loc['mean']['CCC'],
                    'MAE': summary_df.loc['mean']['MAE'],
                })

            logger(epoch=epoch, **loss_dict)

            # Early stopping
            if loss_dict['pred_disc_loss'] < self.best_loss:
                self.update_flag = 0
                self.best_loss = loss_dict['pred_disc_loss']
                torch.save(model.state_dict(), os.path.join(self.cfg.paths.triad_model_path, f'best_model_{self.seed}.pth'))
            else:
                self.update_flag += 1
                if self.update_flag == model.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch:{epoch}, Loss:{loss_dict['total_loss']:.3f}, dag:{loss_dict['dag_loss']:.3f}, pred:{loss_dict['pred_loss']:.3f}, disc:{loss_dict['disc_loss']:.3f}, disc_auc:{loss_dict['disc_auc']:.3f}")

            gc.collect()
        
            # Step the schedulers
            #scheduler1.step()
            #scheduler2.step()

        torch.save(model.state_dict(), os.path.join(self.cfg.paths.triad_model_path, f'last_model.pth'))

    def run_epoch(self, model, epoch, optimizer1, optimizer2, criterion_da, source_label, target_label):
        """
        Train the model for one epoch.
        """
        model.train()
        dag_loss_epoch, pred_loss_epoch, disc_loss_epoch = 0., 0., 0.
        all_preds = []
        all_labels = []
        for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
            target_x = next(iter(self.train_target_loader))[0]  # NOTE: shuffle

            total_steps = model.num_epochs * len(self.train_source_loader)
            p = float(batch_idx + epoch * len(self.train_source_loader)) / total_steps
            a = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            source_x, source_y, target_x = source_x.cuda(), source_y.cuda(), target_x.cuda()
            rec_s, _, _ = model(source_x, a)
            rec_t, _, _ = model(target_x, a)

            # 1. DAG-related loss
            w_adj = model.w_adj
            curr_h = model.losses.compute_h(w_adj)
            curr_mse_s = model.losses.dag_rec_loss(target_x.reshape((target_x.size(0), target_x.size(1), 1)), rec_t)
            curr_mse_t = model.losses.dag_rec_loss(source_x.reshape((source_x.size(0), source_x.size(1), 1)), rec_s)
            curr_mse = curr_mse_s + curr_mse_t
            dag_loss = (curr_mse
                        + self.l1_penalty * torch.norm(w_adj, p=1)
                        + self.alpha * curr_h + 0.5 * self.rho * curr_h * curr_h)

            dag_loss_epoch += dag_loss.data.item()
            dag_loss = model.dag_w * dag_loss

            if not self.w_stop_flag:
                optimizer1.zero_grad()
                dag_loss.backward(retain_graph=True)
                optimizer1.step()

            # NOTE: re-obtain the prediction and domain classification
            _, pred_s, domain_s = model(source_x, a)
            _, pred_t, domain_t = model(target_x, a)

            # 2. prediction
            if model.pred_loss_type == 'L1':
                pred_loss = model.losses.L1_loss(pred_s, source_y.cuda())
            elif model.pred_loss_type == 'custom':
                pred_loss = model.losses.summarize_loss(pred_s, source_y.cuda())
            else:
                raise ValueError("Invalid prediction loss type.")
            pred_loss_epoch += pred_loss.data.item()

            # 3. domain classification
            all_preds.extend(domain_s.cpu().detach().numpy().flatten())  # Source domain predictions
            all_preds.extend(domain_t.cpu().detach().numpy().flatten())  # Target domain predictions
            all_labels.extend([1] * domain_s.shape[0])  # Source domain labels
            all_labels.extend([0] * domain_t.shape[0])  # Target domain labels

            disc_loss = criterion_da(domain_s, source_label[0:domain_s.shape[0], ]) + criterion_da(domain_t, target_label[0:domain_t.shape[0], ])
            disc_loss_epoch += disc_loss.data.item()

            # 4. pred_loss + disc_loss
            loss = model.pred_w * pred_loss + model.disc_w * disc_loss

            optimizer2.zero_grad()
            loss.backward(retain_graph=True)
            optimizer2.step()

        # summarize loss
        dag_loss_epoch = model.dag_w * dag_loss_epoch / len(self.train_source_loader)
        pred_loss_epoch = model.pred_w * pred_loss_epoch / len(self.train_source_loader)
        disc_loss_epoch = model.disc_w * disc_loss_epoch / len(self.train_source_loader)
        loss_all = dag_loss_epoch + pred_loss_epoch + disc_loss_epoch
        auc_score = roc_auc_score(all_labels, all_preds)

        loss_dict = {
            'dag_loss': dag_loss_epoch,
            'pred_loss': pred_loss_epoch,
            'disc_loss': disc_loss_epoch,
            'total_loss': loss_all,
            'pred_disc_loss': pred_loss_epoch + disc_loss_epoch,
            'disc_auc': auc_score,
        }

        return loss_dict, curr_h

    def predict(self):
        """
        Make predictions using the trained model.
        """
        model_path = os.path.join(self.cfg.paths.triad_model_path, f'best_model_{self.seed}.pth')
        model = TRIAD(self.option_list, seed=self.seed).cuda()
        model.load_state_dict(torch.load(model_path))

        model.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            rec, logits, domain = model(x.cuda(), alpha=1.0)
            logits = logits.detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
        final_preds_target = pd.DataFrame(preds, columns=self.target_cells)

        return final_preds_target, gt

    def get_wadj(self):
        """
        Retrieve the W_adj matrix from the trained model.
        """
        model_path = os.path.join(self.cfg.paths.triad_model_path, f'best_model_{self.seed}.pth')
        model = TRIAD(self.option_list, seed=self.seed).cuda()
        model.load_state_dict(torch.load(model_path))

        w_adj = model.w_adj.detach().cpu().numpy()
        gene_names = self.gene_names
        w_df = pd.DataFrame(w_adj, index=gene_names, columns=gene_names)
        return w_df

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
        self.build_dataloader(batch_size=cfg.triad.batch_size)
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

    def train_model(self):
        def inference_fn(model):
            summary_df, _ = self.target_inference(model=model, do_plot=False)
            return summary_df
        super().train_model(inference_fn=inference_fn)

    def target_inference(self, model=None, do_plot=False):
        if model is None:
            model_path = os.path.join(self.cfg.paths.triad_model_path, f'best_model_{self.seed}.pth')
            model = TRIAD(self.option_list, seed=self.seed).cuda()
            model.load_state_dict(torch.load(model_path))
            print("Model loaded from %s" % model_path)

        model.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            rec, logits, domain = model(x.cuda(), alpha=1.0)
            logits = logits.detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
        final_preds_target = pd.DataFrame(preds, columns=self.target_cells)

        dec_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
        val_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
        res = ev.eval_deconv(
            dec_name_list=dec_name_list,
            val_name_list=val_name_list,
            deconv_df=final_preds_target,
            y_df=self.target_y,
            do_plot=do_plot
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

        return summary_df, final_preds_target

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
        self.build_dataloader(batch_size=cfg.triad.batch_size)
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
            seed=self.seed,
            vtop_mode=self.cfg.common.vtop_mode,
        )
        self.source_data = train_data
        self.target_data = test_data
        self.gene_names = gene_names

    def train_model(self):
        super().train_model(inference_fn=None)
