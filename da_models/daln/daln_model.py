#!/usr/bin/env python3
"""
Created on 2026-02-18 (Wed) 00:41:16

@author: I.Azuma
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Tuple, Dict
from torch.autograd import Function

class LossFunctions:
    def custom_loss(self, predicted_props, true_props):
        assert predicted_props.shape == true_props.shape, "Shape mismatch between prediction and ground truth"
        
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        loss_cos = 1.0 - cos_sim(predicted_props, true_props).mean()
        
        mse = F.mse_loss(predicted_props, true_props)
        loss_rmse = torch.sqrt(mse + 1e-8)
    
        total_loss = loss_cos + 0.0 * loss_rmse
        
        return total_loss


class SortedMSE(nn.Module):
    def __init__(self):
        super(SortedMSE, self).__init__()

    @staticmethod
    def regression_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        Sorted MSE Loss 
        """
        min_len = min(y_s.size(0), y_t.size(0))
        y_s = y_s[:min_len]
        y_t = y_t[:min_len]

        # 1. sort the predictions for each cell type
        y_s_sorted, _ = torch.sort(y_s, dim=0)
        y_t_sorted, _ = torch.sort(y_t, dim=0)

        # 2. calculate mse for each sorted value
        loss = F.mse_loss(y_s_sorted, y_t_sorted)
        
        return loss

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        if self.training:
            
            # calculate regression discrepancy
            dist_loss = self.regression_discrepancy(y_s, y_t)
            
            return dist_loss
        else:
            return torch.tensor(0.0, device=y_s.device)

# ==========================================
# Main Model (Feature Extractor + Deconv)
# ==========================================

class DALN_Deconv(nn.Module):
    def __init__(self, option_list: Dict[str, Any]):
        super(DALN_Deconv, self).__init__()

        self.seed = option_list['seed']
        self.batch_size = option_list['batch_size']
        self.feature_num = option_list['feature_num']
        self.latent_dim = option_list['latent_dim']
        self.celltype_num = option_list['celltype_num']
        self.num_epochs = option_list['epochs']
        self.lr = option_list['learning_rate']
        self.early_stop = option_list['early_stop']
        self.pred_loss_type = option_list['pred_loss_type']

        self.pred_w = option_list['pred_w']
        self.nwd_w = option_list['nwd_w']
        
        # seed settings
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # 1. Feature Extractor (Encoder)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_num, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Deconv Predictor (Latent -> Proportions)
        self.deconv_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.Dropout(p=0.2, inplace=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, self.celltype_num),
            nn.Softmax(dim=1)
        )

        # 3. Discrepancy Module
        self.discrepancy = SortedMSE()
        self.loss_fn = LossFunctions()

    def forward(self, x_s: torch.Tensor, y_s: Optional[torch.Tensor] = None, x_t: Optional[torch.Tensor] = None):
        
        # --- 1. Source Data Flow (Main Task) ---
        features_s = self.feature_extractor(x_s)
        features_t = self.feature_extractor(x_t) if x_t is not None else None
        pred_s = self.deconv_predictor(features_s)
        pred_t = self.deconv_predictor(features_t) if features_t is not None else None
        
        # Deconvolution task
        deconv_loss = torch.tensor(0.0, device=x_s.device)
        if self.pred_loss_type == 'custom' and y_s is not None:
            deconv_loss = self.loss_fn.custom_loss(pred_s, y_s)
        elif self.pred_loss_type == 'L1' and y_s is not None:
            deconv_loss = F.l1_loss(pred_s, y_s)

        # --- 2. Domain Adaptation (Discrepancy Loss) ---
        rwd_loss = torch.tensor(0.0, device=x_s.device)
        if features_t is not None:
            rwd_loss = self.discrepancy(features_s, features_t)

        return pred_s, pred_t, deconv_loss, rwd_loss
