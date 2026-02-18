#!/usr/bin/env python3
"""
Created on 2026-02-18 (Wed) 00:41:16

Model Description: A model that faithfully incorporates "DALN: Reusing the Task-specific Classifier as a Discriminator: Discriminator-free Adversarial Domain Adaptation (2022)."

Technical Concern: Since deconvolution tasks are inherently regression-based, there is a concern that calculating NWD (Nuclear Wasserstein Discrepancy) loss using the nuclear norm may be poorly suited for this application.

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


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        return input.view_as(input)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000, auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.training and self.auto_step:
            self.step()
            
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1


class NuclearWassersteinDiscrepancy(nn.Module):
    def __init__(self, classifier: nn.Module, input_dim: int, max_iters: int):

        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=max_iters, auto_step=True)
        self.classifier = classifier
        self.input_dim = input_dim

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        Nuclear-norm Wasserstein Discrepancy (NWD) loss.
        """
        loss = (-torch.norm(y_t, 'nuc')/y_t.shape[0]) + (torch.norm(y_s, 'nuc')/y_s.shape[0])
        return loss

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        """
        f_s: Source Features
        f_t: Target Features
        """
        if self.training:
            f_cat = torch.cat((f_s, f_t), dim=0)
            
            # GRL layer
            f_grl = self.grl(f_cat)
            
            # Classfier
            y_adv = self.classifier(f_grl)
            
            # split
            batch_s = f_s.shape[0]
            batch_t = f_t.shape[0]
            y_s_adv, y_t_adv = y_adv[:batch_s], y_adv[batch_s:batch_s+batch_t]
            
            # NWD Loss
            nwd_loss = self.n_discrepancy(y_s_adv, y_t_adv)
            return nwd_loss
        else:
            return torch.tensor(0.0, device=f_s.device)


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
        self.discrepancy = NuclearWassersteinDiscrepancy(
            classifier=self.deconv_predictor, 
            input_dim=self.latent_dim,
            max_iters=self.num_epochs
        )

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
        nwd_loss = torch.tensor(0.0, device=x_s.device)
        if features_t is not None:
            nwd_loss = self.discrepancy(features_s, features_t)


        return pred_s, pred_t, deconv_loss, nwd_loss
