#!/usr/bin/env python3
"""
Created on 2026-03-09 (Sun) 11:56:25

MMD (Maximum Mean Discrepancy)

@author: I.Azuma
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings('ignore')

from _utils import common_utils

cudnn.deterministic = True

class LossFunctions:
    def custom_loss(self, theta_tensor, prop_tensor):
        # deconvolution loss
        assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
        deconv_loss_dic = common_utils.calc_deconv_loss(theta_tensor, prop_tensor)
        deconv_loss = deconv_loss_dic['cos_sim'] + 0.0 * deconv_loss_dic['rmse']
        return deconv_loss


def _rbf_kernel_matrix(x, y, sigma_list):
    """
    Multi-kernel RBF matrix between x and y.

    Args:
        x: (n, d)
        y: (m, d)
        sigma_list: list of Gaussian bandwidths

    Returns:
        Kernel matrix (n, m)
    """
    dist2 = torch.cdist(x, y, p=2.0) ** 2
    kernels = [torch.exp(-dist2 / (2.0 * (sigma ** 2))) for sigma in sigma_list]
    return sum(kernels) / float(len(kernels))


def MMD_LOSS(H1, H2, sigma_list=None, device='cuda'):
    """
    MMD loss for source/target feature alignment.

    Args:
        H1: Source domain features (batch_size, feature_dim)
        H2: Target domain features (batch_size, feature_dim)
        sigma_list: Gaussian bandwidths for multi-kernel MMD
        device: kept for compatibility (actual device follows H1.device)

    Returns:
        Scalar MMD loss
    """
    if sigma_list is None:
        sigma_list = [1.0, 2.0, 4.0, 8.0, 16.0]

    if H1.ndim != 2 or H2.ndim != 2:
        raise ValueError(f"H1 and H2 must be 2D tensors, got {H1.ndim}D and {H2.ndim}D.")

    b1, p1 = H1.shape
    b2, p2 = H2.shape
    if p1 != p2:
        raise ValueError(f"Feature dimensions must match, got {p1} and {p2}.")

    b = min(b1, b2)
    if b == 0:
        return torch.tensor(0.0, device=H1.device, dtype=H1.dtype)

    X = H1[:b]
    Y = H2[:b]

    K_xx = _rbf_kernel_matrix(X, X, sigma_list)
    K_yy = _rbf_kernel_matrix(Y, Y, sigma_list)
    K_xy = _rbf_kernel_matrix(X, Y, sigma_list)

    if b > 1:
        diag_xx = torch.diagonal(K_xx)
        diag_yy = torch.diagonal(K_yy)
        mmd = (
            (K_xx.sum() - diag_xx.sum()) / (b * (b - 1))
            + (K_yy.sum() - diag_yy.sum()) / (b * (b - 1))
            - 2.0 * K_xy.mean()
        )
    else:
        mmd = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()

    return torch.clamp(mmd, min=0.0)


class MMD_Deconv(nn.Module):
    def __init__(self, option_list):
        super(MMD_Deconv, self).__init__()

        def _to_scalar_float(value, default):
            if value is None:
                return float(default)
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return float(default)
                return float(value.detach().reshape(-1)[0].item())
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return float(default)
                return float(value[0])
            return float(value)

        self.seed = option_list['seed']
        self.batch_size = option_list['batch_size']
        self.feature_num = option_list['feature_num']
        self.latent_dim = option_list['latent_dim']
        self.celltype_num = option_list['celltype_num']
        self.num_epochs = option_list['epochs']
        self.lr = option_list['learning_rate']
        self.early_stop = option_list['early_stop']
        self.pred_loss_type = option_list['pred_loss_type']

        self.pred_w = _to_scalar_float(option_list.get('pred_w', 1.0), 1.0)
        self.mmd_w = _to_scalar_float(
            option_list.get('mmd_w', option_list.get('dare_gram_w', 1.0)),
            1.0,
        )

        default_sigma = [1.0, 2.0, 4.0, 8.0, 16.0]
        sigma_values = option_list.get('mmd_sigma_list', default_sigma)
        if isinstance(sigma_values, torch.Tensor):
            sigma_values = sigma_values.detach().reshape(-1).tolist()
        if not isinstance(sigma_values, (list, tuple)) or len(sigma_values) == 0:
            sigma_values = default_sigma
        self.mmd_sigma_list = [float(s) for s in sigma_values]
        
        # 1. Feature Extractor (Encoder)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_num, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Deconv Predictor
        self.deconv_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.Dropout(p=0.2, inplace=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, self.celltype_num),
            nn.Softmax(dim=1)
        )

        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.losses = LossFunctions()

    def forward(self, x):
        """
        Forward pass for MMD model
        
        Args:
            x: Input features
        
        Returns:
            deconv_output: Deconvolution predictions
            features: Extracted features (for MMD loss computation)
        """
        # feature extraction
        features = self.feature_extractor(x)
        
        # label prediction
        deconv_output = self.deconv_predictor(features)
        
        return deconv_output, features
    
    def compute_mmd_loss(self, source_features, target_features, device='cuda'):
        """
        Compute MMD loss between source and target features
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            device: Device for computation
        
        Returns:
            MMD loss value
        """
        return MMD_LOSS(
            source_features,
            target_features,
            sigma_list=self.mmd_sigma_list,
            device=device
        )


class DAREGRAM_Deconv(MMD_Deconv):
    """Backward-compatible alias for old class name."""
    pass
