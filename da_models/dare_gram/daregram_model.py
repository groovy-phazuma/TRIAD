#!/usr/bin/env python3
"""
Created on 2026-03-09 (Sun) 11:56:25

DARE-GRAM: Unsupervised Domain Adaptation Regression by Aligning Inversed Gram Matrices (CVPR2023)
Adapted for deconvolution tasks

Reference: https://github.com/ismailnejjar/DARE-GRAM

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


def DARE_GRAM_LOSS(H1, H2, tradeoff_angle=0.05, tradeoff_scale=0.001, threshold=0.9, device='cuda'):
    """
    DARE-GRAM Loss: Aligning inversed Gram matrices between source and target features
    
    Args:
        H1: Source domain features (batch_size, feature_dim)
        H2: Target domain features (batch_size, feature_dim)
        tradeoff_angle: Weight for angle alignment
        tradeoff_scale: Weight for scale alignment
        threshold: Threshold for pseudo inverse
        device: Device to use for computation
    
    Returns:
        DARE-GRAM loss value
    """
    b1, p1 = H1.shape
    b2, p2 = H2.shape
    if p1 != p2:
        raise ValueError(f"Feature dimensions must match, got {p1} and {p2}.")

    b = min(b1, b2)
    p = p1
    H1 = H1[:b]
    H2 = H2[:b]
    device = H1.device

    # Augment features with ones column for affine transformation
    A = torch.cat((torch.ones(b, 1, device=device, dtype=H1.dtype), H1), 1)
    B = torch.cat((torch.ones(b, 1, device=device, dtype=H2.dtype), H2), 1)

    # Compute covariance (Gram) matrices
    cov_A = (A.t() @ A)
    cov_B = (B.t() @ B)

    # SVD decomposition to get eigenvalues
    _, L_A, _ = torch.linalg.svd(cov_A)
    _, L_B, _ = torch.linalg.svd(cov_B)
    
    # Compute cumulative eigenvalue ratios
    eigen_A = torch.cumsum(L_A.detach(), dim=0) / L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0) / L_B.sum()

    # Determine threshold for eigenvalue cutoff
    if eigen_A[1] > threshold:
        T = eigen_A[1].detach()
    else:
        T = threshold
    index_A = torch.argwhere(eigen_A.detach() <= T)[-1]

    if eigen_B[1] > threshold:
        T = eigen_B[1].detach()
    else:
        T = threshold
    index_B = torch.argwhere(eigen_B.detach() <= T)[-1]
    
    # Use maximum index for pseudo-inverse
    k = int(max(index_A, index_B)[0].item())
    k = max(k, 1)

    # Compute pseudo-inverse of covariance matrices
    A_inv = torch.linalg.pinv(cov_A, rcond=(L_A[k] / L_A[0]).detach())
    B_inv = torch.linalg.pinv(cov_B, rcond=(L_B[k] / L_B[0]).detach())
    
    # Angle alignment: cosine similarity between inverse matrices
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos = torch.dist(torch.ones((p + 1), device=device, dtype=A_inv.dtype), (cos_sim(A_inv, B_inv)), p=1) / (p + 1)
    
    # Scale alignment: distance between eigenvalues
    scale_dist = torch.dist((L_A[:k]), (L_B[:k])) / k
    
    return tradeoff_angle * cos + tradeoff_scale * scale_dist

class DAREGRAM_Deconv(nn.Module):
    def __init__(self, option_list):
        super(DAREGRAM_Deconv, self).__init__()

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
        self.dare_gram_w = _to_scalar_float(option_list.get('dare_gram_w', 1.0), 1.0)
        
        # DARE-GRAM hyperparameters
        self.tradeoff_angle = _to_scalar_float(option_list.get('tradeoff_angle', 0.05), 0.05)
        self.tradeoff_scale = _to_scalar_float(option_list.get('tradeoff_scale', 0.001), 0.001)
        self.threshold = _to_scalar_float(option_list.get('threshold', 0.9), 0.9)
        
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
        Forward pass for DARE-GRAM model
        
        Args:
            x: Input features
        
        Returns:
            deconv_output: Deconvolution predictions
            features: Extracted features (for DARE-GRAM loss computation)
        """
        # feature extraction
        features = self.feature_extractor(x)
        
        # label prediction
        deconv_output = self.deconv_predictor(features)
        
        return deconv_output, features
    
    def compute_dare_gram_loss(self, source_features, target_features, device='cuda'):
        """
        Compute DARE-GRAM loss between source and target features
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            device: Device for computation
        
        Returns:
            DARE-GRAM loss value
        """
        return DARE_GRAM_LOSS(
            source_features, 
            target_features,
            tradeoff_angle=self.tradeoff_angle,
            tradeoff_scale=self.tradeoff_scale,
            threshold=self.threshold,
            device=device
        )
