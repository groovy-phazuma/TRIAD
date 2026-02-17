#!/usr/bin/env python3
"""
Created on 2026-02-18 (Wed) 00:16:05

DALN: Reusing the Task-specific Classifier as a Discriminator:
Discriminator-free Adversarial Domain Adaptation (2022

@author: I.Azuma
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from da_models.daln.nwd import NuclearWassersteinDiscrepancy

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

class DALN_Deconv(nn.Module):
    def __init__(self, option_list):
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
        self.disc_w = option_list['disc_w']

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

        self.discrepancy = NuclearWassersteinDiscrepancy(classifier=self.deconv_predictor[-1])  # Use the last layer of deconv_predictor as classifier

        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.losses = LossFunctions()

    def forward(self, x):
        # feature extraction
        features = self.feature_extractor(x)
        
        # label prediction
        deconv_output = self.deconv_predictor(features)
        
        # domain classification with GRL
        discrepancy_loss = -self.discrepancy(features)
        
        return deconv_output, discrepancy_loss
