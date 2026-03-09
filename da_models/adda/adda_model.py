#!/usr/bin/env python3
"""
Created on 2026-02-15 (Sun) 11:56:25

ADDA: Adversarial Discriminative Domain Adaptation (2017)

Key differences from DANN:
- Uses separate source and target feature extractors
- Source model is pre-trained and frozen
- Target model is initialized from source and adapted adversarially
- Alternating optimization: discriminator → target feature extractor

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


def set_requires_grad(model, requires_grad=True):
    """Helper function to set requires_grad for all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = requires_grad


class FeatureExtractor(nn.Module):
    """Feature extractor network for ADDA"""
    def __init__(self, feature_num, latent_dim):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_num, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.network(x)


class DeconvPredictor(nn.Module):
    """Deconvolution predictor network"""
    def __init__(self, latent_dim, celltype_num):
        super(DeconvPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Dropout(p=0.2, inplace=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, celltype_num),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)


class DomainDiscriminator(nn.Module):
    """Domain discriminator for ADDA (distinguishes source vs target features)"""
    def __init__(self, latent_dim):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(50, 20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class ADDA_Deconv(nn.Module):
    """
    ADDA for deconvolution task
    
    Training procedure:
    1. Pre-train source model (source_feature_extractor + deconv_predictor) on source domain
    2. Initialize target_feature_extractor from source_feature_extractor
    3. Adversarial adaptation:
       - Train discriminator to distinguish source vs target features
       - Train target_feature_extractor to fool the discriminator (with flipped labels)
    """
    def __init__(self, option_list):
        super(ADDA_Deconv, self).__init__()

        self.seed = option_list['seed']
        self.batch_size = option_list['batch_size']
        self.feature_num = option_list['feature_num']
        self.latent_dim = option_list['latent_dim']
        self.celltype_num = option_list['celltype_num']
        self.num_epochs = option_list['epochs']
        self.lr = option_list['learning_rate']
        self.early_stop = option_list['early_stop']
        self.pred_loss_type = option_list['pred_loss_type']

        self.k_disc = option_list['k_disc']
        self.k_target = option_list['k_target']
        
        # 1. Source Feature Extractor (pre-trained, will be frozen during adaptation)
        self.source_feature_extractor = FeatureExtractor(self.feature_num, self.latent_dim)
        
        # 2. Target Feature Extractor (initialized from source, will be adapted)
        self.target_feature_extractor = FeatureExtractor(self.feature_num, self.latent_dim)
        
        # 3. Deconv Predictor (shared, frozen during adaptation)
        self.deconv_predictor = DeconvPredictor(self.latent_dim, self.celltype_num)

        # 4. Domain Discriminator (trained to distinguish source vs target)
        self.discriminator = DomainDiscriminator(self.latent_dim)

        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.losses = LossFunctions()

    def forward_source(self, x):
        """Forward pass through source model"""
        features = self.source_feature_extractor(x)
        deconv_output = self.deconv_predictor(features)
        return deconv_output, features
    
    def forward_target(self, x):
        """Forward pass through target model"""
        features = self.target_feature_extractor(x)
        deconv_output = self.deconv_predictor(features)
        return deconv_output, features
    
    def init_target_from_source(self):
        """Initialize target feature extractor from source (called after pre-training)"""
        self.target_feature_extractor.load_state_dict(
            self.source_feature_extractor.state_dict()
        )
