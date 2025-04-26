# -*- coding: utf-8 -*-
"""
Created on 2025-04-22 (Tue) 10:45:07

@author: I.Azuma
"""
# %%
import anndata
import numpy as np
import scanpy as sc

import torch
from torch.utils.data import Dataset, DataLoader

def preprocess(trainingdatapath, source='data6k', target='sdy67', 
               priority_genes=[], target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], n_samples=None, n_vtop=None):
    assert target in ['sdy67', 'GSE65133', 'donorA', 'donorC', 'data6k', 'data8k']
    pbmc = sc.read_h5ad(trainingdatapath)
    test = pbmc[pbmc.obs['ds']==target]

    if n_samples is not None:
        np.random.seed(42)
        idx = np.random.choice(8000, n_samples, replace=False)
        donorA = pbmc[pbmc.obs['ds']=='donorA'][idx]
        donorC = pbmc[pbmc.obs['ds']=='donorC'][idx]
        data6k = pbmc[pbmc.obs['ds']=='data6k'][idx]
        data8k = pbmc[pbmc.obs['ds']=='data8k'][idx]
    
    else:    
        donorA = pbmc[pbmc.obs['ds']=='donorA']
        donorC = pbmc[pbmc.obs['ds']=='donorC']
        data6k = pbmc[pbmc.obs['ds']=='data6k']
        data8k = pbmc[pbmc.obs['ds']=='data8k']

    if source == 'all':
        train = anndata.concat([donorA, donorC, data6k, data8k])
    else:
        if n_samples is not None:
            train = pbmc[pbmc.obs['ds']==source][idx]
        else:
            train = pbmc[pbmc.obs['ds']==source]

    train_y = train.obs[target_cells]
    test_y = test.obs[target_cells]
    
    if n_vtop is None:
        #### variance cut off
        label = test.X.var(axis=0) > 0.1  # FIXME: mild cut-off
        label_idx = np.where(label)[0]
    else:
        #### top 1000 highly variable genes
        label_idx = np.argsort(-train.X.var(axis=0))[:n_vtop]
    
    # add priority genes
    priority_label = np.array([True if gene in priority_genes else False for gene in train.var_names])
    priority_idx = np.where(priority_label)[0]
    print(f"Priority genes: {np.sum(priority_label)}/{len(priority_genes)} genes")
    label_idx = np.unique(np.concatenate([label_idx, priority_idx]))
    gene_names = train.var_names[label_idx]
    
    train_data = train[:, label_idx]
    train_data.X = np.log2(train_data.X + 1)
    test_data = test[:, label_idx]
    if target != 'GSE65133':
        test_data.X = np.log2(test_data.X + 1)
    else:
        # GSE65133 is already log2 transformed
        test_data.X = test_data.X

    print("Train data shape: ", train_data.X.shape)
    print("Test data shape: ", test_data.X.shape)

    return train_data, test_data, train_y, test_y, gene_names

def prep_daeldg(trainingdatapath, source_list=['donorA', 'donorC', 'data6k', 'data8k', 'sdy67'], 
               priority_genes=[], n_samples=None, n_vtop=None):
    pbmc = sc.read_h5ad(trainingdatapath)
    pbmc = pbmc[pbmc.obs['ds'].isin(source_list)]

    # index for sample selection
    if n_samples is not None:
        np.random.seed(42)
        idx = np.random.choice(8000, n_samples, replace=False)
    
    # extract train data
    concat_list = []
    for source in source_list:
        if n_samples is not None:
            try:
                tmp_pbmc = pbmc[pbmc.obs['ds']==source][idx]
            except:
                tmp_pbmc = pbmc[pbmc.obs['ds']==source]
                print(f"Warning: {source} has less than {n_samples} samples, using all samples instead.")
        else:
            tmp_pbmc = pbmc[pbmc.obs['ds']==source]
        
        # transform to log2
        if source == 'GSE65133':
            tmp_pbmc.X = tmp_pbmc.X
        else:
            tmp_pbmc.X = np.log2(tmp_pbmc.X + 1)

        concat_list.append(tmp_pbmc)

    train = anndata.concat(concat_list)

    if n_vtop is None:
        #### variance cut off
        label = train.X.var(axis=0) > 0.1  # FIXME: use sdy67 as a reference ?
        label_idx = np.where(label)[0]
    else:
        #### top n_vtop highly variable genes
        label_idx = np.argsort(-train.X.var(axis=0))[:n_vtop]
    
    # add priority genes
    priority_label = np.array([True if gene in priority_genes else False for gene in train.var_names])
    priority_idx = np.where(priority_label)[0]
    print(f"Priority genes: {np.sum(priority_label)}/{len(priority_genes)} genes")
    label_idx = np.unique(np.concatenate([label_idx, priority_idx]))
    gene_names = train.var_names[label_idx]
    
    # gene selection
    train_data = train[:, label_idx]

    print("Data shape: ", train_data.X.shape)
    print("Domain information: ", train_data.obs['ds'].unique().tolist())

    return train_data, gene_names


def DAELdgAug(train_data):
    source_ds = train_data.obs['ds'].unique().tolist()
    domain_dict = {ds: i for i, ds in enumerate(source_ds)}

    data1, data2, y_prop, domains = [], [], [], []
    for i, ds in enumerate(source_ds):
        data = train_data[train_data.obs['ds'] == ds]
        x = torch.tensor(data.X).float()
        y = torch.tensor(data.obs[target_cells].values).float()

        # Add Gaussian noise to the input tensor for data augmentation
        data1.append(add_noise(x, noise1))
        data2.append(add_noise(x, noise2))
        y_prop.append(y)
        domains.append(torch.full((len(x),), domain_dict.get(ds), dtype=torch.long))

    data1 = torch.cat(data1)
    data2 = torch.cat(data2)
    y_prop = torch.cat(y_prop)
    domains = torch.cat(domains)

    return data1, data2, y_prop, domains


class DAELdg_Labeled(Dataset):
    def __init__(self, train_data, feats1, feats2, target_cells=None):
        """
        Args:
            train_data: AnnData object
            feats1: torch.Tensor (features after weak augmentation)
            feats2: torch.Tensor (features after strong augmentation)
            target_cells: list of target cell types
        """
        if target_cells is None:
            target_cells = ['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells']

        source_ds = train_data.obs['ds'].unique().tolist()
        self.domain_dict = {ds: i for i, ds in enumerate(source_ds)}
        self.feats1 = feats1
        self.feats2 = feats2

        train_data.obs = train_data.obs.reset_index(drop=True)

        self.indices = []
        self.y_prop = []
        self.domains = []

        for ds in source_ds:
            data = train_data[train_data.obs['ds'] == ds]
            idx = data.obs.index.values.tolist()  
            y = torch.tensor(data.obs[target_cells].values).float()
            domain = self.domain_dict[ds]

            self.indices.extend(idx)
            self.y_prop.append(y)
            self.domains.extend([domain] * len(idx))

        self.y_prop = torch.cat(self.y_prop, dim=0)
        self.domains = torch.tensor(self.domains, dtype=torch.long)

    def __len__(self):
        return len(self.y_prop)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.feats1[real_idx], self.feats2[real_idx], self.y_prop[idx], self.domains[idx]

def build_daeldg_loader(train_data, feats1, feats2, batch_size=128, shuffle=True, target_cells=None):
    daeldg_loder = DataLoader(DAELdg_Labeled(train_data, feats1, feats2, target_cells), batch_size=batch_size, shuffle=shuffle)

    return daeldg_loder


class DAELda_UnLabeled(Dataset):
    def __init__(self, train_data, feats1, feats2, target_cells=None):
        """
        Args:
            train_data: AnnData object
            feats1: torch.Tensor (features after weak augmentation)
            feats2: torch.Tensor (features after strong augmentation)
            target_cells: list of target cell types
        """
        if target_cells is None:
            target_cells = ['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells']

        all_ds = train_data.obs['ds'].unique().tolist()
        self.domain_dict = {ds: i for i, ds in enumerate(all_ds)}
        self.feats1 = feats1
        self.feats2 = feats2

        train_data.obs = train_data.obs.reset_index(drop=True)

        self.indices = []
        self.domains = []
        target_domain_counts = 0
        for ds in all_ds:
            data = train_data[train_data.obs['ds'] == ds]
            idx = data.obs.index.values.tolist()  
            y = torch.tensor(data.obs[target_cells].values).float()
            domain = self.domain_dict[ds]

            self.indices.extend(idx)
            self.domains.extend([domain] * len(idx))
            target_domain_counts += len(idx)

        self.domains = torch.tensor(self.domains, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.feats1[real_idx], self.feats2[real_idx], self.domains[idx]


def build_daelda_loader(train_data, feats1, feats2, batch_size=128, 
                        source_domains=['donorA', 'donorC', 'data6k', 'data8k'], target_domains=['sdy67'],
                        shuffle=True, target_cells=None):
    if target_cells is None:
        target_cells = ['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells']
    all_ds = train_data.obs['ds'].unique().tolist()
    train_data.obs = train_data.obs.reset_index(drop=True)

    # separate source and target domains
    source_domain_indices = []
    target_domain_indices = []
    for ds in all_ds:
        data = train_data[train_data.obs['ds'] == ds]
        idx = data.obs.index.values.tolist()
        if ds in source_domains:            
            source_domain_indices.extend(idx)
        elif ds in target_domains:
            target_domain_indices.extend(idx)
        else:
            raise ValueError(f"Unknown domain: {ds}")

    # source
    s_data = train_data[source_domain_indices]
    s_feats1 = feats1[source_domain_indices]
    s_feats2 = feats2[source_domain_indices]

    # target
    t_data = train_data[target_domain_indices]
    t_feats1 = feats1[target_domain_indices]
    t_feats2 = feats2[target_domain_indices]

    print(f"Source domain: {s_data.shape}, Target domain: {t_data.shape}")

    del train_data

    # build loaders
    labeled_loder = DataLoader(DAELdg_Labeled(s_data, s_feats1, s_feats2, target_cells), 
                                              batch_size=batch_size, shuffle=shuffle)
    unlabeled_loder = DataLoader(DAELda_UnLabeled(t_data, t_feats1, t_feats2, target_cells),    
                                 batch_size=batch_size, shuffle=shuffle)

    return labeled_loder, unlabeled_loder

def parse_batch_train(batch_x, batch_u, device):
    # labeled data
    input_x = batch_x[0]  # weak augmented
    input_x2 = batch_x[1]  # strong augmented
    prop_x = batch_x[2]
    domain_x = batch_x[3]

    # unlabeled data
    input_u = batch_u[0]  # weak augmented
    input_u2 = batch_u[1]  # strong augmented

    # device settings
    input_x = input_x.to(device)
    input_x2 = input_x2.to(device)
    prop_x = prop_x.to(device)
    domain_x = domain_x.to(device)
    input_u = input_u.to(device)
    input_u2 = input_u2.to(device)

    return input_x, input_x2, prop_x, domain_x, input_u, input_u2


def add_noise(tensor: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to the input tensor for data augmentation.

    Args:
        tensor (torch.Tensor): Input tensor (e.g., shape [2060, 256])
        noise_std (float): Standard deviation (strength) of the noise.

    Returns:
        torch.Tensor: Tensor after adding noise.
    """
    if noise_std <= 0:
        # If noise strength is zero or negative, return the original tensor
        return tensor

    noise = torch.randn_like(tensor) * noise_std
    augmented = tensor + noise
    return augmented
