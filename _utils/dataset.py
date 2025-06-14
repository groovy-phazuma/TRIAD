# -*- coding: utf-8 -*-
"""
Created on 2025-06-13 (Fri) 16:47:01

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'

import os
os.chdir(BASE_DIR)

import gc
import random
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData
from anndata import read_h5ad
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as Data
import torch.backends.cudnn as cudnn


def prep4benchmark(h5ad_path, source_list=['data6k'], target='sdy67', priority_genes=[], 
                   target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], 
                   n_samples=None, n_vtop=None, mm_scale=False, seed=42):
    print(f"Source domain: {source_list}")
    print(f"Target domain: {target}")

    pbmc = sc.read_h5ad(h5ad_path)
    test = pbmc[pbmc.obs['ds'] == target]

    train, label_idx = extract_variable_sources(pbmc, source_list, n_samples=n_samples, n_vtop=n_vtop, seed=seed)
    train_data, test_data, train_y, gene_names = finalize_data(train, test, label_idx, target_cells, priority_genes, log_conv=(target != 'GSE65133'), mm_scale=mm_scale)
    test_y = test.obs[target_cells]

    return train_data, test_data, train_y, test_y, gene_names

def prep4inference(h5ad_path, target_path, source_list=['data6k'], target='TSCA_Lung', priority_genes=[], 
             target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], 
             n_samples=None, n_vtop=None, target_log_conv=True, mm_scale=False, seed=42):

    pbmc = sc.read_h5ad(h5ad_path)
    target_df = pd.read_csv(target_path, index_col=0)

    # Match gene names
    target_df.index = target_df.index.str.upper()
    target_genes = target_df.index
    source_genes = pbmc.var_names.str.upper()
    common_genes = target_genes.intersection(source_genes)
    target_df_filtered = target_df.loc[common_genes]

    target_adata = AnnData(X=target_df_filtered.T.values)
    target_adata.var_names = target_df_filtered.index
    target_adata.obs_names = target_df_filtered.columns
    target_adata.obs['ds'] = target

    print("Target data shape: ", target_adata.X.shape)
    if len(target_adata.obs_names) == 0:
        raise ValueError("No cells found in target data. Please check the input data.")

     # add obs columns from source_pbmc to target_adata
    for col in pbmc.obs.columns:
        if col not in target_adata.obs:
            target_adata.obs[col] = target if col == "ds" else (2 if col == "batch" else np.nan)
    
    combined_adata = sc.concat([pbmc, target_adata], join='inner', merge='first')
    test = combined_adata[combined_adata.obs['ds'] == target]

    train, label_idx = extract_variable_sources(combined_adata, source_list, n_samples=n_samples, n_vtop=n_vtop, seed=seed)
    train_data, test_data, train_y, gene_names = finalize_data(train, test, label_idx, target_cells, priority_genes, log_conv=target_log_conv, mm_scale=mm_scale)

    return train_data, test_data, train_y, gene_names

def extract_variable_sources_legacy(pbmc, source_list, n_samples=None, n_vtop=None, seed=42):
    if n_samples is not None:
        np.random.seed(seed)
        idx = np.random.choice(8000, n_samples, replace=False)
    else:
        idx = None

    def select_and_vtop(ds_name):
        data = pbmc[pbmc.obs['ds'] == ds_name]
        if idx is not None:
            data = data[idx]
        return data, calc_vtop(data, n_vtop=n_vtop)

    data_map = {}
    for ds_name in ['donorA', 'donorC', 'data6k', 'data8k']:
        data, idx_vtop = select_and_vtop(ds_name)
        data_map[ds_name] = (data, idx_vtop)

    train = None
    label_idx = np.array([], dtype=int)
    for s_name in source_list:
        if s_name in data_map:
            current_data, current_idx = data_map[s_name]
            train = current_data if train is None else anndata.concat([train, current_data])
            label_idx = np.unique(np.concatenate([label_idx, current_idx]))
        else:
            print(f"Warning: '{s_name}' not found in data_map. Skipping.")
    
    return train, label_idx

def extract_variable_sources(pbmc, source_list, n_samples=None, n_vtop=None, seed=42):
    # 1. Concatenate data from specified sources
    train = None
    for s_name in source_list:
        data = pbmc[pbmc.obs['ds'] == s_name]
        if n_samples is not None:
            np.random.seed(seed)
            idx = np.random.choice(data.shape[0], n_samples, replace=False)
            data = data[idx]
        train = data if train is None else anndata.concat([train, data])

    # 2. Calculate highly variable genes
    if n_vtop is None:
        label = train.X.var(axis=0) > 0.1
        label_idx = np.where(label)[0]
    else:
        label_idx = np.argsort(-train.X.var(axis=0))[:n_vtop]

    return train, label_idx

def finalize_data(train, test, label_idx, target_cells, priority_genes=[], log_conv=True, mm_scale=False):
    priority_label = np.array([gene in priority_genes for gene in train.var_names])
    priority_idx = np.where(priority_label)[0]
    print(f"Priority genes: {np.sum(priority_label)}/{len(priority_genes)} genes")

    label_idx = np.unique(np.concatenate([label_idx, priority_idx]))
    gene_names = train.var_names[label_idx]

    train_data = train[:, label_idx].copy()
    train_data.X = np.log2(train_data.X + 1)

    test_data = test[:, label_idx].copy()
    if log_conv:
        print("Applying log2 transformation...")
        test_data.X = np.log2(test_data.X + 1)
    
    # min-max scaling
    if mm_scale:
        print("Applying Min-Max scaling...")
        mms = MinMaxScaler()
        train_data.X = mms.fit_transform(train_data.X.T).T
        test_data.X = mms.fit_transform(test_data.X.T).T

    print("Train data shape: ", train_data.X.shape)
    print("Test data shape: ", test_data.X.shape)

    return train_data, test_data, train.obs[target_cells], gene_names

def calc_vtop(train, n_vtop=1000):
    """
    Calculate the top n_vtop highly variable genes from the training data.
    
    Parameters:
    - train: AnnData object containing the training data.
    - n_vtop: Number of top variable genes to select.
    
    Returns:
    - label_idx: Indices of the top n_vtop variable genes.
    """
    if n_vtop is None:
        # variance cut off
        label = train.X.var(axis=0) > 0.1 
        label_idx = np.where(label)[0]
    else:
        # top n_vtop highly variable genes
        label_idx = np.argsort(-train.X.var(axis=0))[:n_vtop]
    
    return label_idx

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
