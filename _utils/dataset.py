# -*- coding: utf-8 -*-
"""
Created on 2025-06-13 (Fri) 16:47:01

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/cluster/HDD/azuma/TopicModel_Deconv'

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
from scipy.sparse import vstack, issparse
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.utils.data as Data
import torch.backends.cudnn as cudnn


def prep4benchmark(h5ad_path, source_list=['data6k'], target='sdy67', priority_genes=[], 
                   target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], 
                   n_samples=None, n_vtop=None, mm_scale=False, seed=42, vtop_mode='train'):
    print(f"Source domain: {source_list}")
    print(f"Target domain: {target}")

    pbmc = sc.read_h5ad(h5ad_path)
    test = pbmc[pbmc.obs['ds'] == target]

    train, label_idx = extract_variable_sources(pbmc, source_list, target, n_samples=n_samples, n_vtop=n_vtop, seed=seed, vtop_mode=vtop_mode)
    train_data, test_data, train_y, gene_names = finalize_data(train, test, label_idx, target_cells, priority_genes, log_conv=(target != 'GSE65133'), mm_scale=mm_scale)
    test_y = test.obs[target_cells]

    return train_data, test_data, train_y, test_y, gene_names

def prep4inference(h5ad_path, target_path, source_list=['data6k'], target='TSCA_Lung', priority_genes=[], 
             target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], 
             n_samples=None, n_vtop=None, target_log_conv=True, mm_scale=False, seed=42, vtop_mode='train'):

    source_adata = sc.read_h5ad(h5ad_path)
    target_df = pd.read_csv(target_path, index_col=0)

    # Match gene names
    target_df.index = target_df.index.str.upper()
    target_genes = target_df.index
    source_adata.var_names = source_adata.var_names.str.upper()
    source_genes = source_adata.var_names
    common_genes = target_genes.intersection(source_genes)
    target_df_filtered = target_df.loc[common_genes]
    if len(common_genes) < 100:
        print(f"Warning: Only {len(common_genes)} common genes found between source and target datasets. This may affect model performance.")

    target_adata = AnnData(X=target_df_filtered.T.values)
    target_adata.var_names = target_df_filtered.index
    target_adata.obs_names = target_df_filtered.columns
    target_adata.obs['ds'] = target

    print("Target data shape: ", target_adata.X.shape)
    if len(target_adata.obs_names) == 0:
        raise ValueError("No cells found in target data. Please check the input data.")

     # add obs columns from source_adata to target_adata
    for col in source_adata.obs.columns:
        if col not in target_adata.obs:
            target_adata.obs[col] = target if col == "ds" else (2 if col == "batch" else np.nan)
    
    combined_adata = sc.concat([source_adata, target_adata], join='inner', merge='first')
    test = combined_adata[combined_adata.obs['ds'] == target]

    train, label_idx = extract_variable_sources(combined_adata, source_list, target, n_samples=n_samples, n_vtop=n_vtop, seed=seed, vtop_mode=vtop_mode)
    train_data, test_data, train_y, gene_names = finalize_data(train, test, label_idx, target_cells, priority_genes, log_conv=target_log_conv, mm_scale=mm_scale)

    return train_data, test_data, train_y, gene_names

def extract_variable_sources(pbmc, source_list, target, n_samples=None, n_vtop=None, seed=42, vtop_mode='train'):
    # 1. Concatenate data from specified sources
    train = None
    for s_name in source_list:
        data = pbmc[pbmc.obs['ds'] == s_name]
        if n_samples is not None:
            np.random.seed(seed)
            idx = np.random.choice(data.shape[0], n_samples, replace=False)
            data = data[idx]
        train = data if train is None else anndata.concat([train, data])
    
    test = pbmc[pbmc.obs['ds'] == target]

    # 2. Calculate highly variable genes
    def get_var_indices(X, top_n):
        # Handle sparse matrix
        if issparse(X):
            X = X.toarray()
        if top_n is None:
            return np.where(X.var(axis=0) > 0.1)[0]
        return np.argsort(-X.var(axis=0))[:top_n]

    if vtop_mode == 'train':
        label_idx = get_var_indices(train.X, n_vtop)

    elif vtop_mode == 'test':
        if test is None:
            raise ValueError("test data must be provided when mode is 'test'")
        label_idx = get_var_indices(test.X, n_vtop)

    elif vtop_mode == 'both':
        if test is None:
            raise ValueError("test data must be provided when mode is 'both'")
        train_idx = get_var_indices(train.X, n_vtop)
        test_idx = get_var_indices(test.X, n_vtop)
        label_idx = np.unique(np.concatenate([train_idx, test_idx]))

    else:
        raise ValueError("mode must be one of ['train', 'test', 'both']")

    return train, label_idx

def calc_vtop(train, test=None, n_vtop=1000, mode='train'):

    if mode not in ['train', 'test', 'both']:
        raise ValueError("mode must be one of ['train', 'test', 'both']")

    if mode in ['test', 'both'] and test is None:
        raise ValueError("test data must be provided when mode is 'test' or 'both'")

    if mode == 'train':
        X = train.X
    elif mode == 'test':
        X = test.X
    elif mode == 'both':
        X = vstack([train.X, test.X]) if issparse(train.X) else np.vstack([train.X, test.X])

    # Convert sparse to dense if needed
    if issparse(X):
        X = X.toarray()

    if n_vtop is None:
        label = X.var(axis=0) > 0.1
        label_idx = np.where(label)[0]
    else:
        label_idx = np.argsort(-X.var(axis=0))[:n_vtop]

    return label_idx


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
