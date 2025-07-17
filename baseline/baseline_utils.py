#!/usr/bin/env python3
"""
Created on 2025-07-17 (Thu) 21:56:49

@author: I.Azuma
"""
# %%
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from anndata import AnnData
from anndata import read_h5ad
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prep4pbmc(h5ad_path, target='sdy67', test_ratio=0.2, target_path=None):
    pbmc = read_h5ad(h5ad_path)
    pbmc1 = pbmc[pbmc.obs['ds']=='sdy67']
    microarray = pbmc[pbmc.obs['ds']=='GSE65133']
    
    donorA = pbmc[pbmc.obs['ds']=='donorA']
    donorC = pbmc[pbmc.obs['ds']=='donorC']
    data6k = pbmc[pbmc.obs['ds']=='data6k']
    data8k = pbmc[pbmc.obs['ds']=='data8k']
    
    if target == 'sdy67':
        test = pbmc1
        train = anndata.concat([donorA,donorC,data6k,data8k])
    elif target == 'GSE65133':
        test = microarray
        train = anndata.concat([donorA,donorC,data6k,data8k,pbmc1])  # FIXME: pbmc1 is included
    else:
        if target_path is None:
            raise ValueError("Please provide target_path for inference mode.")
        print(f"Target domain: {target}")
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
        target_adata.obs = target_adata.obs[pbmc.obs.columns]

        # train and test definition
        gene_map = dict(zip(pbmc.var_names.str.upper(), pbmc.var_names)) 
        ordered_genes_in_pbmc = [gene_map[gene] for gene in target_df_filtered.index if gene in gene_map]
        source_domains = ['donorA', 'donorC', 'data6k', 'data8k']
        train = pbmc[pbmc.obs['ds'].isin(source_domains), ordered_genes_in_pbmc]
        test = target_adata
        
    train_y = train.obs.iloc[:,:-2].values
    test_y = test.obs.iloc[:,:-2].values

    print(train.obs.head())
    print(test.obs.head())

    #### variance cut off
    if target == 'sdy67':
        label = test.X.var(axis=0) > 0.1  # NOTE: not train
    elif target == 'GSE65133':
        label = test.X.var(axis=0) > 0.01
    else:
        label = test.X.var(axis=0) > 0.1
    test_x = np.zeros((test.X.shape[0],np.sum(label)))
    train_x = np.zeros((train.X.shape[0],np.sum(label)))

    test_x = test.X[:,label]
    train_x = train.X[:,label]
    
    train_x = np.log2(train_x + 1)
    test_x = np.log2(test_x + 1)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=test_ratio)

    print("Start scaling data...")
    mms = MinMaxScaler()
    test_x = mms.fit_transform(test_x.T)
    test_x = test_x.T
    train_x = mms.fit_transform(train_x.T)
    train_x = train_x.T
    val_x = mms.fit_transform(val_x.T)
    val_x = val_x.T


    print(f"Train data shape: {train_x.shape}")
    print(f"Test data shape: {test_x.shape}")

    return train_x, val_x, test_x, train_y, val_y, test_y

def prep4tissue(h5ad_path, target_path, source_list=['GSE139107'], target='TSCA_Lung',
             target_cells=['NK', 'T_CD4', 'T_CD8_CytT', 'Monocyte', 'Mast_cells', 'Fibroblast',
       'Ciliated', 'Alveolar_Type1', 'Alveolar_Type2'], test_ratio=0.2):
    if target_path is None:
        raise ValueError("Please provide target_path for inference mode.")
    
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
    target_adata.obs = target_adata.obs[source_adata.obs.columns]
    
    # train and test definition
    gene_map = dict(zip(source_adata.var_names.str.upper(), source_adata.var_names)) 
    ordered_genes_in_tissue = [gene_map[gene] for gene in target_df_filtered.index if gene in gene_map]
    train = source_adata[source_adata.obs['ds'].isin(source_list), ordered_genes_in_tissue]
    test = target_adata

    train_y = train.obs[target_cells].values
    test_y = test.obs[target_cells].values

    print(train.obs.head())
    print(test.obs.head())

    #### variance cut off
    label = test.X.var(axis=0) > 0.1
    test_x = np.zeros((test.X.shape[0],np.sum(label)))
    train_x = np.zeros((train.X.shape[0],np.sum(label)))

    test_x = test.X[:,label]
    train_x = train.X[:,label]
    
    train_x = np.log2(train_x + 1)
    test_x = np.log2(test_x + 1)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=test_ratio)

    print("Start scaling data...")
    mms = MinMaxScaler()
    test_x = mms.fit_transform(test_x.T)
    test_x = test_x.T
    train_x = mms.fit_transform(train_x.T)
    train_x = train_x.T
    val_x = mms.fit_transform(val_x.T)
    val_x = val_x.T

    print(f"Train data shape: {train_x.shape}")
    print(f"Test data shape: {test_x.shape}")

    return train_x, val_x, test_x, train_y, val_y, test_y
