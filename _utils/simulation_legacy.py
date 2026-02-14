# -*- coding: utf-8 -*-
"""
Created on 2024-08-05 (Mon) 20:14:24

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/cluster/HDD/azuma/TopicModel_Deconv'

import gc
import random
import collections
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# %% Proportion Generator
sample_size = 15000
immune_cells = ['NK','T_CD4','T_CD8_CytT','Monocyte','Mast_cells']
non_immune_cells = ['Fibroblast','Ciliated','Alveolar_Type1','Alveolar_Type2']

# Immune
final_res = []
for idx in range(sample_size):
    np.random.seed(seed=idx)
    p_list = np.random.rand(len(immune_cells))
    norm_p_list = list(p_list / sum(p_list)) # sum to 1
    final_res.append(norm_p_list)
summary_df = pd.DataFrame(final_res,columns=immune_cells)
summary_df.to_csv(BASE_DIR+'/datasource/Simulated_Data/Lung/sc_immune_proportion_df.csv')

# Non-immune
final_res = []
for idx in range(sample_size):
    np.random.seed(seed=1000000-idx)
    p_list = np.random.rand(len(non_immune_cells))
    norm_p_list = list(p_list / sum(p_list)) # sum to 1
    final_res.append(norm_p_list)

summary_df = pd.DataFrame(final_res,columns=non_immune_cells)
summary_df.to_csv(BASE_DIR+'/datasource/Simulated_Data/Lung/sc_nonimmune_proportion_df.csv')
gc.collect()

# %% Immune
raw_data = sc.read_h5ad(BASE_DIR+"/datasource/scRNASeq/Tissue_Stability_Cell_Atlas/lung.cellxgene.h5ad")
info_df = raw_data.obs

# Immune cell types
immune_cells = ['NK','T_CD4','T_CD8_CytT','Monocyte','Mast_cells']
immune_summary = pd.read_csv(BASE_DIR+'/datasource/Simulated_Data/Lung/sc_immune_proportion_df.csv',index_col=0)
for cell in immune_cells:
    # Extract info
    target_cell = info_df[info_df['Celltypes_updated_July_2020']==cell]
    target_idx = [info_df.index.tolist().index(t) for t in target_cell.index.tolist()]
    cell_size = len(target_idx)
    print("{}: {} cells are detected".format(cell,cell_size))

    # Train / Test Split
    random.seed(123)
    shuffle_idx = random.sample(target_idx,cell_size)
    train_idx = shuffle_idx[0:int(cell_size*0.7)]
    test_idx = shuffle_idx[int(cell_size*0.7):]

    #train_exp = np.array(raw_data.X[train_idx,:].todense())

    pd.to_pickle(train_idx,BASE_DIR+'/datasource/Simulated_Data/Lung/cell_idxs/{}_train_idx.pkl'.format(cell))
    pd.to_pickle(test_idx,BASE_DIR+'/datasource/Simulated_Data/Lung/cell_idxs/{}_test_idx.pkl'.format(cell))
gc.collect()

pool_size = 500
# Pool
pooled_idx = []
for idx in range(len(immune_summary)):
    p_list = immune_summary.iloc[idx,:].tolist()
    final_idx = []
    for j in range(len(p_list)):
        cell = immune_cells[j]
        p = p_list[j]
        tmp_size = int(pool_size*p)

        train_idx = pd.read_pickle(BASE_DIR+'/datasource/Simulated_Data/Lung/cell_idxs/{}_train_idx.pkl'.format(cell))

        if len(train_idx) > tmp_size:
            random.seed(idx*j)
            tmp_idx = random.sample(train_idx, tmp_size)
        else:
            random.seed(idx*j)
            tmp_idx = random.choices(train_idx, k=tmp_size)

        final_idx.extend(tmp_idx)
    pooled_idx.append(final_idx)
gc.collect()

# Exp
raw_df = np.array(raw_data.X.todense())
pooled_exp = []
for exp_idx in tqdm(pooled_idx):
    tmp_exp = raw_df[exp_idx,:]
    tmp_sum = tmp_exp.sum(axis=0)
    pooled_exp.append(tmp_sum)
exp_df = pd.DataFrame(pooled_exp).T
exp_df.index = raw_data.var['gene_ids-HCATisStab7509734'].tolist()

exp_df.to_csv(BASE_DIR+'/datasource/Simulated_Data/Lung/exp/immune_25204x15000.csv')  # 240805
