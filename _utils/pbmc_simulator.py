# -*- coding: utf-8 -*-
"""
Created on 2024-11-27 (Wed) 21:52:13

@author: I.Azuma
"""
# %%
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

# %%
class MyPBMC_Simulator():
    def __init__(self, adata, adata_counts, cell_idx_dict=None, sample_size=8000):
        self.adata = adata
        self.adata_counts = adata_counts
        self.sample_size = sample_size

        cell_types = sorted(self.adata.obs['celltype'].unique().tolist())
        self.cell_types = cell_types

        if cell_idx_dict is not None:
            self.cell_idx_dict = cell_idx_dict
        else:
            raw_idx = self.adata.obs.index.tolist()
            cell_idx = {}
            for c in self.cell_types:
                tmp_idx = self.adata.obs[self.adata.obs['celltype']==c].index.tolist()
                n_idx = [raw_idx.index(i) for i in tmp_idx]
                cell_idx[c] = n_idx
            self.cell_idx_dict = cell_idx
    
    def assign_proportion(self, sparse=True):
        final_res = []
        for idx in range(self.sample_size):
            if sparse:
                # select consisting cell types from cell_types at random
                np.random.seed(seed=idx)
                use_cell_types = np.random.choice(self.cell_types, size=np.random.randint(1,len(self.cell_types)+1), replace=False)
                p_list = np.random.rand(len(use_cell_types))

                # assign random proportion to each cell type
                final_p_list = [0]*len(self.cell_types)
                for j, c in enumerate(use_cell_types):
                    final_p_list[self.cell_types.index(c)] = p_list[j]
                
                norm_p_list = list(final_p_list / sum(final_p_list)) # sum to 1
                final_res.append(norm_p_list)
            else:
                np.random.seed(seed=idx)
                p_list = np.random.rand(len(self.cell_types))
                norm_p_list = list(p_list / sum(p_list)) # sum to 1
                final_res.append(norm_p_list)
        summary_df = pd.DataFrame(final_res,columns=self.cell_types)

        return summary_df
    
    def create_simulation(self, summary_df, pool_size=500):
        rs = 0
        pooled_exp = []
        for idx in tqdm(range(len(summary_df))):
            p_list = summary_df.iloc[idx].tolist()
            final_idx = []
            for j, p in enumerate(p_list):
                cell = self.cell_types[j]
                tmp_size = int(pool_size*p)  # number of cells to be selected

                candi_idx = self.cell_idx_dict[cell]
                # select tmp_size from tmp_df at random
                np.random.seed(seed=rs)
                if len(candi_idx) < tmp_size:
                    select_idx = np.random.choice(candi_idx, size=tmp_size, replace=True)
                else:
                    select_idx = np.random.choice(candi_idx, size=tmp_size, replace=False)
            
                assert len(self.adata.X[select_idx]) == tmp_size
                final_idx.extend(select_idx)
            
            # QC
            if len(final_idx) < (pool_size - len(self.cell_types)) or len(final_idx) > (pool_size + len(self.cell_types)):
                print("Error: {} cells are selected".format(len(final_idx)))
                break

            # sum up the expression (counts)
            tmp_sum = list(np.array(self.adata_counts.X[final_idx].sum(axis=0))[0])
            pooled_exp.append(tmp_sum)

        exp_df = pd.DataFrame(pooled_exp).T

        return exp_df


        

