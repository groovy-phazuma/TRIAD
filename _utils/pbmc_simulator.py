# -*- coding: utf-8 -*-
"""
Created on 2024-11-27 (Wed) 21:52:13

@author: I.Azuma
"""
# %%
BASE_DIR = "/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv"

import gc
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys

sys.path.append(BASE_DIR+'/github/LiverDeconv')
import liver_deconv as ld

sys.path.append(BASE_DIR+'/github/GLDADec')
from _utils import plot_utils as pu

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
    
    def assign_proportion_uniform(self, sparse=True):
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
        self.summary_df = summary_df

        return summary_df
    
    def assign_proportion_dirichlet(self, a=1.0, do_viz=False):
        alpha = [a]*len(self.cell_types)

        np.random.seed(seed=0)
        data = np.random.dirichlet(alpha, size=self.sample_size)

        if do_viz:
            # visualization
            plt.hist(data[:, 1], bins=50, alpha=0.7, color='blue', label=f'a={a}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of proportion')
            plt.legend()
            plt.show()
        
        summary_df = pd.DataFrame(data,columns=self.cell_types)
        self.summary_df = summary_df

        return summary_df
    
    def create_sim_bulk(self, summary_df=None, pool_size=500):
        rs = 0
        pooled_exp = []
        if summary_df is None:
            summary_df = self.summary_df
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

        bulk_df = pd.DataFrame(pooled_exp).T
        bulk_df.index = self.adata_counts.var_names

        return bulk_df

    def create_ref(self):
        pooled_exp = []
        for i,k in enumerate(self.cell_idx_dict):
            c_idx = self.cell_idx_dict[k]
            tmp_mean = np.array(self.adata_counts.X[c_idx].mean(axis=0))[0]
            pooled_exp.append(tmp_mean)

        ref_df = pd.DataFrame(pooled_exp).T
        ref_df.index = self.adata_counts.var_names  # gene names
        ref_df.columns = self.cell_idx_dict.keys()  # cell types
    
        return ref_df

    def bulk_ref_qc(self, bulk_df, ref_df, summary_df):
        # preprocessing
        bulk_df.index = [t.upper() for t in bulk_df.index] 
        ref_df.index = [t.upper() for t in ref_df.index]
        bulk_df = np.log1p(bulk_df)
        ref_df = np.log1p(ref_df)

        # ElasticNet deconvolution
        dat = ld.LiverDeconv()
        dat.set_data(df_mix=bulk_df, df_all=ref_df)
        dat.pre_processing(do_ann=False,ann_df=None,do_log2=False,do_quantile=False,do_trimming=False,do_drop=True)
        dat.narrow_intersec()

        dat.create_ref(sep="",number=100,limit_CV=1,limit_FC=0.1,log2=False,verbose=True,do_plot=True)

        dat.do_fit()
        res = dat.get_res()
        norm_res = res.div(res.sum(axis=1),axis=0)  # normalize to sum to 1

        # visualize the result
        for target_cell in ref_df.columns:
            eval_deconv(dec_name=[target_cell], val_name=[target_cell], color='tab:blue', deconv_df=norm_res, y_df=summary_df)


def eval_deconv(dec_name = [0,7], val_name = ["Monocytes"], color='tab:blue', deconv_df=None, y_df=None):
    # overall
    plot_dat = pu.DeconvPlot(deconv_df=deconv_df,val_df=y_df,dec_name=dec_name,val_name=val_name,plot_size=20,dpi=100)
    res = plot_dat.plot_simple_corr(color=color,title=f'Topic 0:{dec_name} vs {val_name}',target_samples=None)
