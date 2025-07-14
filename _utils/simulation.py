# -*- coding: utf-8 -*-
"""
Created on 2024-11-27 (Wed) 21:52:13

@author: I.Azuma
"""
# %%
BASE_DIR = "/workspace/cluster/HDD/azuma/TopicModel_Deconv"

import os
import gc
import random
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

sys.path.append(BASE_DIR+'/github/deconv-utils')
from src import evaluation as ev

# %%
class BaseSimulator():
    def __init__(self,sample_size, cell_idx, adata=None):
        self.sample_size = sample_size
        self.cell_idx = cell_idx
        self.adata = adata
    
    def assign_uniform(self, cell_types:list, sparse=True):
        final_res = []
        for idx in range(self.sample_size):
            if sparse:
                # select consisting cell types from cell_types at random
                np.random.seed(seed=idx)
                use_cell_types = np.random.choice(cell_types, size=np.random.randint(1,len(cell_types)+1), replace=False)
                p_list = np.random.rand(len(use_cell_types))

                # assign random proportion to each cell type
                final_p_list = [0]*len(cell_types)
                for j, c in enumerate(use_cell_types):
                    final_p_list[cell_types.index(c)] = p_list[j]
                
                norm_p_list = list(final_p_list / sum(final_p_list)) # sum to 1
                final_res.append(norm_p_list)
            else:
                np.random.seed(seed=idx)
                p_list = np.random.rand(len(cell_types))
                norm_p_list = list(p_list / sum(p_list)) # sum to 1
                final_res.append(norm_p_list)
        summary_df = pd.DataFrame(final_res,columns=cell_types)

        return summary_df
    
    def assign_dirichlet(self, cell_types:list, a=1.0, do_viz=False):
        alpha = [a]*len(cell_types)

        np.random.seed(seed=42)
        data = np.random.dirichlet(alpha, size=self.sample_size)

        if do_viz:
            # visualization
            plt.hist(data[:, 1], bins=50, alpha=0.7, color='blue', label=f'a={a}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of proportion')
            plt.legend()
            plt.show()
        
        summary_df = pd.DataFrame(data,columns=cell_types)

        return summary_df

    def create_ref(self):
        if hasattr(self.adata.X, 'todense'):
            raw_exp = np.array(self.adata.X.todense())
        else:
            raw_exp = np.array(self.adata.X)
            
        pooled_exp = []
        for i,k in enumerate(self.cell_idx_dict):
            c_idx = self.cell_idx_dict[k]['train']
            tmp_mean = raw_exp[c_idx].mean(axis=0)
            pooled_exp.append(tmp_mean)

        ref_df = pd.DataFrame(pooled_exp).T
        ref_df.index = self.adata.var_names  # gene names
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
            res = ev.eval_deconv(dec_name_list=[[target_cell]], val_name_list=[[target_cell]], deconv_df=norm_res, y_df=summary_df, do_plot=True)

class GSE139107_Simulator(BaseSimulator):
    def __init__(self, sample_size=8000, method='dirichlet'):
        self.sample_size = sample_size
        self.method = method
        self.immune_cells = ['NK','T_CD4','T_CD8_CytT','Monocyte','Mast_cells']
        self.non_immune_cells = ['Fibroblast','Ciliated','Alveolar_Type1','Alveolar_Type2']

        self.summary_df = None
        self.cell_idx_dict = None

        self.filter_dict = {
            'NK': 'NK',
            'T_CD4': 'CD4',
            'T_CD8_CytT':'CD8',
            'Monocyte':'Monocyte',
            'Mast_cells':'MAST',
            'Fibroblast':'Fibrosis',
            'Ciliated': 'Ciliated',
            'Alveolar_Type1': 'AT1',
            'Alveolar_Type2': 'AT2'
        }
    
    def filter_exp(self):
        """
        #target_cells = ['NK']
        #target_cells = ['Monocytes']
        #target_cells = ['CD4+ Th', 'Naive CD4+ T', ]
        #target_cells = ['Cytotoxic CD8+ T', 'Naive CD8+ T']
        #target_cells = ['MAST']
        #target_cells = ['Myofibroblasts']
        #target_cells = ['Ciliated']
        #target_cells = ['AT1']
        #target_cells = ['AT2']
        """
        info_df = pd.read_table(BASE_DIR+'/datasource/scRNASeq/GSE131907/GSE131907_Lung_Cancer_cell_annotation.txt')
        lung_info = info_df[info_df['Sample_Origin'].isin(['nLung', 'tLung'])]
        cell_types = lung_info['Cell_type'].unique().tolist()
        sub_types = sorted(lung_info['Cell_subtype'].dropna().unique().tolist())

        target_cells = ['NK']
        target_samples = lung_info[lung_info['Cell_subtype'].isin(target_cells)]['Index'].tolist()
        input_file = BASE_DIR+'/datasource/scRNASeq/GSE131907/GSE131907_Lung_Cancer_raw_UMI_matrix.txt'
        output_file = BASE_DIR+'/datasource/Simulated_Data/GSE139107/signature/NK_filtered_matrix.txt'
        with open(input_file, 'r') as fin:
            header = fin.readline().rstrip('\n').split('\t')
            
            # target columns ('Gene' + selected samples)
            selected_idx = [0] + [i for i, col in enumerate(header) if col in target_samples]
            
            # write
            with open(output_file, 'w') as fout:
                fout.write('\t'.join([header[i] for i in selected_idx]) + '\n')
                for line in fin:
                    values = line.rstrip('\n').split('\t')
                    selected_values = [values[i] for i in selected_idx]
                    fout.write('\t'.join(selected_values) + '\n')
    
    def assign(self):
        assert self.method in ['uniform_sparse', 'uniform', 'dirichlet'], "Method not supported. Choose 'uniform_sparse', 'uniform', or 'dirichlet'."
        if self.method == 'uniform_sparse':
            # assign uniform distribution
            im_summary = self.assign_uniform(cell_types=self.immune_cells, sparse=True)
            non_im_summary = self.assign_uniform(cell_types=self.non_immune_cells, sparse=True)
        elif self.method == 'uniform':
            # assign uniform distribution
            im_summary = self.assign_uniform(cell_types=self.immune_cells, sparse=False)
            non_im_summary = self.assign_uniform(cell_types=self.non_immune_cells, sparse=False)
        elif self.method == 'dirichlet':
            im_summary = self.assign_dirichlet(a=1.0, cell_types=self.immune_cells)
            non_im_summary = self.assign_dirichlet(a=1.0, cell_types=self.non_immune_cells)
        else:
            raise ValueError("Method not supported. Choose 'uniform_sparse', 'uniform', or 'dirichlet'.")

        # normalize (sum to 1)
        self.im_summary = im_summary.div(im_summary.sum(axis=1), axis=0)
        self.non_im_summary = non_im_summary.div(non_im_summary.sum(axis=1), axis=0)

        summary_df = pd.concat([self.im_summary, self.non_im_summary], axis=1)
        self.summary_df = summary_df.div(summary_df.sum(axis=1), axis=0)  # normalize to sum to 1
    
    def set_data(self, summary_df=None, cell_idx_dict=None):
        if summary_df is not None:
            self.summary_df = summary_df
        if cell_idx_dict is not None:
            self.cell_idx_dict = cell_idx_dict
    
    def split_cell_idx(self, save_dir='./data/cell_idx', train_ratio=0.7):
        # Extract info
        cell_types = self.immune_cells + self.non_immune_cells
        cell_idx_dict = {}
        for cell in cell_types:
            cellname = self.filter_dict[cell]
            df = pd.read_table(BASE_DIR+f'/datasource/Simulated_Data/GSE139107/signature/{cellname}_filtered_matrix.txt', index_col=0)
            target_idx = [i for i in range(df.shape[1])]
            cell_size = len(target_idx)
            print("{}: {} cells are detected".format(cell, cell_size))
            # Train / Test Split
            random.seed(42)
            shuffle_idx = random.sample(target_idx, cell_size)
            train_idx = shuffle_idx[0:int(cell_size * train_ratio)]
            test_idx = shuffle_idx[int(cell_size * train_ratio):]
            cell_idx_dict[cell] = {'train': train_idx, 'test': test_idx}

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
            pd.to_pickle(train_idx, os.path.join(save_dir, f'{cell}_train_idx.pkl'))
            pd.to_pickle(test_idx, os.path.join(save_dir, f'{cell}_test_idx.pkl'))
        
        self.cell_idx_dict = cell_idx_dict
    
    def create_sim_bulk(self, pool_size=500, mode='train'):
        summary_df = self.summary_df
        cell_idx_dict = self.cell_idx_dict
        tototal_cells = summary_df.columns.tolist()

        pooled_exp = []
        np.random.seed(42)  # Set seed once outside the loop
        
        for idx in tqdm(range(len(summary_df))):
            p_list = summary_df.iloc[idx].values  # Use .values instead of .tolist() for speed
            for j, p in enumerate(p_list):
                cell = tototal_cells[j]
                tmp_size = int(pool_size * p)
                candi_idx = cell_idx_dict[cell][mode]
                cellname = self.filter_dict[cell]

                # Use different random state for each sample to maintain reproducibility
                rng = np.random.RandomState(42 + idx * len(tototal_cells) + j)
                if len(candi_idx) < tmp_size:
                    select_idx = rng.choice(candi_idx, size=tmp_size, replace=True)
                else:
                    select_idx = rng.choice(candi_idx, size=tmp_size, replace=False)
                
                df = pd.read_table(BASE_DIR+f'/datasource/Simulated_Data/GSE139107/signature/{cellname}_filtered_matrix.txt', index_col=0, usecols=[0] + select_idx)
                tmp_sum = df.sum(axis=1).values  # sum across selected cells
                pooled_exp.append(tmp_sum.flatten())  # Ensure 1D array
        
        # Convert to DataFrame more efficiently
        pooled_exp = np.array(pooled_exp)
        bulk_df = pd.DataFrame(pooled_exp.T)
        bulk_df.index = df.index  # gene names

        return bulk_df
        

class TSCA_Simulator(BaseSimulator):
    def __init__(self, adata=None, sample_size=8000, method='dirichlet'):
        self.adata = adata
        self.sample_size = sample_size
        self.method = method
        self.immune_cells = ['NK','T_CD4','T_CD8_CytT','Monocyte','Mast_cells']
        self.non_immune_cells = ['Fibroblast','Ciliated','Alveolar_Type1','Alveolar_Type2']

        self.summary_df = None
        self.cell_idx_dict = None
    
    def assign(self):
        assert self.method in ['uniform_sparse', 'uniform', 'dirichlet'], "Method not supported. Choose 'uniform_sparse', 'uniform', or 'dirichlet'."
        if self.method == 'uniform_sparse':
            # assign uniform distribution
            im_summary = self.assign_uniform(cell_types=self.immune_cells, sparse=True)
            non_im_summary = self.assign_uniform(cell_types=self.non_immune_cells, sparse=True)
        elif self.method == 'uniform':
            # assign uniform distribution
            im_summary = self.assign_uniform(cell_types=self.immune_cells, sparse=False)
            non_im_summary = self.assign_uniform(cell_types=self.non_immune_cells, sparse=False)
        elif self.method == 'dirichlet':
            im_summary = self.assign_dirichlet(a=1.0, cell_types=self.immune_cells)
            non_im_summary = self.assign_dirichlet(a=1.0, cell_types=self.non_immune_cells)
        else:
            raise ValueError("Method not supported. Choose 'uniform_sparse', 'uniform', or 'dirichlet'.")

        # normalize (sum to 1)
        self.im_summary = im_summary.div(im_summary.sum(axis=1), axis=0)
        self.non_im_summary = non_im_summary.div(non_im_summary.sum(axis=1), axis=0)

        summary_df = pd.concat([self.im_summary, self.non_im_summary], axis=1)
        self.summary_df = summary_df.div(summary_df.sum(axis=1), axis=0)  # normalize to sum to 1
    
    def set_data(self, summary_df=None, cell_idx_dict=None):
        if summary_df is not None:
            self.summary_df = summary_df
        if cell_idx_dict is not None:
            self.cell_idx_dict = cell_idx_dict
    
    def split_cell_idx(self, info_df=None, save_dir='./data/cell_idx'):
        if info_df is None:
            # adata = sc.read_h5ad("../Tissue_Stability_Cell_Atlas/lung.cellxgene.h5ad")
            info_df = self.adata.obs
        # Extract info
        cell_types = self.immune_cells + self.non_immune_cells
        cell_idx_dict = {}
        for cell in cell_types:
            target_cell = info_df[info_df['Celltypes_updated_July_2020'] == cell]
            target_idx = [info_df.index.tolist().index(t) for t in target_cell.index.tolist()]
            cell_size = len(target_idx)
            print("{}: {} cells are detected".format(cell, cell_size))
            # Train / Test Split
            random.seed(42)
            shuffle_idx = random.sample(target_idx, cell_size)
            train_idx = shuffle_idx[0:int(cell_size * 0.7)]
            test_idx = shuffle_idx[int(cell_size * 0.7):]
            cell_idx_dict[cell] = {'train': train_idx, 'test': test_idx}

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
            pd.to_pickle(train_idx, os.path.join(save_dir, f'{cell}_train_idx.pkl'))
            pd.to_pickle(test_idx, os.path.join(save_dir, f'{cell}_test_idx.pkl'))
        
        self.cell_idx_dict = cell_idx_dict

    def create_sim_bulk(self, pool_size=500, mode='train'):
        summary_df = self.summary_df
        cell_idx_dict = self.cell_idx_dict
        tototal_cells = summary_df.columns.tolist()

        # Pre-convert sparse matrix to dense array once
        if hasattr(self.adata.X, 'todense'):
            raw_exp = np.array(self.adata.X.todense())
        else:
            raw_exp = np.array(self.adata.X)

        pooled_exp = []
        np.random.seed(42)  # Set seed once outside the loop
        
        for idx in tqdm(range(len(summary_df))):
            p_list = summary_df.iloc[idx].values  # Use .values instead of .tolist() for speed
            final_idx = []
            
            for j, p in enumerate(p_list):
                cell = tototal_cells[j]
                tmp_size = int(pool_size * p)  # number of cells to be selected

                candi_idx = cell_idx_dict[cell][mode]
                
                # Use different random state for each sample to maintain reproducibility
                rng = np.random.RandomState(42 + idx * len(tototal_cells) + j)
                if len(candi_idx) < tmp_size:
                    select_idx = rng.choice(candi_idx, size=tmp_size, replace=True)
                else:
                    select_idx = rng.choice(candi_idx, size=tmp_size, replace=False)
                final_idx.extend(select_idx)
            
            if final_idx:  # Only process if there are selected indices
                # Direct sum operation on selected rows
                tmp_sum = raw_exp[final_idx, :].sum(axis=0)
                pooled_exp.append(tmp_sum.flatten())  # Ensure 1D array
        
        # Convert to DataFrame more efficiently
        pooled_exp = np.array(pooled_exp)
        bulk_df = pd.DataFrame(pooled_exp.T)
        bulk_df.index = self.adata.var_names  # gene names

        return bulk_df
    
    def create_sim_bulk_legacy(self, summary_df=None, pool_size=500, mode='train'):
        if summary_df is None:
            summary_df = self.summary
        tototal_cells = summary_df.columns.tolist()

        pooled_idx = []
        for idx in tqdm(range(len(summary_df))):
            p_list = summary_df.iloc[idx].tolist()
            final_idx = []
            for j, p in enumerate(p_list):
                cell = tototal_cells[j]
                tmp_size = int(pool_size * p)  # number of cells to be selected

                candi_idx = self.cell_idx_dict[cell][mode]
                # select tmp_size from tmp_df at random
                np.random.seed(seed=42)
                if len(candi_idx) < tmp_size:
                    select_idx = np.random.choice(candi_idx, size=tmp_size, replace=True)
                else:
                    select_idx = np.random.choice(candi_idx, size=tmp_size, replace=False)
                final_idx.extend(select_idx)
            pooled_idx.append(final_idx)
        
        raw_exp = np.array(self.adata.X.todense())
        pooled_exp = []
        for exp_idx in tqdm(pooled_idx):
            tmp_exp = raw_exp[exp_idx, :]
            tmp_sum = tmp_exp.sum(axis=0)
            pooled_exp.append(tmp_sum)
        bulk_df = pd.DataFrame(pooled_exp).T
        bulk_df.index = self.adata.var_names  # gene names

        return bulk_df


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
