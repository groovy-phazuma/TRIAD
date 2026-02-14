# -*- coding: utf-8 -*-
"""
Created on 2024-11-27 (Wed) 21:52:13

@author: I.Azuma
"""

import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm

class BaseSimulator:
    def __init__(self, sample_size, method='dirichlet'):
        self.sample_size = sample_size
        self.method = method
        self.summary_df = None
        self.cell_idx_dict = None
        self.adata = None
    
    def assign_uniform(self, cell_types: list, sparse=True):
        """Generate uniform distribution for cell type proportions."""
        final_res = []
        for idx in range(self.sample_size):
            np.random.seed(seed=idx)
            if sparse:
                # Randomly select subset of cell types
                use_cell_types = np.random.choice(
                    cell_types, 
                    size=np.random.randint(1, len(cell_types) + 1), 
                    replace=False
                )
                p_list = np.random.rand(len(use_cell_types))
                
                # Create full proportion list
                final_p_list = np.zeros(len(cell_types))
                for j, c in enumerate(use_cell_types):
                    final_p_list[cell_types.index(c)] = p_list[j]
            else:
                final_p_list = np.random.rand(len(cell_types))
            
            # Normalize to sum to 1
            norm_p_list = final_p_list / final_p_list.sum() if final_p_list.sum() > 0 else final_p_list
            final_res.append(norm_p_list)
        
        return pd.DataFrame(final_res, columns=cell_types)
    
    def assign_dirichlet(self, cell_types: list, alpha=1.0, do_viz=False):
        """Generate Dirichlet distribution for cell type proportions."""
        alpha_vec = [alpha] * len(cell_types)
        np.random.seed(seed=42)
        data = np.random.dirichlet(alpha_vec, size=self.sample_size)

        if do_viz and len(cell_types) > 1:
            plt.hist(data[:, 1], bins=50, alpha=0.7, color='blue', label=f'alpha={alpha}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of proportion')
            plt.legend()
            plt.show()
        
        return pd.DataFrame(data, columns=cell_types)

    def create_ref(self):
        """Create reference matrix from training data."""
        raw_exp = self._get_expression_matrix()
        pooled_exp = []
        
        for cell_type, idx_dict in self.cell_idx_dict.items():
            train_idx = idx_dict['train']
            tmp_mean = raw_exp[train_idx].mean(axis=0)
            pooled_exp.append(tmp_mean)

        ref_df = pd.DataFrame(pooled_exp).T
        ref_df.index = self.adata.var_names
        ref_df.columns = self.cell_idx_dict.keys()
        return ref_df
    
    def _get_expression_matrix(self):
        """Helper method to get expression matrix from adata."""
        if hasattr(self.adata.X, 'todense'):
            return np.array(self.adata.X.todense())
        else:
            return np.array(self.adata.X)
    def assign(self, nonim_w=None):
        """Assign cell type proportions using specified method."""
        methods = {
            'uniform_sparse': lambda types: self.assign_uniform(types, sparse=True),
            'uniform': lambda types: self.assign_uniform(types, sparse=False), 
            'dirichlet': lambda types: self.assign_dirichlet(types, alpha=1.0)
        }
        
        if self.method not in methods:
            raise ValueError(f"Method not supported. Choose from: {list(methods.keys())}")
        
        # Generate proportions for immune and non-immune cells
        im_summary = methods[self.method](self.immune_cells)
        non_im_summary = methods[self.method](self.non_immune_cells)
        
        # Normalize to sum to 1
        self.im_summary = im_summary.div(im_summary.sum(axis=1), axis=0)
        self.non_im_summary = non_im_summary.div(non_im_summary.sum(axis=1), axis=0)

        # Apply random weights if specified
        if nonim_w is not None:
            random_w = np.random.uniform(low=nonim_w, high=1.0, size=self.sample_size)
            for i in range(self.sample_size):
                w = random_w[i]
                self.im_summary.iloc[i] *= (1 - w)
                self.non_im_summary.iloc[i] *= w

        # Combine and normalize final result
        summary_df = pd.concat([self.im_summary, self.non_im_summary], axis=1)
        self.summary_df = summary_df.div(summary_df.sum(axis=1), axis=0)
    
    def set_data(self, summary_df=None, cell_idx_dict=None, adata=None):
        if summary_df is not None:
            self.summary_df = summary_df
        if cell_idx_dict is not None:
            self.cell_idx_dict = cell_idx_dict
        if adata is not None:
            self.adata = adata

class LiverCellAtlas_Simulator(BaseSimulator):
    def __init__(self, sample_size=8000, method='dirichlet', base_dir=None):
        super().__init__(sample_size, method)
        self.base_dir = base_dir or "/workspace/cluster/HDD/azuma/TopicModel_Deconv"
        self.immune_cells = ['Neutrophils', 'Monocytes & Monocyte-derived cells', 'Kupffer cells', 
                           'NK cells', 'T cells', 'B cells', 'pDCs', 'cDC1s', 'cDC2s']
        self.non_immune_cells = ['Hepatocytes', 'Cholangiocytes', 'Fibroblasts']

        self.filter_dict = {
            'Neutrophils': 'Neutrophils', 'Monocytes & Monocyte-derived cells': 'Monocytes',
            'Kupffer cells': 'Kupffer', 'NK cells': 'NK', 'T cells': 'T', 'B cells': 'B',
            'pDCs': 'pDCs', 'cDC1s': 'cDC1s', 'cDC2s': 'cDC2s',
            'Hepatocytes': 'Hepatocytes', 'Cholangiocytes': 'Cholangiocytes', 'Fibroblasts': 'Fibroblasts'
        }
    
    def split_cell_idx(self, save_dir='./data/cell_idx', train_ratio=0.7):
        """Split cell indices into train/test sets."""
        if self.adata is None:
            raise ValueError("adata must be set before splitting cell indices")
            
        cell_types = self.immune_cells + self.non_immune_cells
        cell_idx_dict = {}
        
        for cell in cell_types:
            cellname = self.filter_dict[cell]
            cell_mask = self.adata.obs['cell_type'] == cell
            target_idx = list(range(cell_mask.sum()))
            cell_size = len(target_idx)
            print(f"{cell}: {cell_size} cells detected")
            
            # Train/Test split
            random.seed(42)
            shuffle_idx = random.sample(target_idx, cell_size)
            split_point = int(cell_size * train_ratio)
            train_idx = shuffle_idx[:split_point]
            test_idx = shuffle_idx[split_point:]
            cell_idx_dict[cell] = {'train': train_idx, 'test': test_idx}

            # Save indices if directory specified
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                pd.to_pickle(train_idx, os.path.join(save_dir, f'{cellname}_train_idx.pkl'))
                pd.to_pickle(test_idx, os.path.join(save_dir, f'{cellname}_test_idx.pkl'))
        
        self.cell_idx_dict = cell_idx_dict
    
    def create_sim_bulk(self, pool_size=500, mode='train', adata_path=None):
        """Create simulated bulk expression data."""
        if self.summary_df is None or self.cell_idx_dict is None:
            raise ValueError("summary_df and cell_idx_dict must be set")
        
        # Load adata if path provided, otherwise use self.adata
        if adata_path:
            adata = sc.read_h5ad(adata_path)
        elif self.adata is not None:
            adata = self.adata
        else:
            # Default path as fallback
            adata_path = f"{self.base_dir}/datasource/scRNASeq/LiverCellAtlas/mouseStStAll/processed/liver_adata_148202x19052.h5ad"
            adata = sc.read_h5ad(adata_path)

        total_cells = self.summary_df.columns.tolist()
        pooled_exp = []
        np.random.seed(42)
        
        #reverse_dict = {v: k for k, v in self.filter_dict.items()}
        
        for idx in tqdm(range(len(self.summary_df))):
            p_list = self.summary_df.iloc[idx].values
            bulk_single = None
            
            for j, p in enumerate(p_list):
                cell = total_cells[j]
                #cellname = self.reverse_dict[cell]
                cellname = cell
                tmp_size = int(pool_size * p)
                candi_idx = self.cell_idx_dict[cellname][mode]

                # Generate reproducible random indices
                rng = np.random.RandomState(42 + idx * len(total_cells) + j)
                select_idx = rng.choice(candi_idx, size=tmp_size, replace=len(candi_idx) < tmp_size)
                
                # Get expression data for selected cells
                unique_cols = list(dict.fromkeys(select_idx))
                tmp_adata = adata[adata.obs['cell_type'] == cellname, :].copy()
                df = tmp_adata[unique_cols, :].to_df().T
                
                # Handle repeated selections
                col_pos_map = {val: i for i, val in enumerate(unique_cols)}
                col_map = [col_pos_map[i] for i in select_idx]
                df_repeated = df.iloc[:, col_map]
                tmp_sum = df_repeated.sum(axis=1).values

                if tmp_sum.size > 0:
                    bulk_single = tmp_sum if bulk_single is None else bulk_single + tmp_sum
            
            if bulk_single is not None:
                pooled_exp.append(bulk_single.flatten())
        
        # Create DataFrame
        pooled_exp = np.array(pooled_exp)
        bulk_df = pd.DataFrame(pooled_exp.T)
        bulk_df.index = df.index  # gene names
        return bulk_df


class TSCA_Simulator(BaseSimulator):
    def __init__(self, adata=None, sample_size=8000, method='dirichlet'):
        super().__init__(sample_size, method)
        self.adata = adata
        self.immune_cells = ['NK', 'T_CD4', 'T_CD8_CytT', 'Monocyte', 'Mast_cells']
        self.non_immune_cells = ['Fibroblast', 'Ciliated', 'Alveolar_Type1', 'Alveolar_Type2']
    
    def split_cell_idx(self, info_df=None, save_dir='./data/cell_idx', train_ratio=0.7):
        """Split cell indices for TSCA data."""
        if info_df is None:
            info_df = self.adata.obs
            
        cell_types = self.immune_cells + self.non_immune_cells
        cell_idx_dict = {}
        
        for cell in cell_types:
            target_cell = info_df[info_df['Celltypes_updated_July_2020'] == cell]
            target_idx = [info_df.index.tolist().index(t) for t in target_cell.index.tolist()]
            cell_size = len(target_idx)
            print(f"{cell}: {cell_size} cells detected")
            
            # Train/Test split
            random.seed(42)
            shuffle_idx = random.sample(target_idx, cell_size)
            split_point = int(cell_size * train_ratio)
            train_idx = shuffle_idx[:split_point]
            test_idx = shuffle_idx[split_point:]
            cell_idx_dict[cell] = {'train': train_idx, 'test': test_idx}

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                pd.to_pickle(train_idx, os.path.join(save_dir, f'{cell}_train_idx.pkl'))
                pd.to_pickle(test_idx, os.path.join(save_dir, f'{cell}_test_idx.pkl'))
        
        self.cell_idx_dict = cell_idx_dict

    def create_sim_bulk(self, pool_size=500, mode='train'):
        """Create simulated bulk expression data for TSCA."""
        if self.summary_df is None or self.cell_idx_dict is None:
            raise ValueError("summary_df and cell_idx_dict must be set")
            
        total_cells = self.summary_df.columns.tolist()
        raw_exp = self._get_expression_matrix()
        pooled_exp = []
        np.random.seed(42)
        
        for idx in tqdm(range(len(self.summary_df))):
            p_list = self.summary_df.iloc[idx].values
            final_idx = []
            
            for j, p in enumerate(p_list):
                cell = total_cells[j]
                tmp_size = int(pool_size * p)
                candi_idx = self.cell_idx_dict[cell][mode]
                
                rng = np.random.RandomState(42 + idx * len(total_cells) + j)
                select_idx = rng.choice(candi_idx, size=tmp_size, replace=len(candi_idx) < tmp_size)
                final_idx.extend(select_idx)
            
            if final_idx:
                tmp_sum = raw_exp[final_idx, :].sum(axis=0)
                pooled_exp.append(tmp_sum.flatten())
        
        pooled_exp = np.array(pooled_exp)
        bulk_df = pd.DataFrame(pooled_exp.T)
        bulk_df.index = self.adata.var_names
        return bulk_df


class MyPBMC_Simulator:
    def __init__(self, adata, adata_counts, cell_idx_dict=None, sample_size=8000):
        self.adata = adata
        self.adata_counts = adata_counts
        self.sample_size = sample_size
        self.cell_types = sorted(self.adata.obs['celltype'].unique().tolist())
        
        # Build cell index dictionary if not provided
        if cell_idx_dict is not None:
            self.cell_idx_dict = cell_idx_dict
        else:
            raw_idx = self.adata.obs.index.tolist()
            self.cell_idx_dict = {}
            for c in self.cell_types:
                tmp_idx = self.adata.obs[self.adata.obs['celltype'] == c].index.tolist()
                self.cell_idx_dict[c] = [raw_idx.index(i) for i in tmp_idx]
    
    def create_sim_bulk(self, summary_df=None, pool_size=500):
        """Create simulated bulk expression data for PBMC."""
        if summary_df is None:
            summary_df = self.summary_df
            
        pooled_exp = []
        for idx in tqdm(range(len(summary_df))):
            p_list = summary_df.iloc[idx].tolist()
            final_idx = []
            
            for j, p in enumerate(p_list):
                cell = self.cell_types[j]
                tmp_size = int(pool_size * p)
                candi_idx = self.cell_idx_dict[cell]
                
                # Random selection with reproducible seed
                np.random.seed(seed=idx)
                select_idx = np.random.choice(candi_idx, size=tmp_size, replace=len(candi_idx) < tmp_size)
                final_idx.extend(select_idx)
            
            # Quality check
            expected_range = (pool_size - len(self.cell_types), pool_size + len(self.cell_types))
            if not (expected_range[0] <= len(final_idx) <= expected_range[1]):
                print(f"Warning: {len(final_idx)} cells selected (expected ~{pool_size})")

            # Sum expression counts
            tmp_sum = list(np.array(self.adata_counts.X[final_idx].sum(axis=0))[0])
            pooled_exp.append(tmp_sum)

        bulk_df = pd.DataFrame(pooled_exp).T
        bulk_df.index = self.adata_counts.var_names
        return bulk_df
