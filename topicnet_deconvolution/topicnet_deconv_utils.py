# -*- coding: utf-8 -*-
"""
Created on 2024-11-28 (Thu) 13:08:09

Utils for TopicNet deconvolution.

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'

import copy
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(BASE_DIR+'/github/deconv-utils')
from src import preprocessing as pp
from src import evaluation as ev

sys.path.append(BASE_DIR+'/github/GSTMDec')
from _utils import common_utils

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# %%
class DataPrepSCADEN():
    def __init__(self, adata, raw_genes=None, sampling_size=1024, target_cells=['Monocytes', 'Unknown', 'Bcells', 'CD4Tcells', 'CD8Tcells', 'NK']):
        self.adata = adata
        self.raw_genes = [t.upper() for t in list(adata.var_names)]
        self.target_cells = target_cells
        self.sampling_size = sampling_size
    
    def load_data(self):
        np.random.seed(42)
        sampled_idx = sorted(np.random.choice(8000, self.sampling_size, replace=False))  # shuffle index

        adata = self.adata
        data_6k = adata.X[adata.obs['ds']=='data6k'][sampled_idx]
        self.data_6k_y = adata.obs[adata.obs['ds']=='data6k'].iloc[sampled_idx][self.target_cells]
        data_8k = adata.X[adata.obs['ds']=='data8k'][sampled_idx]
        self.data_8k_y = adata.obs[adata.obs['ds']=='data8k'].iloc[sampled_idx][self.target_cells]
        data_a = adata.X[adata.obs['ds']=='donorA'][sampled_idx]
        self.data_a_y = adata.obs[adata.obs['ds']=='donorA'].iloc[sampled_idx][self.target_cells]
        data_c = adata.X[adata.obs['ds']=='donorC'][sampled_idx]
        self.data_c_y = adata.obs[adata.obs['ds']=='donorC'].iloc[sampled_idx][self.target_cells]

        # real bulk
        data_sdy67 = adata.X[adata.obs['ds']=='sdy67']
        self.data_sdy67_y = adata.obs[adata.obs['ds']=='sdy67'][self.target_cells]
        data_gse65133 = adata.X[adata.obs['ds']=='GSE65133']
        data_gse65133  = np.expm1(data_gse65133)  # log1p to original scale
        self.data_gse65133_y = adata.obs[adata.obs['ds']=='GSE65133'][self.target_cells]

        # log1p transformation
        self.raw_data_6k = np.log1p(data_6k)
        self.raw_data_8k = np.log1p(data_8k)
        self.raw_data_a = np.log1p(data_a)
        self.raw_data_c = np.log1p(data_c)
        self.raw_data_sdy67 = np.log1p(data_sdy67)
        self.raw_data_gse65133 = np.log1p(data_gse65133)
    
    def set_genes(self, target_genes):
        target_genes = [t.upper() for t in target_genes]
        common_genes = sorted(set(self.raw_genes) & set(target_genes))
        print(f'Raw genes: {len(self.raw_genes)}')
        print(f'Input genes: {len(target_genes)}')
        print(f'Common genes: {len(common_genes)}')
        self.target_gene_idx = [self.raw_genes.index(gene) for gene in common_genes]
        self.target_genes = common_genes
    
    def processing(self, bath_norm=True, ds_list=['data6k', 'data8k', 'donorA', 'donorC', 'sdy67', 'GSE65133']):
        if bath_norm:
            # concatenate
            concat_pool = []
            for ds in ds_list:
                if ds == 'data6k':
                    concat_pool.append(self.raw_data_6k)
                elif ds == 'data8k':
                    concat_pool.append(self.raw_data_8k)
                elif ds == 'donorA':
                    concat_pool.append(self.raw_data_a)
                elif ds == 'donorC':
                    concat_pool.append(self.raw_data_c)
                elif ds == 'sdy67':
                    concat_pool.append(self.raw_data_sdy67)
                elif ds == 'GSE65133':
                    concat_pool.append(self.raw_data_gse65133)
                else:
                    raise ValueError('Invalid dataset name')
            concatenated = np.concatenate(concat_pool, axis=0)[:,self.target_gene_idx]  # gene selection

            concat_labels = []
            for i, e in enumerate(concat_pool):
                concat_labels += [i]*len(e)

            #concatenated = np.concatenate((self.raw_data_6k, self.raw_data_8k, self.raw_data_a, self.raw_data_c, self.raw_data_sdy67, self.raw_data_gse65133), axis=0)[:,self.target_gene_idx]  # gene selection
            #concat_labels = [0]*len(self.raw_data_6k) + [1]*len(self.raw_data_8k) + [2]*len(self.raw_data_a) + [3]*len(self.raw_data_c) + [4]*len(self.raw_data_sdy67) + [5]*len(self.raw_data_gse65133)
            # combat
            batch_df = pp.batch_norm(df=pd.DataFrame(concatenated).T, lst_batch=concat_labels)
            batch_df = np.expm1(batch_df)  # back to original (linear) scale
            batch_df.index = self.target_genes

            # scaling between 0 and 100
            self.batch_df = batch_df / batch_df.max().max() * 100

            # sample selection
            # FIXME: Depending on the order in the ds_list, it may break down.
            sampling_size = self.sampling_size
            data_6k = self.batch_df.iloc[:,0:sampling_size].T.values.astype(np.float32)
            data_8k = self.batch_df.iloc[:,sampling_size:sampling_size*2].T.values.astype(np.float32)
            data_a = self.batch_df.iloc[:,sampling_size*2:sampling_size*3].T.values.astype(np.float32)
            data_c = self.batch_df.iloc[:,sampling_size*3:sampling_size*4].T.values.astype(np.float32)
            data_sdy67 = self.batch_df.iloc[:,sampling_size*4:sampling_size*4+12].T.values.astype(np.float32)
            data_gse65133 = self.batch_df.iloc[:,sampling_size*4+12:].T.values.astype(np.float32)
        else:
            data_6k = np.expm1(self.data_6k[:,self.target_gene_idx]).astype(np.float32)
            data_8k = np.expm1(self.data_8k[:,self.target_gene_idx]).astype(np.float32)
            data_a = np.expm1(self.data_a[:,self.target_gene_idx]).astype(np.float32)
            data_c = np.expm1(self.data_c[:,self.target_gene_idx]).astype(np.float32)
            data_sdy67 = np.expm1(self.data_sdy67[:,self.target_gene_idx]).astype(np.float32)
            data_gse65133 = np.expm1(self.data_gse65133[:,self.target_gene_idx]).astype(np.float32)
        
        # train, valid, test split
        self.data_6k = copy.deepcopy(data_6k)
        self.data_8k = copy.deepcopy(data_8k)
        self.data_a = copy.deepcopy(data_a)
        self.data_c = copy.deepcopy(data_c)
        self.data_sdy67 = copy.deepcopy(data_sdy67)
        self.data_gse65133 = copy.deepcopy(data_gse65133)
    
        return


class CustomDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y.values

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]


class LossSummary():
    def __init__(self, trainer):
        self.trainer = trainer
        self.train_loss_history = trainer.train_loss_history
        self.valid_loss_history = trainer.valid_loss_history
        self.train_graph_loss_history = trainer.train_graph_kl_loss_history
        self.valid_graph_loss_history = trainer.valid_graph_kl_loss_history
        self.train_loss_sum_history = trainer.train_loss_sum_history
        self.valid_loss_sum_history = trainer.valid_loss_sum_history

        # separate to encoder and decoder loss
        self.train_enc_loss_history = [t[0] for t in self.train_loss_history]
        self.train_dec_loss_history = [t[1] for t in self.train_loss_history]
        self.train_enc_graph_kl = [t[0] for t in self.train_graph_loss_history]
        self.train_dec_graph_kl = [t[1] for t in self.train_graph_loss_history]

    def plot_sum(self):
        """ 1. sum of all losses """
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.plot(self.train_loss_sum_history, label='Train')
        axs.plot([j * 10 for j in range(len(self.valid_loss_sum_history))], self.valid_loss_sum_history, label='Valid')
        axs.set_ylabel('Loss')
        axs.set_title('Sum of Loss')
        axs.legend()
        plt.show()
    
    def plot_likelihood_deconv_loss(self, loss_labels):
        """ 2. Likelihood and deconvolution loss """
        #loss_labels = ['layer=0', 'layer=1', 'layer=2', 'layer=3', 'layer=4', 'Deconvolution']
        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)  # 3行2列の設定
        axs = axs.flatten()  # 配列を1次元に

        for t in range(len(loss_labels)):
            print(loss_labels[t])
            axs[t].plot([i[t] for i in self.train_dec_loss_history], label='Train_Dec ' + loss_labels[t])
            axs[t].plot([i[t] for i in self.train_enc_loss_history], label='Train_Enc ' + loss_labels[t], linestyle='-.')
            axs[t].plot([j * 10 for j in range(len(self.valid_loss_history))], 
                        [i[t] for i in self.valid_loss_history], label='Valid ' + loss_labels[t], linestyle='--')
            axs[t].set_ylabel('Loss')
            axs[t].set_title(f'Likelihood Loss ({loss_labels[t]})')
            axs[t].legend()

        axs[-2].set_xlabel('Epoch')  # X軸ラベルを追加
        plt.tight_layout()
        plt.show()
    
    def plot_graph_kl_loss(self, loss_labels):
        """ 3. Graph KL divergence """
        #loss_labels = ['layer=0', 'layer=1', 'layer=2', 'layer=3', 'layer=4']
        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)  # 3行2列の設定
        axs = axs.flatten()  # 配列を1次元に

        for t in range(len(loss_labels)):
            print(loss_labels[t])
            axs[t].plot([i[t] for i in self.train_enc_graph_kl], label='Train_Dec ' + loss_labels[t])
            axs[t].plot([i[t] for i in self.train_dec_graph_kl], label='Train_Enc ' + loss_labels[t], linestyle='-.')
            axs[t].plot([j * 10 for j in range(len(self.valid_graph_loss_history))], 
                        [i[t] for i in self.valid_graph_loss_history], label='Valid ' + loss_labels[t], linestyle='--')
            axs[t].set_ylabel('Loss')
            axs[t].set_title(f'Graph KL Loss ({loss_labels[t]})')
            axs[t].legend()

        # 不要なグラフを非表示
        axs[-1].axis('off')

        axs[-2].set_xlabel('Epoch')  # X軸ラベルを追加
        plt.tight_layout()
        plt.show()
    
    def viz_graph_prior(self, model, graph, do_z=True):
        Phi = []
        for t in range(self.trainer.layer_num):
            Phi.append(model.decoder[t].w.cpu().detach().numpy())
            print(model.decoder[t].w.cpu().detach().numpy().shape)
        
        sns.clustermap(z_score_transform(torch.tensor(Phi[0])).cpu().detach().numpy())
        plt.show()
        
        for i in range(len(Phi)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            dep_matrix = Phi[i]
            prior_matrix = graph[i]
            # z-score (index)
            dep_matrix_z = z_score_transform(torch.tensor(dep_matrix)).cpu().detach().numpy()
            if do_z:
                sns.heatmap(dep_matrix_z, ax=axes[0])
            else:
                sns.heatmap(dep_matrix, ax=axes[0])
            sns.heatmap(prior_matrix.cpu().detach(), ax=axes[1])
            plt.title('Dependence Matrix')
            plt.show()
    
class EvalModel():
    def __init__(self, model, checkpoint_path, args):
        self.args = args
        self.model = model
        self.checkpoint_path = checkpoint_path

    def load_model(self):
        if not hasattr(self, 'model_loaded'):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            self.model_loaded = True 

    def eval_corr(self, data_x, data_y, deconv_layer=1,
                     dec_name_list=[[0],[1],[2],[3],[4],[5]], 
                     val_name_list=[["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]],
                     run_n=1):
        
        deconv_df_summary = []
        for seed_idx in tqdm(range(run_n)):
            set_seed(seed_idx) 
            self.load_model()
            self.model.eval()
            phi_theta, theta, loss, likelihood, graph_kl_loss = self.model(torch.tensor(data_x).to(self.args.device))

            topic_size = [self.args.vocab_size] + self.args.topic_size
            layer_num = len(topic_size) - 1

            # sum to 1 across all topics
            for t in range(layer_num):
                theta[t] = theta[t] / torch.sum(theta[t], 0, keepdim=True)  

            # layer used for calculating deconvolution loss
            tmp_output = pd.DataFrame(theta[deconv_layer].detach().cpu().numpy()).T
            deconv_df_summary.append(tmp_output)
        deconv_df_mean = pd.concat(deconv_df_summary).groupby(level=0).mean()

        y_df = data_y.reset_index(drop=True)

        summary_df = pd.concat([deconv_df_mean, y_df],axis=1)
        self.corr_df = summary_df.corr()  
        #sns.clustermap(self.corr_df)  # visualize correlation matrix
        #plt.show()

        self.res = ev.eval_deconv(dec_name_list=dec_name_list, val_name_list=val_name_list, deconv_df=deconv_df_mean, y_df=y_df)

        # summarize
        r_list = []
        mae_list = []
        ccc_list = []
        rmse_list = []
        for i in range(len(dec_name_list)):
            tmp_res = self.res[i][0]
            r, mae, ccc, rmse = tmp_res['R'], tmp_res['MAE'], tmp_res['CCC'], tmp_res['RMSE']
            r_list.append(r)
            mae_list.append(mae)
            ccc_list.append(ccc)
            rmse_list.append(rmse)
        summary_df = pd.DataFrame({'R':r_list, 'CCC':ccc_list, 'MAE':mae_list, 'RMSE':rmse_list})
        summary_df.index = [t[0] for t in val_name_list]

        self.summary_df = summary_df

        return


# %%
def tf_idf_selection(ref_df, raw_genes, threshold=0.001, fold_change=1.5):
    ref_df.columns = [t.upper() for t in ref_df.columns.tolist()]
    ref_df = ref_df[raw_genes]

    # calc TF-IDF
    transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    tfidf_matrix = transformer.fit_transform(ref_df).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=ref_df.columns)

    # Gene selection based on TF-IDF 
    tfidf_array = tfidf_df.values  # (n_cells, n_genes)
    sorted_indices = np.argsort(-tfidf_array, axis=0)  # sort in descending order
    top_1_indices = sorted_indices[0, :]  # max TF-IDF cell index
    top_2_indices = sorted_indices[1, :]  # second max TF-IDF cell index
    top_1_values = tfidf_array[top_1_indices, np.arange(tfidf_array.shape[1])]
    top_2_values = tfidf_array[top_2_indices, np.arange(tfidf_array.shape[1])]

    # convert cell index to cell name
    top_1_cells = tfidf_df.index[top_1_indices]
    top_2_cells = tfidf_df.index[top_2_indices]

    # summarize the results
    results = pd.DataFrame({
        "Top Cell": top_1_cells,
        "Top TF-IDF": top_1_values,
        "Second Top Cell": top_2_cells,
        "Second TF-IDF": top_2_values
    }, index=tfidf_df.columns)  # gene names
    results["Fold_change"] = results["Top TF-IDF"] / results["Second TF-IDF"]
    results = results.sort_values("Fold_change", ascending=False)

    target_results = results[(results["Top TF-IDF"] > threshold) & (results["Fold_change"] > fold_change)]
    target_genes = [t.upper() for t in target_results.index]
    target_gene_idx = [raw_genes.index(gene) for gene in target_genes if gene in raw_genes]
    assert len(target_gene_idx) == len(target_genes)

    return target_genes, target_results



def adjust_column_to_minimum(df_binary, original_df):
    """
    Adjusts the number of ones in all columns to match the column with the minimum number of ones.
    Returns a new DataFrame.

    Parameters:
        df_binary (pd.DataFrame): Input binary DataFrame
        original_df (pd.DataFrame): Original DataFrame containing the values
        
    Returns:
        pd.DataFrame: A new binary DataFrame after adjustment
    """
    adjusted_df = copy.deepcopy(df_binary)

    column_ones = df_binary.sum(axis=0)
    min_ones = column_ones.min()
    
    for col in adjusted_df.columns:
        current_ones = column_ones[col]
        if current_ones > min_ones:
            # remove extra 1s (remove in order of smallest original values)
            indices_with_1 = adjusted_df[adjusted_df[col] == 1].index
            sorted_indices = original_df.loc[indices_with_1, col].sort_values().index
            indices_to_remove = sorted_indices[:current_ones - min_ones]
            adjusted_df.loc[indices_to_remove, col] = 0
    
    return adjusted_df


def get_topic_tree_prior_1(ref_df, target_genes, topic_tree_path='./path/to/xxx.pkl', sparse=True):
    """
    Generates topic-tree prior data for a given reference DataFrame and target genes.

    This function processes the reference data to assign genes to topics based on their expression values 
    and structures the data for use in topic models. It supports both sparse and dense assignment 
    strategies and builds a graph representation of the topics.
    """
    ref_df.columns = [t.upper() for t in ref_df.columns.tolist()]
    ref_df = ref_df[target_genes]

    topic_list = []
    graph = []
    graph_net = pd.read_pickle(topic_tree_path)
    for i,k in enumerate(graph_net):
        if i == 0:
            # assign 1 to the column with the largest value in each row and 0 otherwise
            tmp_df = ref_df.T
            df_binary = (tmp_df.eq(tmp_df.max(axis=1), axis=0)).astype(int)
            if sparse:
                adjusted_df = adjust_column_to_minimum(df_binary, tmp_df)
                tmp = adjusted_df.values
            # Scenario in which target genes are always assigned to one of the topics.
            else:  
                tmp = df_binary.values
        else:
            tmp = graph_net[k]
        tmp = torch.tensor(tmp).float().cuda()
        print(tmp.shape)
        topic_list.append(tmp.shape[1])
        graph.append(tmp)
    
    return topic_list, graph

def get_topic_tree_prior_0(ref_df, target_genes, topic_tree_path='./path/to/xxx.pkl'):
    """
    Legacy
    """
    ref_df.columns = [t.upper() for t in ref_df.columns.tolist()]
    ref_df = ref_df[target_genes]

    topic_list = []
    graph = []
    graph_net = pd.read_pickle(topic_tree_path)
    for i,k in enumerate(graph_net):
        if i == 0:
            # assign 1 to the column with the largest value in each row and 0 otherwise
            tmp_df = ref_df.T
            tmp = tmp_df.values
        else:
            tmp = graph_net[k]
        tmp = torch.tensor(tmp).float().cuda()
        print(tmp.shape)
        topic_list.append(tmp.shape[1])
        graph.append(tmp)
    
    return topic_list, graph

def z_score_transform(matrix):
    """
    Perform z-score normalization on each row of the matrix for each column.

    Args:
        matrix (torch.Tensor): Input matrix to be transformed (z-score is computed for each row).

    Returns:
        torch.Tensor: The z-score normalized matrix.
    """
    # Calculate the mean and standard deviation for each row
    mean = matrix.mean(dim=1, keepdim=True)  # Mean for each row
    std = matrix.std(dim=1, keepdim=True)    # Standard deviation for each row
    
    # Perform z-score normalization
    z_score_matrix = (matrix - mean) / std
    return z_score_matrix
