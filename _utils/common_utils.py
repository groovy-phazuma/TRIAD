# -*- coding: utf-8 -*-
"""
Created on 2024-09-27 (Fri) 14:17:06

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'

import gc
import sys
import time
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib import colors as mcolors
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

sys.path.append(BASE_DIR+'/github/GLDADec')
from _utils import gldadec_processing
from _utils import plot_utils as pu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def kl_annealing_schedule(epoch, total_epochs, max_beta=1.0):
    """linear KL annealing scheduler""" 
    return min(max_beta, epoch / total_epochs * max_beta)

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def loop_iterable(iterable):
    while True:
        yield from iterable

def eval_deconv(dec_name_list = [[0],[1]], val_name_list = [["Monocytes"],["CD4Tcells"]], deconv_df=None, y_df=None):
    # overall
    assert len(dec_name_list) == len(val_name_list)
    tab_colors = mcolors.TABLEAU_COLORS
    color_list = list(tab_colors.keys())

    overall_res = []
    for i in range(len(dec_name_list)):
        dec_name = dec_name_list[i]
        val_name = val_name_list[i]
        plot_dat = pu.DeconvPlot(deconv_df=deconv_df,val_df=y_df,dec_name=dec_name,val_name=val_name,plot_size=20,dpi=50)
        res = plot_dat.plot_simple_corr(color=color_list[i],title=f'Topic:{dec_name} vs {val_name}',target_samples=None)
        overall_res.append(res)
    
    return overall_res

def eval_data(model, data_x, data_y, level=1):
    # inference mode
    model.Net.eval()
    with torch.no_grad():
        output = model.Net(torch.tensor(data_x).to(device))

    doc_topic_dist_2 = model.Net.get_doc_topic_dist(level=2)  # (64, 8)
    doc_topic_dist_1 = model.Net.get_doc_topic_dist(level=1)  # (64, 32)
    doc_topic_dist_0 = model.Net.get_doc_topic_dist(level=0)  # (64, 128) = theta_1

    doc_topic_dist = model.Net.get_doc_topic_dist(level=level) 

    # concat (doc-topic) and (prop)
    deconv_df = pd.DataFrame(doc_topic_dist.cpu().detach().numpy())
    deconv_df = deconv_df.div(deconv_df.sum(axis=1), axis=0)  # normalize to make it sum to 1 (row-wise)

    y_df = data_y.reset_index(drop=True)
    summary_df = pd.concat([deconv_df, y_df],axis=1)

    # visualize doc-topic distribution
    sns.heatmap(summary_df)
    plt.show()

    # clustermap of correlaton matrix
    corr_df = summary_df.corr()
    corr_df.style.background_gradient(cmap='coolwarm')
    corr_df = corr_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    sns.clustermap(corr_df)
    plt.show()

    return deconv_df, y_df, corr_df, output

def calc_deconv_loss(theta_tensor, prop_tensor):
    ext_theta = theta_tensor[:,0:prop_tensor.shape[1]]  # (batch_size, n_cell_types)

    mse = torch.mean((ext_theta - prop_tensor) ** 2)
    rmse = torch.sqrt(mse)
    cel = -torch.mean(torch.log(ext_theta) * prop_tensor)

    # cosine similarity
    cos_sim = 1 - F.cosine_similarity(ext_theta, prop_tensor, dim=0)  # (n_cell_types,)
    cos_sim = cos_sim.mean()

    return {'mse':mse, 'rmse':rmse, 'cel':cel, 'cos_sim':cos_sim}

def get_batches(train_data, batch_size=300, rand=True, device="cuda:0"):
    """
    (
        train_data,
        train_label,
        train_text,
    ) = reader.get_sparse_matrix("train+valid", mode="count")
    """
    n, d = train_data.shape

    batchs = n // batch_size
    while True:
        idxs = np.arange(train_data.shape[0])

        if rand:
            # seed
            np.random.seed(0)
            np.random.shuffle(idxs)

        for count in range(batchs):
            wordcount = []
            beg = count * batch_size
            end = (count + 1) * batch_size

            idx = idxs[beg:end]
            # data format
            if isinstance(train_data, csr_matrix):
                data = train_data[idx].toarray()  # csr_matrix
            else:
                data = train_data[idx]  # NumPy
            data = torch.from_numpy(data).to(device)
            yield idx, data  # update 240915s
