# -*- coding: utf-8 -*-
"""
Created on 2025-05-21 (Wed) 17:15:53

Trainner for route8

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'

import os
os.chdir(BASE_DIR)

import gc
import anndata
import numpy as np
from anndata import AnnData

import torch

import sys
sys.path.append(BASE_DIR+'/github/GSTMDec/dann_autoencoder_dec')
from model.route8.gae_grl_model import *

sys.path.append(BASE_DIR+'/github/wandb-util')  
from wandbutil import WandbLogger

sys.path.append(BASE_DIR+'/github/deconv-utils')
from src import evaluation as ev

class BaseTrainer():
    def __init__(self, cfg, seed=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.seed = seed
    
    def build_dataloader(self, batch_size):
        ### Prepare data loader for training ###
        g = torch.Generator()
        g.manual_seed(self.seed)

        source_data = self.source_data
        target_data = self.target_data

        # 1. Source dataset
        if self.target_cells is None:
            source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
            self.target_cells = source_data.uns['cell_types']
        else:
            source_ratios = [source_data.obs[ctype] for ctype in self.target_cells]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
        self.celltype_num = len(self.target_cells)
        self.used_features = list(source_data.var_names)

        # 2. Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)

        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)
    
    def set_options(self):
        # prepare option list
        option_list = defaultdict(list)
        for key, value in vars(self.cfg.gaegrl).items():
            option_list[key] = value
        option_list['feature_num'] = self.source_data.shape[1]
        option_list['celltype_num'] = len(self.target_cells)

        self.option_list = option_list

        # parameter initialization
        self.best_loss = 1e10
        self.update_flag = 0
        self.w_stop_flag = False
        self.alpha, self.beta, self.rho = 0.0, 2.0, 1.0
        self.gamma = 0.25
        self.l1_penalty = 0.0
        self.h_thresh = 1e-4
        self.pre_h = np.inf

    
    def train_model(self, inference_fn=None):
        model = MultiTaskAutoEncoder(self.option_list, seed=self.seed).to(self.device)
        optimizer1 = torch.optim.Adam([{'params': model.encoder.parameters()},
                                       {'params': model.decoder.parameters()},
                                       {'params': model.w}],
                                       lr=model.lr)  # FIXME

        optimizer2 = torch.optim.Adam([#{'params': model.encoder.parameters()},  # FIXME
                                      {'params': model.embedder.parameters()},
                                      {'params': model.predictor.parameters()},
                                      {'params': model.discriminator.parameters()},],
                                      lr=model.lr)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.8)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.8)
        criterion_da = nn.BCELoss().to(self.device)

        source_label = torch.ones(model.batch_size).unsqueeze(1).to(self.device)
        target_label = torch.zeros(10000).unsqueeze(1).to(self.device)

        # WandB logger settings
        logger = WandbLogger(
            entity=self.cfg.wandb.entity,  
            project=self.cfg.wandb.project,  
            group=self.cfg.wandb.group, 
            name=self.cfg.wandb.name+f"_seed{self.seed}",
            config=self.option_list,
        )

        for epoch in range(model.num_epochs + 1):
            loss_dict, curr_h = self.run_epoch(model, epoch, optimizer1, optimizer2, criterion_da,
                                               source_label, target_label)

            if (epoch + 1) % 10 == 0 and not self.w_stop_flag:
                if curr_h > self.gamma * self.pre_h:
                    self.rho *= self.beta
                self.alpha += self.rho * curr_h.detach().cpu()
                self.pre_h = curr_h
                if curr_h <= self.h_thresh and epoch > 100:
                    print(f"Stopped updating W at epoch {epoch+1}")
                    self.w_stop_flag = True
            
            # Inference
            if inference_fn is not None:
                summary_df = inference_fn(model)
                loss_dict.update({
                    'R': summary_df.loc['mean']['R'],
                    'CCC': summary_df.loc['mean']['CCC'],
                    'MAE': summary_df.loc['mean']['MAE'],
                })


            logger(epoch=epoch, **loss_dict)

            if loss_dict['pred_disc_loss'] < self.best_loss:
                self.update_flag = 0
                self.best_loss = loss_dict['pred_disc_loss']
                torch.save(model.state_dict(), os.path.join(self.cfg.paths.gaegrl_model_path, f'best_model_{self.seed}.pth'))
            else:
                self.update_flag += 1
                if self.update_flag == model.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch:{epoch}, Loss:{loss_dict['total_loss']:.3f}, dag:{loss_dict['dag_loss']:.3f}, pred:{loss_dict['pred_loss']:.3f}, disc:{loss_dict['disc_loss']:.3f}, disc_auc:{loss_dict['disc_auc']:.3f}")
            
            gc.collect()

        torch.save(model.state_dict(), os.path.join(self.cfg.paths.gaegrl_model_path, f'last_model.pth'))
    
    def run_epoch(self, model, epoch, optimizer1, optimizer2, criterion_da, source_label, target_label):
        model.train()
        dag_loss_epoch, pred_loss_epoch, disc_loss_epoch = 0., 0., 0.
        all_preds = []
        all_labels = []
        for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
            target_x = next(iter(self.train_target_loader))[0]   # NOTE: shuffle
            #target_x = torch.Tensor(test_data.X)

            total_steps = model.num_epochs * len(self.train_source_loader)
            p = float(batch_idx + epoch * len(self.train_source_loader)) / total_steps
            a = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            source_x, source_y, target_x = source_x.cuda(), source_y.cuda(), target_x.cuda()
            rec_s, _, _= model(source_x, a)
            rec_t, _, _ = model(target_x, a)

            #### 1. DAG-related loss
            w_adj = model.w_adj
            curr_h = model.losses.compute_h(w_adj)
            curr_mse_s = model.losses.dag_rec_loss(target_x.reshape((target_x.size(0), target_x.size(1), 1)), rec_t)  # NOTE target_x: (12, n_genes) --> (12, n_genes, 1)
            curr_mse_t = model.losses.dag_rec_loss(source_x.reshape((source_x.size(0), source_x.size(1), 1)), rec_s)
            curr_mse = curr_mse_s + curr_mse_t
            #curr_mse = curr_mse_t
            dag_loss = (curr_mse
                        + self.l1_penalty * torch.norm(w_adj, p=1)
                        + self.alpha * curr_h + 0.5 * self.rho * curr_h * curr_h)

            dag_loss_epoch += dag_loss.data.item()
            dag_loss = model.dag_w * dag_loss

            if self.w_stop_flag:
                pass
            else:
                optimizer1.zero_grad()
                dag_loss.backward(retain_graph=True)
                optimizer1.step()

            # NOTE: re-obtain the prediction and domain classification
            _, pred_s, domain_s = model(source_x, a)
            _, pred_t, domain_t = model(target_x, a)

            #### 2. prediction
            if model.pred_loss_type == 'L1':
                pred_loss = model.losses.L1_loss(pred_s, source_y.cuda())
            elif model.pred_loss_type == 'custom':
                pred_loss = model.losses.summarize_loss(pred_s, source_y.cuda())
            else:
                raise ValueError("Invalid prediction loss type.")
            pred_loss_epoch += pred_loss.data.item()

            #### 3. domain classification
            all_preds.extend(domain_s.cpu().detach().numpy().flatten())  # Source domain predictions
            all_preds.extend(domain_t.cpu().detach().numpy().flatten())  # Target domain predictions
            all_labels.extend([1]*domain_s.shape[0])  # Source domain labels
            all_labels.extend([0]*domain_t.shape[0])  # Target domain labels

            disc_loss = criterion_da(domain_s, source_label[0:domain_s.shape[0],]) + criterion_da(domain_t, target_label[0:domain_t.shape[0],])
            disc_loss_epoch += disc_loss.data.item()

            #### 4. pred_loss + disc_loss
            loss = model.pred_w * pred_loss + model.disc_w * disc_loss

            optimizer2.zero_grad()
            loss.backward(retain_graph=True)
            optimizer2.step()
        
        # summarize loss
        dag_loss_epoch = model.dag_w * dag_loss_epoch / len(self.train_source_loader)
        pred_loss_epoch = model.pred_w * pred_loss_epoch / len(self.train_source_loader)
        disc_loss_epoch = model.disc_w * disc_loss_epoch / len(self.train_source_loader)
        loss_all = dag_loss_epoch + pred_loss_epoch + disc_loss_epoch
        auc_score = roc_auc_score(all_labels, all_preds)

        loss_dict = {
            'dag_loss': dag_loss_epoch,
            'pred_loss': pred_loss_epoch,
            'disc_loss': disc_loss_epoch,
            'total_loss': loss_all,
            'pred_disc_loss': pred_loss_epoch + disc_loss_epoch,
            'disc_auc': auc_score,
        }

        return loss_dict, curr_h
    
    def predict(self):
        # load best model
        model_path = os.path.join(self.cfg.paths.gaegrl_model_path, f'best_model_{self.seed}.pth')
        model = MultiTaskAutoEncoder(self.option_list, seed=self.seed).cuda()
        model.load_state_dict(torch.load(model_path))

        # inference with the best model
        model.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            rec, logits, domain = model(x.cuda(), alpha=1.0)
            logits = logits.detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
        final_preds_target = pd.DataFrame(preds, columns=self.target_cells)

        return final_preds_target, gt


class BenchmarkTrainer(BaseTrainer):
    def __init__(self, cfg, seed=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.seed = seed

        self.set_data()
        self.build_dataloader(batch_size=cfg.gaegrl.batch_size)
        self.set_options()
    
    def set_data(self):
        train_data, test_data, train_y, test_y, gene_names = prep4benchmark(h5ad_path=self.cfg.paths.h5ad_path,
                                                                            source_list=self.cfg.common.source_domain,
                                                                            target=self.cfg.common.target_domain,
                                                                            target_cells=self.target_cells,
                                                                            n_samples=self.cfg.common.n_samples, 
                                                                            n_vtop=self.cfg.common.n_vtop,
                                                                            seed=self.seed)
        self.source_data = train_data
        self.target_data = test_data
        self.target_y = test_y
    
    def train_model(self):
        def inference_fn(model):
            summary_df, _ = self.target_inference(model=model, do_plot=False)
            return summary_df

        super().train_model(inference_fn=inference_fn)

    
    def target_inference(self, model=None, do_plot=False):
        if model is None:
            # load model
            model_path = os.path.join(self.cfg.paths.gaegrl_model_path, f'best_model_{self.seed}.pth')
            model = MultiTaskAutoEncoder(self.option_list, seed=self.seed).cuda()
            model.load_state_dict(torch.load(model_path))
            print("Model loaded from %s" % model_path)

        # inference
        model.eval()

        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            rec, logits, domain = model(x.cuda(), alpha=1.0)
            logits = logits.detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
        final_preds_target = pd.DataFrame(preds, columns=self.target_cells)

        dec_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
        val_name_list = [["Monocytes"],["Unknown"],["Bcells"],["CD4Tcells"],["CD8Tcells"],["NK"]]
        res = ev.eval_deconv(dec_name_list=dec_name_list, val_name_list=val_name_list, deconv_df=final_preds_target, y_df=self.target_y, do_plot=do_plot)

        # summarize
        r_list = []
        mae_list = []
        ccc_list = []
        for i in range(len(dec_name_list)):
            tmp_res = res[i][0]
            r, mae, ccc = tmp_res['R'], tmp_res['MAE'], tmp_res['CCC']
            r_list.append(r)
            mae_list.append(mae)
            ccc_list.append(ccc)
        summary_df = pd.DataFrame({'R':r_list, 'CCC':ccc_list, 'MAE':mae_list})
        summary_df.index = [t[0] for t in val_name_list]
        summary_df.loc['mean'] = summary_df.mean()

        return summary_df, final_preds_target
    
class InferenceTrainer(BaseTrainer):
    def __init__(self, cfg, seed=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.seed = seed

        self.set_data()
        self.build_dataloader(batch_size=cfg.gaegrl.batch_size)
        self.set_options()
    
    def set_data(self):
        train_data, test_data, train_y, gene_names = prep4inference(h5ad_path=self.cfg.paths.h5ad_path,
                                                                    target_path=self.cfg.paths.target_path,
                                                                    source_list=self.cfg.common.source_domain,
                                                                    target=self.cfg.common.target_domain,
                                                                    target_cells=self.target_cells,
                                                                    n_samples=self.cfg.common.n_samples, 
                                                                    n_vtop=self.cfg.common.n_vtop,
                                                                    seed=self.seed)
        self.source_data = train_data
        self.target_data = test_data
    
    def train_model(self):
        super().train_model(inference_fn=None)


def prep4benchmark(h5ad_path, source_list=['data6k'], target='sdy67', priority_genes=[], 
                   target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], 
                   n_samples=None, n_vtop=None, seed=42):
    print(f"Source domain: {source_list}")
    print(f"Target domain: {target}")

    pbmc = sc.read_h5ad(h5ad_path)
    test = pbmc[pbmc.obs['ds'] == target]

    train, label_idx = extract_variable_sources(pbmc, source_list, n_samples=n_samples, n_vtop=n_vtop, seed=seed)
    train_data, test_data, train_y, gene_names = finalize_data(train, test, label_idx, target_cells, priority_genes, log_conv=(target != 'GSE65133'))
    test_y = test.obs[target_cells]

    return train_data, test_data, train_y, test_y, gene_names

def prep4inference(h5ad_path, target_path, source_list=['data6k'], target='TSCA_Lung', priority_genes=[], 
             target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], 
             n_samples=None, n_vtop=None, target_log_conv=True, seed=42):

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
            target_adata.obs[col] = 'TSCA_Lung' if col == "ds" else (2 if col == "batch" else np.nan)
    

    combined_adata = sc.concat([pbmc, target_adata], join='inner', merge='first')
    test = combined_adata[combined_adata.obs['ds'] == target]

    train, label_idx = extract_variable_sources(combined_adata, source_list, n_samples=n_samples, n_vtop=n_vtop, seed=seed)
    train_data, test_data, train_y, gene_names = finalize_data(train, test, label_idx, target_cells, priority_genes, log_conv=target_log_conv)

    return train_data, test_data, train_y, gene_names

def extract_variable_sources(pbmc, source_list, n_samples=None, n_vtop=None, seed=42):
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

def finalize_data(train, test, label_idx, target_cells, priority_genes=[], log_conv=True):
    priority_label = np.array([gene in priority_genes for gene in train.var_names])
    priority_idx = np.where(priority_label)[0]
    print(f"Priority genes: {np.sum(priority_label)}/{len(priority_genes)} genes")

    label_idx = np.unique(np.concatenate([label_idx, priority_idx]))
    gene_names = train.var_names[label_idx]

    train_data = train[:, label_idx].copy()
    train_data.X = np.log2(train_data.X + 1)

    test_data = test[:, label_idx].copy()
    if log_conv:
        test_data.X = np.log2(test_data.X + 1)

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

