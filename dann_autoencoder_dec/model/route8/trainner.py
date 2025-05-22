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
import numpy as np

import torch

import sys
sys.path.append(BASE_DIR+'/github/GSTMDec/dann_autoencoder_dec')
from model.route8.gae_grl_model import *

sys.path.append(BASE_DIR+'/github/wandb-util')  
from wandbutil import WandbLogger

sys.path.append(BASE_DIR+'/github/deconv-utils')
from src import evaluation as ev


class SimpleTrainer():
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.target_cells = cfg.common.target_cells
        self.seed = cfg.common.seed

        self.set_data()
        self.build_dataloader(batch_size=cfg.gaegrl.batch_size)
        self.set_options()
    
    def set_data(self):
        train_data, test_data, train_y, test_y, gene_names = preprocess(h5ad_path=self.cfg.paths.h5ad_path,
                                                                        source_list=self.cfg.common.source_domain,
                                                                        target=self.cfg.common.target_domain,
                                                                        target_cells=self.target_cells,
                                                                        n_samples=self.cfg.common.n_samples, 
                                                                        n_vtop=self.cfg.common.n_vtop,
                                                                        seed=self.seed)
        self.source_data = train_data
        self.target_data = test_data
        self.target_y = test_y

    
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
        option_list['n_domain'] = len(self.cfg.common.source_domain)

        self.option_list = option_list

    
    def train_model(self):
        # prepare model structure
        model = MultiTaskAutoEncoder(self.option_list,seed=self.seed).cuda()

        # setup optimizer
        optimizer1 = torch.optim.Adam([{'params': model.encoder.parameters()},
                                       {'params': model.decoder.parameters()},
                                       {'params': model.w}],
                                       lr=model.lr)

        optimizer2 = torch.optim.Adam([#{'params': model.encoder.parameters()},  # FIXME
                                      {'params': model.embedder.parameters()},
                                      {'params': model.predictor.parameters()},
                                      {'params': model.discriminator.parameters()},],
                                      lr=model.lr)

        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.8)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.8)

        criterion_da = nn.BCELoss().cuda()
        source_label = torch.ones(model.batch_size).unsqueeze(1).cuda()   # source domain label as 1
        target_label = torch.zeros(10000).unsqueeze(1).cuda()  # target domain label as 0

        # WandB logger settings
        logger = WandbLogger(
            entity=self.cfg.wandb.entity,  
            project=self.cfg.wandb.project,  
            group=self.cfg.wandb.group, 
            name=self.cfg.wandb.name,
            config=self.option_list,
        )


        model.metric_logger = defaultdict(list) 
        best_loss = 1e10  
        update_flag = 0  
        w_stop_flag = False

        l1_penalty = 0.0
        alpha, beta, rho = 0.0, 2.0, 1.0
        gamma = 0.25
        h_thresh = 1e-4 # FIXME: default 1e-8
        pre_h = np.inf

        for epoch in range(model.num_epochs+1):
            model.train()

            train_target_iterator = iter(self.train_target_loader)
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
                            + l1_penalty * torch.norm(w_adj, p=1)
                            + alpha * curr_h + 0.5 * rho * curr_h * curr_h)

                dag_loss_epoch += dag_loss.data.item()
                dag_loss = model.dag_w * dag_loss

                if w_stop_flag:
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
            
            # update alpha and rho
            if (epoch+1) % 10 == 0:
                if w_stop_flag:
                    pass
                else:
                    if curr_h > gamma * pre_h:
                        rho *= beta
                    else:
                        pass
                    alpha += rho * curr_h.detach().cpu()
                    pre_h = curr_h
                    if curr_h <= h_thresh and epoch > 100:
                        print("Stopped updating W at epoch %d" % (epoch+1))
                        w_stop_flag = True
            
            # summarize loss
            dag_loss_epoch = model.dag_w * dag_loss_epoch / len(self.train_source_loader)
            pred_loss_epoch = model.pred_w * pred_loss_epoch / len(self.train_source_loader)
            disc_loss_epoch = model.disc_w * disc_loss_epoch / len(self.train_source_loader)
            loss_all = dag_loss_epoch + pred_loss_epoch + disc_loss_epoch
            auc_score = roc_auc_score(all_labels, all_preds)

            # inference
            summary_df = self.target_inference(model, do_plot=False)
            

            # wandb logging
            logger(
                epoch=epoch,
                dag_loss=dag_loss_epoch,
                pred_loss=pred_loss_epoch,
                disc_loss=disc_loss_epoch,
                total_loss=loss_all,
                disc_auc=auc_score,
                pred_disc_loss=pred_loss_epoch + disc_loss_epoch,
                R=summary_df.loc['mean']['R'],
                CCC=summary_df.loc['mean']['CCC'],
                MAE=summary_df.loc['mean']['MAE'],
            )


            # early stopping
            target_loss = pred_loss_epoch + disc_loss_epoch  # NOTE
            if target_loss < best_loss:
                update_flag = 0
                best_loss = target_loss
                model.metric_logger['best_epoch'] = epoch
                torch.save(model.state_dict(), os.path.join(self.cfg.paths.gaegrl_model_path, f'best_model.pth'))
            else:
                update_flag += 1
                if update_flag == model.early_stop:
                    print("Early stopping at epoch %d" % (epoch+1))
                    break

            if epoch % 10 == 0:
                print(f"Epoch:{epoch}, Loss:{loss_all:.3f}, dag:{dag_loss_epoch:.3f}, pred:{pred_loss_epoch:.3f}, disc:{disc_loss_epoch:.3f}, disc_auc:{auc_score:.3f}")
            
            gc.collect()

    def target_inference(self, model=None, do_plot=False):
        if model is None:
            # load model
            model_path = os.path.join(self.cfg.paths.gaegrl_model_path, f'best_model.pth')
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

        return summary_df

def preprocess(h5ad_path, source_list=['data6k'], target='sdy67', 
               priority_genes=[], target_cells=['Monocytes', 'Unknown', 'CD4Tcells', 'Bcells', 'NK', 'CD8Tcells'], n_samples=None, n_vtop=None, seed=42):
    assert target in ['sdy67', 'GSE65133', 'donorA', 'donorC', 'data6k', 'data8k']
    pbmc = sc.read_h5ad(h5ad_path)
    test = pbmc[pbmc.obs['ds']==target]

    if n_samples is not None:
        np.random.seed(seed)
        idx = np.random.choice(8000, n_samples, replace=False)
        donorA = pbmc[pbmc.obs['ds']=='donorA'][idx]
        donorC = pbmc[pbmc.obs['ds']=='donorC'][idx]
        data6k = pbmc[pbmc.obs['ds']=='data6k'][idx]
        data8k = pbmc[pbmc.obs['ds']=='data8k'][idx]
    
    else:    
        donorA = pbmc[pbmc.obs['ds']=='donorA']
        donorC = pbmc[pbmc.obs['ds']=='donorC']
        data6k = pbmc[pbmc.obs['ds']=='data6k']
        data8k = pbmc[pbmc.obs['ds']=='data8k']

    if source_list == 'all':
        train = anndata.concat([donorA, donorC, data6k, data8k])
    else:
        if n_samples is not None:
            train = pbmc[pbmc.obs['ds'].isin(source_list)][idx]
        else:
            train = pbmc[pbmc.obs['ds'].isin(source_list)]

    train_y = train.obs[target_cells]
    test_y = test.obs[target_cells]
    
    if n_vtop is None:
        #### variance cut off
        label = test.X.var(axis=0) > 0.1  # FIXME: mild cut-off
        label_idx = np.where(label)[0]
    else:
        #### top 1000 highly variable genes
        label_idx = np.argsort(-train.X.var(axis=0))[:n_vtop]
    
    # add priority genes
    priority_label = np.array([True if gene in priority_genes else False for gene in train.var_names])
    priority_idx = np.where(priority_label)[0]
    print(f"Priority genes: {np.sum(priority_label)}/{len(priority_genes)} genes")
    label_idx = np.unique(np.concatenate([label_idx, priority_idx]))
    gene_names = train.var_names[label_idx]
    
    train_data = train[:, label_idx]
    train_data.X = np.log2(train_data.X + 1)
    test_data = test[:, label_idx]
    if target != 'GSE65133':
        test_data.X = np.log2(test_data.X + 1)
    else:
        # GSE65133 is already log2 transformed
        test_data.X = test_data.X

    print("Train data shape: ", train_data.X.shape)
    print("Test data shape: ", test_data.X.shape)

    return train_data, test_data, train_y, test_y, gene_names

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

