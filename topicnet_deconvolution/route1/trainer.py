import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.cluster import KMeans

from topicnet_dec import *
from utils import *

import sys
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'
sys.path.append(BASE_DIR+'/github/GSTMDec')
from _utils import common_utils

class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)

        self.model = GBN_model(args)
        self.optimizer = torch.optim.Adam([{'params': self.model.h_encoder.parameters()},
                                           {'params': self.model.shape_encoder.parameters()},
                                           {'params': self.model.scale_encoder.parameters()}],
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(),
                                                  lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_data_loader, valid_data=None, enc_loss_weights=[1e-4, 1e-3, 1e-2], dec_loss_weights=[1e-4, 1e-3, 1e-2, 1e5], deconv_layer=1):
        self.train_loss_history = []
        self.valid_loss_history = []

        best_loss = 1e10
        for epoch in tqdm(range(self.epochs)):
            for t in range(self.layer_num - 1):
                self.model.decoder[t + 1].mu = self.model.decoder[t].mu_c
                self.model.decoder[t + 1].log_sigma = self.model.decoder[t].log_sigma_c

            self.model.cuda()

            loss_t = [0] * (self.layer_num + 2)
            likelihood_t = [0] * (self.layer_num + 1)
            graph_kl_loss_t = [0] * (self.layer_num + 1)
            num_data = len(train_data_loader)
            deconv_running_loss = 0.0

            # 1. training phase
            for i, (train_data, train_label) in enumerate(train_data_loader):

                self.model.h_encoder.train()
                self.model.shape_encoder.train()
                self.model.scale_encoder.train()
                self.model.decoder.eval()

                train_data = torch.tensor(train_data, dtype=torch.float).cuda()

                re_x, theta, loss_list, likelihood, graph_kl_loss = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (enc_loss_weights[0] * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += enc_loss_weights[0] * loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        graph_kl_loss_t[t] += graph_kl_loss[t].item()/num_data

                    elif t < self.layer_num:
                        (enc_loss_weights[1] * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] +=  enc_loss_weights[1] * loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        graph_kl_loss_t[t] += graph_kl_loss[t].item() / num_data

                    else:  # graph_kl_loss is not taken into account when t == self.layer_num.
                        (enc_loss_weights[2] * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] +=  enc_loss_weights[2] * loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data


                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.train()

                re_x, theta, loss_list, likelihood, graph_kl_loss = self.model(train_data)
                # sum to 1 constraint
                self.theta = theta
                for t in range(self.layer_num):
                    self.theta[t] = self.theta[t] / torch.sum(self.theta[t], 0, keepdim=True)  # sum to 1 across all topics
                for t in range(self.layer_num + 1):
                    if t == 0:
                        (dec_loss_weights[0] * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] +=  dec_loss_weights[0] * loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        graph_kl_loss_t[t] += graph_kl_loss[t].item() / num_data

                    elif t < self.layer_num:
                        (dec_loss_weights[1] * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] +=  dec_loss_weights[1] * loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        graph_kl_loss_t[t] += graph_kl_loss[t].item() / num_data
                    else:
                        (dec_loss_weights[2] * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] +=  dec_loss_weights[2] * loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                
                # deconvolution loss (NOTE: the position is correct?)
                # print(theta[1].T.shape, train_label.shape)  # >> torch.Size([256, 6]) torch.Size([256, 6])
                deconv_loss = self.summarize_loss(self.theta[deconv_layer].T, train_label)
                deconv_loss = dec_loss_weights[3] * deconv_loss / num_data
                deconv_loss.backward()
                deconv_running_loss += deconv_loss.item()

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=20, norm_type=2)
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()
            """
            if epoch % 1 == 0:
                for t in range(self.layer_num + 1):
                    print('epoch {}|{}, layer {}|{}, loss: {}, likelihood: {}, lb: {}, graph_kl_loss: {}'.format(
                        epoch, self.epochs, t, self.layer_num, loss_t[t]/2, likelihood_t[t]/2, loss_t[t]/2, graph_kl_loss_t[t]/2))
                self.vis_txt()
            """
            loss_t[-1] += deconv_running_loss
            self.train_loss_history.append(loss_t)
        
            # 2. validation phase
            if valid_data is not None:
                # valid_data: (valid_x, valid_y)
                valid_x = torch.tensor(valid_data[0], dtype=torch.float).cuda()
                valid_y = torch.tensor(valid_data[1], dtype=torch.float).cuda()

                if epoch % 10 == 0:
                    loss_v = [0] * (self.layer_num + 2)
                    likelihood_v = [0] * (self.layer_num + 1)

                    self.model.eval()
                    re_x, theta, loss_list, likelihood, graph_kl_loss = self.model(valid_x)
                    # sum to 1 constraint
                    for t in range(self.layer_num):
                        theta[t] = theta[t] / torch.sum(theta[t], 0, keepdim=True)  # sum to 1 across all topics

                    for t in range(self.layer_num + 1):
                        if t == 0:
                            loss_v[t] += dec_loss_weights[0] *loss_list[t].item()
                        elif t < self.layer_num:
                            loss_v[t] += dec_loss_weights[1] *loss_list[t].item()
                        else:
                            loss_v[t] += dec_loss_weights[2] *loss_list[t].item()
                        likelihood_v[t] += likelihood[t].item()

                    # deconvolution loss
                    deconv_loss_v = self.summarize_loss(theta[deconv_layer].T, valid_y)
                    loss_v[-1] += dec_loss_weights[3] * deconv_loss_v.item()
                    self.valid_loss_history.append(loss_v)

                    valid_sum_loss = sum(loss_v)
                    if valid_sum_loss < best_loss:
                        best_loss = valid_sum_loss
                        # save model
                        torch.save(self.model.state_dict(), self.save_path)


    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def vision_phi(self, Phi, outpath='phi_output', top_n=50):
        if self.voc is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words

    def vis_txt(self, outpath='phi_output'):
        phi = []
        for t in range(self.layer_num):
            phi.append(self.model.decoder[t].w.cpu().detach().numpy())

        self.vision_phi(phi, outpath)
    
    def summarize_loss(self, theta_tensor, prop_data):
        # deconvolution loss
        # if prop_data is not tensor, convert it to tensor
        if type(prop_data) == torch.Tensor:
            prop_tensor = prop_data.cuda()
        else:
            prop_tensor = torch.tensor(prop_data.values).cuda()

        assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
        deconv_loss_dic = common_utils.calc_deconv_loss(theta_tensor, prop_tensor)
        deconv_loss = deconv_loss_dic['cos_sim'] + 0.0*deconv_loss_dic['rmse']

        return deconv_loss
