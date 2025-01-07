from models import *
from data_process import *
import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(123)
from scipy import interp
import warnings
warnings.filterwarnings("ignore")
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from multiprocessing import Pool
import pickle
from Bio.Align import substitution_matrices
import sys
sys.path.append('.')

n_layers = 1  # number of Encoder of Decoder Layer
batch_size = 1024
epochs = 150
threshold = 0.5
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

    

def train_step(model, train_loader, fold, epoch, epochs, use_cuda = True):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    time_train_ep = 0
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []
    for train_pep_inputs, train_cdr_inputs, train_labels in tqdm(train_loader):
        '''
        pep_inputs: [batch_size, pep_len]
        cdr3_inputs: [batch_size, cdr_len]
        train_outputs: [batch_size, 2]
        '''
        train_pep_inputs, train_cdr_inputs, train_labels = train_pep_inputs.to(device), train_cdr_inputs.to(device), train_labels.to(device)
        t1 = time.time()
        train_outputs, _, _, train_dec_self_attns = model(train_cdr_inputs, train_pep_inputs)
        train_loss = criterion(train_outputs, train_labels)
        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim = 1)(train_outputs)[:, 1].cpu().detach().numpy()
        
        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss)
#         dec_attns_train_list.append(train_dec_self_attns)
        
    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    
    print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs, f_mean(loss_train_list), time_train_ep))
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_ = True)
    
    return ys_train, loss_train_list, metrics_train, time_train_ep#, dec_attns_train_list

def eval_step(model, val_loader, fold, epoch, epochs, use_cuda = True):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model.eval()
    torch.manual_seed(19961231)
    torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_cdr_inputs, val_labels in tqdm(val_loader):
            val_pep_inputs, val_cdr_inputs, val_labels = val_pep_inputs.to(device), val_cdr_inputs.to(device), val_labels.to(device)
            val_outputs, _, _, val_dec_self_attns = model(val_cdr_inputs,val_pep_inputs)
            val_loss = criterion(val_outputs, val_labels)

            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim = 1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss)
#             dec_attns_val_list.append(val_dec_self_attns)
            
        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
        
        print('Fold-{} ****Test  Epoch-{}/{}: Loss = {:.6f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_ = True)
    return ys_val, loss_val_list, metrics_val#, dec_attns_val_list








def data_with_loader(type_ = 'train',fold = None,  batch_size = 128):
    if type_ != 'train' and type_ != 'val':
#         data = pd.read_csv('../data/justina_test.csv')
        data = pd.read_csv('./inputs/inputs_bd.csv')
#         data = pd.read_csv('../data/test/GILGLVFTL.csv')
#         data = pd.read_csv('../data/posi_length.csv')
        
    elif type_ == 'train':
        data = pd.read_csv('./mutation_data/add_10xneg/add_10xneg/train_add10Xneg_{}.csv'.format(fold))

    elif type_ == 'val':
        data = pd.read_csv('./mutation_data/add_10xneg/add_10xneg/eva_add10Xneg_{}.csv'.format(fold))
    pep_inputs, cdr_inputs,labels = make_data(data)
    loader = Data.DataLoader(MyDataSet(pep_inputs, cdr_inputs,labels), batch_size, shuffle = True, num_workers = 0)
    n_samples = len(pep_inputs)
    len_cdr3 = len(cdr_inputs[0])
    len_epi = len(pep_inputs[0])
    encoding_mask = np.zeros([n_samples, len_cdr3,len_epi])
    for idx_sample, (enc_cdr3_this, enc_epi_this) in enumerate(zip(cdr_inputs, pep_inputs)):
        mask = np.ones([len_cdr3,len_epi])
        zero_cdr3 = (enc_cdr3_this == 0)
        mask[zero_cdr3,:] = 0
        zero_epi = (enc_epi_this == 0)
        mask[:,zero_epi] = 0
#         print(mask.shape)
        encoding_mask[idx_sample] = mask
    return data, pep_inputs, cdr_inputs, labels,loader,encoding_mask
for n_heads in range(5,6):
    
    ys_train_fold_dict, ys_val_fold_dict = {}, {}
    train_fold_metrics_list, val_fold_metrics_list = [], []
    independent_fold_metrics_list, external_fold_metrics_list, ys_independent_fold_dict, ys_external_fold_dict = [], [], {}, {}
    attns_train_fold_dict, attns_val_fold_dict, attns_independent_fold_dict, attns_external_fold_dict = {}, {}, {}, {}
    loss_train_fold_dict, loss_val_fold_dict, loss_independent_fold_dict, loss_external_fold_dict = {}, {}, {}, {}

    for fold in range(1,6):
        print('=====Fold-{}====='.format(fold))
        print('-----Generate data loader-----')
        train_data, train_pep_inputs, train_cdr_inputs, train_labels, train_loader,_ = data_with_loader(type_ = 'train', fold = fold,  batch_size = batch_size)
        val_data, val_pep_inputs, val_cdr_inputs, val_labels, val_loader,_ = data_with_loader(type_ = 'val', fold = fold,  batch_size = batch_size)
        print('Fold-{} Label info: Train = {} | Val = {}'.format(fold, Counter(train_data.label), Counter(val_data.label)))

        print('-----Compile model-----')
        model = Transformer().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = 1e-3)#, momentum = 0.99)

        print('-----Train-----')
        dir_saver = './model2'
    
        path_saver = './model2/tcr_st_layer{}_multihead{}_fold{}_netmhcpan.pkl'.format(n_layers, n_heads, fold)
        metric_best, ep_best = 0, -1
        time_train = 0
        for epoch in range(1, epochs + 1):

            ys_train, loss_train_list, metrics_train, time_train_ep = train_step(model, train_loader, fold, epoch, epochs, use_cuda) # , dec_attns_train
            ys_val, loss_val_list, metrics_val = eval_step(model, val_loader, fold, epoch, epochs, use_cuda) #, dec_attns_val

            metrics_ep_avg = sum(metrics_val[:4])/4
            if metrics_ep_avg > metric_best: 
                metric_best, ep_best = metrics_ep_avg, epoch
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                print('****Saving model: Best epoch = {} | 5metrics_Best_avg = {:.4f}'.format(ep_best, metric_best))
                print('*****Path saver: ', path_saver)
                torch.save(model.eval().state_dict(), path_saver)

            time_train += time_train_ep

        print('-----Optimization Finished!-----')
        print('-----Evaluate Results-----')
        if ep_best >= 0:
            print('*****Path saver: ', path_saver)
            model.load_state_dict(torch.load(path_saver))
            model_eval = model.eval()

            ys_res_train, loss_res_train_list, metrics_res_train = eval_step(model_eval, train_loader, fold, ep_best, epochs, use_cuda) # , train_res_attns
            ys_res_val, loss_res_val_list, metrics_res_val = eval_step(model_eval, val_loader, fold, ep_best, epochs, use_cuda) # , val_res_attns


            train_fold_metrics_list.append(metrics_res_train)
            val_fold_metrics_list.append(metrics_res_val)

        print("Total training time: {:6.2f} sec".format(time_train))




