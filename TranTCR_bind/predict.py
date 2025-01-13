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
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
d_model=64 # Embedding Size
d_ff = 256 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer


n_heads = 5
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual) # [batch_size, seq_len, d_model]
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask): # dec_inputs = enc_outputs
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
#         self.tgt_emb = nn.Embedding(d_model * 2, d_model)
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tgt_len
        
    def forward(self, dec_inputs): # dec_inputs = enc_outputs (batch_size, peptide_cdr_maxlen_sum, d_model)
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device) # [batch_size, tgt_len, d_model]

        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((dec_inputs.shape[0], tgt_len, tgt_len))).bool().to(device)
 
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
            
        return dec_outputs, dec_self_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        self.pep_encoder = Encoder().to(device)
        self.cdr_encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.tgt_len = tgt_len
        self.projection = nn.Sequential(
                                        nn.Linear(tgt_len * d_model, 256),
                                        nn.ReLU(True),

                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 64),
                                        nn.ReLU(True),

                                        #output layer
                                        nn.Linear(64, 2)
                                        ).to(device)
        
    def forward(self, cdr_inputs,pep_inputs):
        '''
        pep_inputs: [batch_size, pep_len]
        cdr_inputs: [batch_size, cdr_len]
        '''
        cdr_enc_outputs, cdr_enc_self_attns = self.cdr_encoder(cdr_inputs)
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        
        enc_outputs = torch.cat((cdr_enc_outputs,pep_enc_outputs), 1) # concat pep & cdr embedding
        dec_outputs, dec_self_attns = self.decoder(enc_outputs)
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1) # Flatten [batch_size, tgt_len * d_model]
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        return dec_logits.view(-1, dec_logits.size(-1)), pep_enc_self_attns, cdr_enc_self_attns, dec_self_attns

def make_data(data):
    cdr3 = data['CDR3b'].values
    epitope = data['peptide'].values
    labels = data['binder'].values
    mat = Tokenizer() 
    cdr_inputs = encode_cdr3(cdr3, mat)
    pep_inputs = encode_epi(epitope, mat)

    return torch.LongTensor(pep_inputs), torch.LongTensor(cdr_inputs), torch.LongTensor(labels)

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, cdr_inputs,labels):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.cdr_inputs = cdr_inputs
        self.labels = labels
        

    def __len__(self): # 样本数
        return self.pep_inputs.shape[0] # 改成cdr_inputs也可以哦！

    def __getitem__(self, idx):
        return self.pep_inputs[idx],self.cdr_inputs[idx], self.labels[idx]


def eval_step(model, val_loader, fold, epoch, epochs, dir_head, use_cuda = True):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model.eval()
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
            
        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
        
        print('Fold-{} ****Test  Epoch-{}/{}: Loss = {:.6f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        dir_name = str(dir_head)+'_Fold-{}.csv'.format(fold)
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, dir_name,True)
    return ys_val, loss_val_list, metrics_val#, dec_attns_val_list

def data_with_loader(type_ = 'train',fold = None,  batch_size = 128):
    if type_ != 'train' and type_ != 'val':
        data = pd.read_csv('./extend_testdata/mari_data_100%test.csv')
        
        
    elif type_ == 'train':
        data = pd.read_csv('../突变负样本/VDJ_10X_McPAS_1V5/train_VDJ_10X_McPAS_1V5_{}.csv'.format(fold))

    elif type_ == 'val':
        data = pd.read_csv('../突变负样本/VDJ_10X_McPAS_1V5/eva_VDJ_10X_McPAS_1V5_{}.csv'.format(fold))

    pep_inputs, cdr_inputs,labels = make_data(data)
    loader = Data.DataLoader(MyDataSet(pep_inputs, cdr_inputs,labels), batch_size, shuffle = False, num_workers = 0)
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
        encoding_mask[idx_sample] = mask
    return data, pep_inputs, cdr_inputs, labels,loader,encoding_mask
import csv
path_saver = './model2/tcr_st_layer1_multihead5_fold1_netmhcpan.pkl'
fold = 4
epochs = None
threshold = 0.5
model = Transformer().to(device)
model.load_state_dict(torch.load(path_saver))
type_ = 'test'
save_ = False
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
criterion = nn.CrossEntropyLoss()
ep_best = None
dir_head = './shiyan/all_IEDB_newneg1V1.csv_Fold-4.csv'
data, pep_inputs, cdr_inputs, labels, loader,_ = data_with_loader(type_,fold = fold,  batch_size = batch_size)
print(len(data))
independent_metrics_res, independent_ys_res, independent_attn_res= eval_step(model, loader, fold, ep_best, epochs,dir_head,use_cuda)
data['y_pred'], data['y_prob']= independent_metrics_res[1],independent_metrics_res[2]
print("data:",data)
data.to_csv('./extend_testdata/result_mira.csv',index=False)