from Bio.Align import substitution_matrices
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from collections import Counter
def get_numbering(seqs, ):
    """
    get the IMGT numbering of CDR3 with ANARCI tool
    """
    template = ['GVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGTTDQGEVPNGYNVSRSTIEDFPLRLLSAAPSQTSVYF', 'GEGSRLTVL']
    # # save fake tcr file
    save_path = 'tmp_faketcr.fasta'
    id_list = []
    seqs_uni = np.unique(seqs)
    with open(save_path, 'w+') as f:
        for i, seq in enumerate(seqs_uni):
            f.write('>'+str(i)+'\n')
            id_list.append(i)
            total_seq = ''.join([template[0], seq ,template[1]])
            f.write(str(total_seq))
            f.write('\n')
    print('Save fasta file to '+save_path + '\n Aligning...')
    df_seqs = pd.DataFrame(list(zip(id_list, seqs_uni)), columns=['Id', 'cdr3'])
    
    # # using ANARCI to get numbering file

   # this environment name should be the same as the one you install anarci
    cmd = ("conda run -n TranTCR "  # this environment name should be the same as the one you install anarci
            " ANARCI"
            " -i tmp_faketcr.fasta  -o tmp_align --csv -p 24")
    res = os.system(cmd)
    
    # # parse numbered seqs data
    try:
        df = pd.read_csv('tmp_align_B.csv')
    except FileNotFoundError:
        raise FileNotFoundError('Error: ANARCI failed to align, please check whether ANARCI exists in your environment')
        
    cols = ['104', '105', '106', '107', '108', '109', '110', '111', '111A', '111B', '112C', '112B', '112A', '112', '113', '114', '115', '116', '117', '118']
    seqs_al = []
    for col in cols:
        if col in df.columns:
            seqs_al_curr = df[col].values
            seqs_al.append(seqs_al_curr)
        else:
            seqs_al_curr = np.full([len(df)], '-')
            seqs_al.append(seqs_al_curr)
    seqs_al = [''.join(seq) for seq in np.array(seqs_al).T]
    df_al = df[['Id']]
    df_al['cdr3_align'] = seqs_al
    
    ## merge
    # os.remove('tmp_align_B.csv')
#     os.remove('tmp_faketcr.fasta')
    df = df_seqs.merge(df_al, how='inner', on='Id')
    df = df.set_index('cdr3')
    return df.loc[seqs, 'cdr3_align'].values


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        shape = [input.shape[0]] + list(self.shape)
        return input.view(*shape)
def load_ae_model(tokenizer, path='./epi_ae.ckpt',):
    # tokenizer = Tokenizer()
    ## load model
    model_args = dict(
        tokenizer = tokenizer,
        dim_hid = 32,
        len_seq = 12,
    )
    model = AutoEncoder(**model_args)
    model.eval()

    ## load weights
    state_dict = torch.load(path, map_location=device)
    state_dict = {k[6:]:v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model
class AutoEncoder(nn.Module):
    def __init__(self, 
        tokenizer,
        dim_hid,
        len_seq,
    ):
        super().__init__()
        embedding = tokenizer.embedding_mat()
        vocab_size, dim_emb = embedding.shape
        self.embedding_module = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), padding_idx=0, )
        self.encoder = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Conv1d(dim_hid, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )

        self.seq2vec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len_seq * dim_hid, dim_hid),
            nn.ReLU()
        )
        self.vec2seq = nn.Sequential(
            nn.Linear(dim_hid, len_seq * dim_hid),
            nn.ReLU(),
            View(dim_hid, len_seq)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(dim_hid, vocab_size)

    def forward(self, inputs):
        inputs = inputs.long()
        seq_emb = self.embedding_module(inputs)
        
        seq_enc = self.encoder(seq_emb.transpose(1, 2))
        vec = self.seq2vec(seq_enc)
        seq_repr = self.vec2seq(vec)
        indices = None
        seq_dec = self.decoder(seq_repr)
        out = self.out_layer(seq_dec.transpose(1, 2))
        return out, seq_enc, vec, indices
def GetBlosumMat(residues_list):
    n_residues = len(residues_list)  # the number of amino acids _ 'X'
    blosum62_mat = np.zeros([n_residues, n_residues])  # plus 1 for gap
    bl_dict = substitution_matrices.load('BLOSUM62')
    for pair, score in bl_dict.items():
        if (pair[0] not in residues_list) or (pair[1] not in residues_list):  # special residues not considered here
            continue
        idx_pair0 = residues_list.index(pair[0])  # index of residues
        idx_pair1 = residues_list.index(pair[1])
        blosum62_mat[idx_pair0, idx_pair1] = score
        blosum62_mat[idx_pair1, idx_pair0] = score
    return blosum62_mat
class Tokenizer:
    def __init__(self,):
        self.res_all = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'N',
                     'E', 'K', 'Q', 'M', 'S', 'T', 'C', 'P', 'H', 'R'] #+ ['X'] #BJZOU
        self.tokens = ['-'] + self.res_all # '-' for padding encoding

    def tokenize(self, index): # int 2 str
        return self.tokens[index]

    def id(self, token): # str 2 int
        try:
            return self.tokens.index(token.upper())
        except ValueError:
            print('Error letter in the sequences:', token)
            if str.isalpha(token):
                return self.tokens.index('X')

    def tokenize_list(self, seq):
        return [self.tokenize(i) for i in seq]

    def id_list(self, seq):
        return [self.id(s) for s in seq]

    def embedding_mat(self):
        blosum62 = GetBlosumMat(self.res_all)
        mat = np.eye(len(self.tokens))
        mat[1:len(self.res_all) + 1, 1:len(self.res_all) + 1] = blosum62
        return mat
def encode_cdr3(cdr3, tokenizer):
    len_cdr3 = [len(s) for s in cdr3]
    max_len_cdr3 = np.max(len_cdr3)
    assert max_len_cdr3 <= 20, 'The cdr3 length must <= 20'
    max_len_cdr3 = 20
    
    seqs_al = get_numbering(cdr3)
    num_samples = len(seqs_al)

    # encoding
    encoding_cdr3 = np.zeros([num_samples, max_len_cdr3], dtype='int32')
    for i, seq in enumerate(seqs_al):
        encoding_cdr3[i, ] = tokenizer.id_list(seq)
    return encoding_cdr3

def encode_epi(epi, tokenizer):
    tokenizer = Tokenizer()
    encoding_epi = np.zeros([len(epi),12], dtype='int32')
    for i, seq in enumerate(epi):
        len_epi = len(seq)
        
        if len_epi == 8:
        
            encoding_epi[i,2:len_epi+2] = tokenizer.id_list(seq)
        elif (len_epi == 9) or (len_epi == 10) or (len_epi ==11):
            
            encoding_epi[i,1:len_epi+1] = tokenizer.id_list(seq)
        else:
            
            encoding_epi[i,:len_epi] = tokenizer.id_list(seq)
    print(encoding_epi)
    return encoding_epi
def make_data(data):
#     labels = []
    cdr3 = data['CDR3'].values
    epitope = data['Epitope'].values
    labels = data['label'].values
    mat = Tokenizer() 
    hla_inputs = encode_cdr3(cdr3, mat)
    pep_inputs = encode_epi(epitope,mat)

    return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels)

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs,labels):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels
        

    def __len__(self): # 样本数
        return self.pep_inputs.shape[0] # 改成hla_inputs也可以哦！

    def __getitem__(self, idx):
#         return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx],self.pep_lens[idx]
        return self.pep_inputs[idx],self.hla_inputs[idx], self.labels[idx]
def performances(y_true, y_pred, y_prob, print_ = True):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    try:
        mcc = ((tp*tn) - (fn*fp)) / np.sqrt(np.float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
    except:
        print('MCC Error: ', (tp+fn)*(tn+fp)*(tp+fp)*(tn+fn))
        mcc = np.nan
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    try:
        recall = tp / (tp+fn)
    except:
        recall = np.nan
        
    try:
        precision = tp / (tp+fp)
    except:
        precision = np.nan
        
    try: 
        f1 = 2*precision*recall / (precision+recall)
    except:
        f1 = np.nan
        
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    
    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity, specificity, accuracy, mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))
        
    return (roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall, aupr)

def transfer(y_prob, threshold = 0.5):
    # return np.array([[0, 1][x > threshold] for x in y_prob])
    y_prob = np.array(y_prob)
    return np.where(y_prob > threshold, 1, 0)
f_mean = lambda l: sum(l)/len(l)

def performances_to_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'sensitivity', 'specificity', 'precision', 'recall', 'aupr']

    performances_pd = pd.DataFrame(performances_list, columns = metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis = 0)
    performances_pd.loc['std'] = performances_pd.std(axis = 0)
    
    return performances_pd