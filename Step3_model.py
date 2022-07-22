# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-model.py
@time:2021/9/15 16:33 
"""

import os
import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
import copy
import time
import pickle

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import dropout, nn
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler

from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE
from model_helper import Encoder_MultipleLayers, Embeddings
from Step2_DataEncoding import DataEncoding

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df, rna_df):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df = drug_df
        self.rna_df = rna_df


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.drug_df.iloc[index]['drug_encoding']
        v_p = np.array(self.rna_df.iloc[index])
        y = self.labels[index]

        #print(v_d)
        #print(type(v_d))
        #print(type(v_p))
        #print(type(y))

        return v_d, v_p, y


class transformer(nn.Sequential):
    def __init__(self, input_dim_drug,
                 transformer_emb_size_drug, dropout,
                 transformer_n_layer_drug,
                 transformer_intermediate_size_drug,
                 transformer_num_attention_heads_drug,
                 transformer_attention_probs_dropout,
                 transformer_hidden_dropout_rate):
        super(transformer, self).__init__()

        self.emb = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              dropout)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

    def forward(self, v):
        e = v[0].long().to(device)
        e_mask = v[1].long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]


class MLP(nn.Sequential):
    def __init__(self, input_dim):
        input_dim_gene = input_dim
        hidden_dim_gene = 256
        mlp_hidden_dims_gene = [1024, 256, 64]
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v


class Classifier(nn.Sequential):
    def __init__(self, args, model_drug, model_gene):
        super(Classifier, self).__init__()
        self.input_dim_drug = args.input_dim_drug_classifier
        self.input_dim_gene = args.input_dim_gene_classifier
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_dims = [1024, 1024, 512]
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_gene] + \
            self.hidden_dims + [1]
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v_D, v_P):
        # each encoding
        v_D = self.model_drug(v_D)
        v_P = self.model_gene(v_P)
        # concatenate and classify
        v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f


class DeepTTC:
    def __init__(self, modeldir, args):
        self.model_drug = transformer(args.input_dim_drug,
                                 args.transformer_emb_size_drug,
                                 args.dropout,
                                 args.transformer_n_layer_drug,
                                 args.transformer_intermediate_size_drug,
                                 args.transformer_num_attention_heads_drug,
                                 args.transformer_attention_probs_dropout,
                                 args.transformer_hidden_dropout_rate)
        self.device = torch.device('cuda:0')
        self.modeldir = modeldir
        self.record_file = os.path.join(
            self.modeldir, "valid_markdowntable.txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        self.args = args
        self.model = None

    def test(self, datagenerator, model):
        y_label = []
        y_pred = []
        model.eval()
        for i, (v_drug, v_gene, label) in enumerate(datagenerator):
            score = model(v_drug, v_gene)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            loss = loss_fct(n, Variable(torch.from_numpy(
                np.array(label)).float()).to(self.device))
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        model.train()

        return y_label, y_pred, \
            mean_squared_error(y_label, y_pred), \
            np.sqrt(mean_squared_error(y_label, y_pred)), \
            pearsonr(y_label, y_pred)[0], \
            pearsonr(y_label, y_pred)[1], \
            spearmanr(y_label, y_pred)[0], \
            spearmanr(y_label, y_pred)[1], \
            concordance_index(y_label, y_pred), \
            loss

    def train(self, train_drug, train_rna, val_drug, val_rna):
        model_gene = MLP(input_dim=np.shape(train_rna)[1])
        self.model = Classifier(self.args, self.model_drug, model_gene)

        lr = 1e-4
        decay = 0
        BATCH_SIZE = 64
        train_epoch = 3
        self.model = self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 5])
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=lr, weight_decay=decay)
        loss_history = []

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}
        training_generator = data.DataLoader(data_process_loader(
            train_drug.index.values, train_drug.Label.values, train_drug, train_rna), **params)
        validation_generator = data.DataLoader(data_process_loader(
            val_drug.index.values, val_drug.Label.values, val_drug, val_rna), **params)
        print(training_generator)

        max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ['# epoch', "MSE", 'RMSE',
                               "Pearson Correlation", "with p-value",
                               'Spearman Correlation', "with p-value2",
                               "Concordance Index"]
        table = PrettyTable(valid_metric_header)
        def float2str(x): return '%0.4f' % x
        print('--- Go for Training ---')
        #writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')
        t_start = time.time()
        iteration_loss = 0

        for epo in range(train_epoch):
            for i, (v_d, v_p, label) in enumerate(training_generator):
                # print(v_d,v_p)
                #v_d = v_d.float().to(self.device)
                score = self.model(v_d, v_p)
                label = Variable(torch.from_numpy(
                    np.array(label))).float().to(self.device)

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                #writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
                if (i % 1000 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) +
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] +
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

            with torch.set_grad_enabled(False):
                # regression: MSE, Pearson Correlation, with p-value, Concordance Index
                y_true, y_pred, mse, rmse, \
                    person, p_val, \
                    spearman, s_p_val, CI,\
                    loss_val = self.test(validation_generator, self.model)
                lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                   s_p_val, CI]))
                valid_metric_record.append(lst)
                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    max_MSE = mse
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' with loss:' + str(loss_val.item())[:7] +
                          ', MSE: ' + str(mse)[:7] +
                          ' , Pearson Correlation: ' + str(person)[:7] +
                          ' with p-value: ' + str(p_val)[:7] +
                          ' Spearman Correlation: ' + str(spearman)[:7] +
                          ' with p_value: ' + str(s_p_val)[:7] +
                          ' , Concordance Index: ' + str(CI)[:7])
            table.add_row(lst)

        self.model = model_max

        with open(self.record_file, 'w') as fp:
            fp.write(table.get_string())
        with open(self.pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)

        print('--- Training Finished ---')

    def predict(self, drug_data, rna_data):
        print('predicting...')
        self.model.to(device)
        info = data_process_loader(drug_data.index.values,
                                   drug_data.Label.values,
                                   drug_data, rna_data)
        params = {'batch_size': 16,
                  'shuffle': False,
                  'num_workers': 8,
                  'drop_last': False,
                  'sampler': SequentialSampler(info)}
        generator = data.DataLoader(info, **params)

        y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val = \
            self.test(generator, self.model)

        return y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI

    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)


    def preprocess(self, rna_data, drug_data, response_data, response_metric='AUC'):
        args = self.args
        obj = DataEncoding(args.vocab_dir, args.cancer_id, args.sample_id, args.target_id, args.drug_id)
        drug_smiles = drug_data

        drugid2smile = dict(
            zip(drug_smiles['DrugID'], drug_smiles['SMILES']))
        smile_encode = pd.Series(drug_smiles['SMILES'].unique()).apply(
            obj._drug2emb_encoder)
        uniq_smile_dict = dict(
            zip(drug_smiles['SMILES'].unique(), smile_encode))

        drug_data.drop(['SMILES'], inplace=True, axis=1)
        drug_data['smiles'] = [drugid2smile[i] for i in drug_data['DrugID']]
        drug_data['drug_encoding'] = [uniq_smile_dict[i]
                                      for i in drug_data['smiles']]
        drug_data = drug_data.reset_index()
        drug_data['Label'] = response_data['AUC']

        response_data = response_data[['CancID', 'DrugID']]
        rna_data = pd.merge(response_data, rna_data, on='CancID', how='inner')
        #train_rnadata = train_rnadata.T
        drug_data.index = range(drug_data.shape[0])
        rna_data.index = range(rna_data.shape[0])

        return rna_data, drug_data

if __name__ == '__main__':

    # step1 数据切分
    vocab_dir = '.'
    obj = DataEncoding(vocab_dir=vocab_dir)

    # 切分完成
    traindata, testdata = obj.Getdata.ByCancer(random_seed=1)
    # encoding 完成
    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)

    # step2：构造模型
    modeldir = './Model_80'
    modelfile = modeldir + '/model.pt'
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    net = DeepTTC(modeldir=modeldir)
    net.train(train_drug=traindata, train_rna=train_rnadata,
              val_drug=testdata, val_rna=test_rnadata)
    net.save_model()
    print("Model Saved :{}".format(modelfile))
