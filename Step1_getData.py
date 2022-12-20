# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-Step1_getData.py
@time:2021/8/12 15:48 
"""
import os
import sys
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import warnings
from pubchempy import download
import wget
import zipfile


# warnings.filterwarnings("ignore")


class GetData():
    def __init__(self, cancer_id, sample_id, target_id, drug_id, generate_smiles=True):
        candle_data_dir = os.getenv('CANDLE_DATA_DIR')
        if candle_data_dir is None:
            candle_data_dir = '.'
        PATH = os.path.join(candle_data_dir, 'GDSC_data')

        rnafile = PATH + '/Cell_line_RMA_proc_basalExp.txt'
        smilefile = PATH + '/smile_inchi.csv'
        pairfile = PATH + '/GDSC2_fitted_dose_response_25Feb20.xlsx'
        drug_infofile = PATH + "/Drug_listTue_Aug10_2021.csv"
        drug_thred = PATH + '/IC50_thred.txt'
        rna_url = 'https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources///Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip'

        self.cancer_id = cancer_id
        self.sample_id = sample_id
        self.target_id = target_id
        self.drug_id = drug_id
        self.generate_smiles = generate_smiles

        self.rna_url = rna_url
        self.PATH = PATH
        self.pairfile = pairfile
        self.drugfile = drug_infofile
        self.rnafile = rnafile
        self.smilefile = smilefile
        self.drug_thred = drug_thred

        self.drug_data = None
        self.rna_data = None

    def _create_smiles(self):
        if os.path.isfile(self.smilefile):
            return
        smile_file_tmp = 's.csv'
        drug_data = pd.read_csv(self.drugfile).astype(str)

        gdsc_ids_input = drug_data['drug_id']
        pubchem_cids_input = drug_data['PubCHEM']

        idx_to_keep = ~np.logical_or(
            pubchem_cids_input == 'nan', pubchem_cids_input == 'none')
        gdsc_ids_input = gdsc_ids_input[idx_to_keep]
        pubchem_cids_input = pubchem_cids_input[idx_to_keep]

        gdsc_ids = []
        pubchem_cids = []
        for id, cid_input in zip(gdsc_ids_input, pubchem_cids_input):
            cids = cid_input.strip(' ').split(',')
            for cid in cids:
                pubchem_cids.append(int(cid))
                gdsc_ids.append(int(id))

        print(list(pubchem_cids))
        download('CSV', smile_file_tmp,
                 pubchem_cids,
                 operation='property/CanonicalSMILES,IsomericSMILES',
                 overwrite=True)

        smile_data = pd.read_csv(smile_file_tmp)
        smile_data['drug_id'] = gdsc_ids
        print(smile_data)
        print(self.smilefile)
        smile_data.to_csv(self.smilefile)
        os.remove(smile_file_tmp)

    def setDrug(self, drug_data):
        self.drug_data = drug_data

    def getDrug(self):
        # 读取 smile_inchi.csv
        if self.drug_data is not None:
            return self.drug_data
        if self.generate_smiles:
            self._create_smiles()
        drugdata = pd.read_csv(self.smilefile, index_col=None)
        return drugdata

    def _filter_pair(self, drug_cell_df):
        print("#"*50)
        print("step1 过滤细胞系....")
        print("在检查细胞系rna 表达矩阵的时候发现4个细胞系没有表达记录")
        # ['DATA.908134', 'DATA.1789883', 'DATA.908120', 'DATA.908442'] not in index
        not_index = [908134, 1789883, 908120, 908442]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[~drug_cell_df[self.sample_id].isin(
            not_index)]
        print(drug_cell_df.shape)

        print("step2 过滤药物....")
        print("对于部分Drug没有记录PuchemID，得不到smile")
        pub_df = pd.read_csv(self.drugfile)
        pub_df = pub_df.dropna(subset=['PubCHEM'])
        pub_df = pub_df[(pub_df['PubCHEM'] != 'none') &
                        (pub_df['PubCHEM'] != 'several')]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[drug_cell_df[self.drug_id].isin(
            pub_df['drug_id'])]
        print(drug_cell_df.shape)
        return drug_cell_df

    def _stat_cancer(self, drug_cell_df):
        print("#" * 50)
        cancer_num = drug_cell_df[self.cancer_id].value_counts().shape[0]
        print('#\t 癌症类型一共有：{}'.format(cancer_num))
        min_cancer_drug = min(drug_cell_df[self.cancer_id].value_counts())
        max_cancer_drug = max(drug_cell_df[self.cancer_id].value_counts())
        mean_cancer_drug = np.mean(drug_cell_df[self.cancer_id].value_counts())
        print('#\t 其中最少的癌症类型对应{}个药物，\n\t 最多的对应{}个药物，\n\t 平均对应{}个药物'.format(
            min_cancer_drug, max_cancer_drug, mean_cancer_drug))

    def _stat_cell(self, drug_cell_df):
        print("#" * 50)
        cell_num = drug_cell_df[self.sample_id].value_counts().shape[0]
        print('#\t 使用的细胞系有：{}'.format(cell_num))
        min_drug = min(drug_cell_df[self.sample_id].value_counts())
        max_drug = max(drug_cell_df[self.sample_id].value_counts())
        mean_drug = np.mean(drug_cell_df[self.sample_id].value_counts())
        print('#\t 其中最少的细胞系对应{}个药物，\n\t 最多的对应{}个药物，\n\t 平均对应{}个药物'.format(
            min_drug, max_drug, mean_drug))

    def _stat_drug(self, drug_cell_df):
        print("#" * 50)
        drug_num = drug_cell_df[self.drug_id].value_counts().shape[0]
        print('#\t 使用的药物有：{}'.format(drug_num))
        min_cell = min(drug_cell_df[self.drug_id].value_counts())
        max_cell = max(drug_cell_df[self.drug_id].value_counts())
        mean_cell = np.mean(drug_cell_df[self.drug_id].value_counts())
        print('#\t 其中最少的药物对应{}个细胞系，\n\t 最多的对应{}个细胞系，\n\t 平均对应{}个细胞系'.format(
            min_cell, max_cell, mean_cell))

    def _split(self, df, col, ratio, random_seed):

        col_list = df[col].value_counts().index
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for instatnce in col_list:
            sub_df = df[df[col] == instatnce]
            sub_df = sub_df[[self.drug_id, self.sample_id,
                             self.cancer_id, self.target_id]]
            ## 按照 col 来拆分数据集 ##
            # 对于任意一个 instance，1 - ratio 的用于训练，10=test，10=validation
            sub_train, sub_test = train_test_split(
                sub_df, test_size=ratio, random_state=random_seed)
            if train_data.shape[0] == 0:
                train_data = sub_train
                test_data = sub_test
            else:
                train_data = train_data.append(sub_train)
                test_data = test_data.append(sub_test)
        print('#' * 50)
        print('#\t 数据对一共有：{}'.format(df.shape[0]))
        print('#\t 按照{}对数据进行切割，对于每个instance，{}的数据进行训练，{}的数据进行验证'.format(
            col, (1-ratio), ratio))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))

        return train_data, test_data

    def setRnaData(self, rna_data):
        self.rna_data = rna_data

    def getRnaData(self):
        return self.rna_data

    def ByCancer(self, random_seed):

        # 理解作者的意思就是按照 癌症类型，随机选95的作为训练
        # 评价没有癌症的准确性，评价不同药物的准确性

        if self.rna_data is not None:
            drug_cell_df = self.rna_data
        else:
            drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)

        #drug_cell_df = drug_cell_df.head(10000)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        print(drug_cell_df[self.cancer_id].value_counts())

        train_data, test_data = self._split(df=drug_cell_df, col=self.cancer_id,
                                            ratio=0.2, random_seed=random_seed)

        return train_data, test_data

    def ByDrug(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        train_data, test_data = self._split(
            df=drug_cell_df, col=self.drug_id, ratio=0.2)

        return train_data, test_data

    def ByCell(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        train_data, test_data = self._split(
            df=drug_cell_df, col=self.sample_id, ratio=0.2)

        return train_data, test_data

    def MissingData(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        cell_list = drug_cell_df[self.sample_id].value_counts().index
        drug_list = drug_cell_df[self.drug_id].value_counts().index

        all_df = pd.DataFrame()
        dup_drug = []
        [dup_drug.extend([i]*len(cell_list)) for i in drug_list]
        all_df[self.drug_id] = dup_drug

        dup_cell = []
        for i in range(len(drug_list)):
            dup_cell.extend(cell_list)
        all_df[self.sample_id] = dup_cell

        all_df['ID'] = all_df[self.drug_id].astype(str).str.cat(
            all_df[self.sample_id].astype(str), sep='_')
        drug_cell_df['ID'] = drug_cell_df[self.drug_id].astype(
            str).str.cat(drug_cell_df[self.sample_id].astype(str), sep='_')
        MissingData = all_df[~all_df['ID'].isin(drug_cell_df['ID'])]

        print("#"*50)
        print('使用药物{}个，细胞系有{}个'.format(len(drug_list), len(cell_list)))
        print('理论上，每种药物都作用所有细胞系的话，应该有{} Pairs'.format(
            len(drug_list)*len(cell_list)))
        print('但是有的药物和细胞系没有做实验，共有{} Pairs'.format(MissingData.shape[0]))

        # drug_cell_df = drug_cell_df[[self.sample_id, self.cancer_id]].drop_duplicates()
        # cell2cancer_dict = pd.Series(list(drug_cell_df[self.cancer_id]), index=drug_cell_df[self.sample_id])

        return drug_cell_df, MissingData

    def _LeaveOut(self, df, col, ratio=0.8, random_num=1):
        random.seed(random_num)
        col_list = list(set(df[col]))
        col_list = list(col_list)

        sub_start = int(len(col_list)/5)*random_num
        if random_num == 4:
            sub_end = len(col_list)
        else:
            sub_end = int(len(col_list)/5)*(random_num+1)

        # leave_instatnce = random.sample(col_list,int(len(col_list)*ratio))
        leave_instatnce = list(
            set(col_list) - set(col_list[sub_start:sub_end]))

        df = df[[self.drug_id, self.sample_id, self.cancer_id, self.target_id]]
        train_data = df[df[col].isin(leave_instatnce)]
        test_data = df[~df[col].isin(leave_instatnce)]

        print('#' * 50)
        print(len(col_list))
        print(len(set(list(train_data[col]))))
        print(len(set(list(test_data[col]))))
        print('#\t 数据对一共有：{}，leave out 方法'.format(df.shape[0]))
        print('#\t 按照{}对数据进行划分，对于每个instance，{}的数据进行训练'.format(col, ratio))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))

        return train_data, test_data

    def Cell_LeaveOut(self, random):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        traindata, testdata = self._LeaveOut(
            df=drug_cell_df, col=self.sample_id, ratio=0.8, random_num=random)

        return traindata, testdata

    def Drug_LeaveOut(self, random):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        traindata, testdata = self._LeaveOut(
            df=drug_cell_df, col=self.drug_id, ratio=0.8, random_num=random)

        return traindata, testdata

    def Drug_Thred(self):
        thred_data = pd.read_csv(self.drug_thred, sep='\t')
        thred_df = thred_data.T
        thred_df['drug_name'] = thred_df.index
        thred_df['threds'] = thred_df[0]
        thred_df = thred_df.drop(0, axis=1)
        thred_df.loc['VX-680', 'drug_name'] = 'Tozasertib'
        thred_df.loc['Mitomycin C', 'drug_name'] = 'Mitomycin-C'
        thred_df.loc['HG-6-64-1', 'drug_name'] = 'HG6-64-1'
        thred_df.loc['BAY 61-3606', 'drug_name'] = 'BAY-61-3606'
        thred_df.loc['Zibotentan, ZD4054', 'drug_name'] = 'Zibotentan'
        thred_df.loc['PXD101, Belinostat', 'drug_name'] = 'Belinostat'
        thred_df.loc['NU-7441', 'drug_name'] = 'NU7441'
        thred_df.loc['BIRB 0796', 'drug_name'] = 'BIRB-796'
        thred_df.loc['Nutlin-3a', 'drug_name'] = 'Nutlin-3a (-)'
        thred_df.loc['AZD6482.1', 'drug_name'] = 'AZD6482'
        thred_df.loc['BMS-708163.1', 'drug_name'] = 'BMS-708163'
        thred_df.loc['BMS-536924.1', 'drug_name'] = 'BMS-536924'
        thred_df.loc['GSK269962A.1', 'drug_name'] = 'GSK269962A'
        thred_df.loc['SB-505124', 'drug_name'] = 'SB505124'
        thred_df.loc['JQ1.1', 'drug_name'] = 'JQ1'
        thred_df.loc['UNC0638.1', 'drug_name'] = 'UNC0638'
        thred_df.loc['CHIR-99021.1', 'drug_name'] = 'CHIR-99021'
        thred_df.loc['piperlongumine', 'drug_name'] = 'Piperlongumine'
        thred_df.loc['PLX4720 (rescreen)', 'drug_name'] = 'PLX4720'
        thred_df.loc['Afatinib (rescreen)', 'drug_name'] = 'Afatinib'
        thred_df.loc['Olaparib.1', 'drug_name'] = 'Olaparib'
        thred_df.loc['AZD6244.1', 'drug_name'] = 'AZD6244'
        thred_df.loc['Bicalutamide.1', 'drug_name'] = 'Bicalutamide'
        thred_df.loc['RDEA119 (rescreen)', 'drug_name'] = 'RDEA119'
        thred_df.loc['GDC0941 (rescreen)', 'drug_name'] = 'GDC0941'
        thred_df.loc['MLN4924 ', 'drug_name'] = 'MLN4924'
        # only one I-BET 151

        drug_info = pd.read_csv(self.drugfile)
        drugname2drugid = {}
        drugid2pubchemid = {}
        for idx, row in drug_info.iterrows():
            name = row['Name']
            drug_id = row['drug_id']
            pub_id = row['PubCHEM']
            drugname2drugid[name] = drug_id
            drugid2pubchemid[drug_id] = pub_id

        drug_info_filter_name = drug_info.dropna(subset=['Synonyms'])
        for idx, row in drug_info_filter_name.iterrows():
            name = row['Name']
            pub_id = row['PubCHEM']
            drug_id = row['drug_id']
            drugname2drugid[name] = drug_id
            Synonyms_list = row['Synonyms'].split(', ')
            for drug in Synonyms_list:
                drugname2drugid[drug] = drug_id

        drugid2thred = {}
        for idx, row in thred_df.iterrows():
            name = row['drug_name']
            thred = row['threds']
            if name in drugname2drugid:
                drugid2thred[drugname2drugid[name]] = thred

        id_li = []
        PubChem_li = []
        thred_li = []
        for i in drugid2thred:
            id_li.append(i)
            PubChem_li.append(drugid2pubchemid[i])
            thred_li.append(drugid2thred[i])

        # data = pd.DataFrame()
        # data['Drug_id'] = id_li
        # data['PubChem'] = PubChem_li
        # data['Thred'] = thred_li
        #
        # print(data)
        # data.to_csv('Drug_Thred.csv')
        drug_list = [drugname2drugid[i]
                     for i in list(thred_df['drug_name']) if i in drugname2drugid]

        return drug_list, drugid2thred

    def _split_no_balance_binary(self, df, col, ratio, random_seed):

        col_list = df[col].value_counts().index
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for instatnce in col_list:
            sub_df = df[df[col] == instatnce]
            sub_df = sub_df[[self.drug_id, self.sample_id,
                             self.cancer_id, self.target_id, 'Binary_IC50']]
            ## 按照 col 来拆分数据集 ##
            # 对于任意一个 instance，1 - ratio 的用于训练，10=test，10=validation
            sub_train, sub_test = train_test_split(sub_df, test_size=ratio,
                                                   random_state=random_seed)
            if train_data.shape[0] == 0:
                train_data = sub_train
                test_data = sub_test
            else:
                train_data = train_data.append(sub_train)
                test_data = test_data.append(sub_test)
        print('#' * 50)
        print('#\t 数据对一共有：{}'.format(df.shape[0]))
        print('#\t 按照{}对数据进行切割，对于每个instance，{}的数据进行训练，{}的数据进行验证'.format(
            col, (1-ratio), ratio))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))

        return train_data, test_data

    def _split_balance_binary(self, df, col, ratio, random_seed):

        col_list = df[col].value_counts().index

        pos_data = df[df[col] == 1]
        neg_data = df[df[col] == 0]

        down_pos_data = pos_data.loc[random.sample(
            list(pos_data.index), neg_data.shape[0])]

        combine_data = neg_data.append(down_pos_data)

        combine_data = combine_data[[
            self.drug_id, self.sample_id, self.cancer_id, self.target_id, 'Binary_IC50']]

        train_data, test_data = train_test_split(combine_data, test_size=ratio,
                                                 random_state=random_seed)

        print('#' * 50)
        print('#\t 数据对一共有：{}'.format(df.shape[0]))
        print('#\t 构建平衡数据集，{}为大于-2的样本，{}为小于-2的样本,选择1：1的样本各{}个'.format(
            pos_data.shape[0], neg_data.shape[0], neg_data.shape[0]))
        print('#\t 按照{}对数据进行切割，对于每个instance，{}的数据进行训练，{}的数据进行验证'.format(
            col, (1-ratio), ratio))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))

        return train_data, test_data

    def ByBinary(self, random_num):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        drug_list, drugid2thred = self.Drug_Thred()
        ##################################################
        # 按照每种药物得阈值,第一种，直接过滤
        Binary_Drug_list = []
        drug_cell_df = drug_cell_df[drug_cell_df[self.drug_id].isin(drug_list)]

        # print(drug_cell_df[self.drug_id].value_counts().shape)
        for idx, row in drug_cell_df.iterrows():
            drug_name = row['DRUG_NAME']
            drug_id = row[self.drug_id]
            ic50 = row[self.target_id]
            if (ic50 > drugid2thred[drug_id]):
                Binary_Drug_list.append(1)
            else:
                Binary_Drug_list.append(0)
        # 数量：2811*2 = Train * 4497 + Test 1125
        drug_cell_df['Binary_IC50'] = Binary_Drug_list
        ############################################################################
        # 第二种，补充-2的阈值
        # Binary_Drug_list = []
        #
        # print(drug_cell_df[self.drug_id].value_counts().shape)
        # for idx, row in drug_cell_df.iterrows():
        #     drug_name = row['DRUG_NAME']
        #     drug_id = row[self.drug_id]
        #     ic50 = row[self.target_id]
        #     if drug_id in drug_list:
        #         if ic50 > drugid2thred[drug_id]:
        #             Binary_Drug_list.append(1)
        #         else:
        #             Binary_Drug_list.append(0)
        #     else:
        #         if ic50 > -2:
        #             Binary_Drug_list.append(1)
        #         else:
        #             Binary_Drug_list.append(0)
        # drug_cell_df['Binary_IC50'] = Binary_Drug_list

        ############################################################################
        # 第三种 直接使用-2的阈值
        # Binary_IC50_list = []
        # for ic50 in drug_cell_df[self.target_id]:
        #     if ic50 > -2:
        #         Binary_IC50_list.append(1)
        #     else:
        #         Binary_IC50_list.append(0)
        # drug_cell_df['Binary_IC50'] = Binary_IC50_list
        # 数量：9102*2 = Train 14571 + Test 3643

        #############################################################################
        # print(drug_cell_df['Binary_IC50'].value_counts())
        train_data, test_data = self._split_balance_binary(df=drug_cell_df, col='Binary_IC50',
                                                           ratio=0.2, random_seed=random_num)
        print(train_data, test_data)

        return train_data, test_data

    def getRna(self, traindata, testdata, args=None):
        train_rnaid = list(traindata[self.sample_id])
        test_rnaid = list(testdata[self.sample_id])
        train_rnaid = ['DATA.'+str(i) for i in train_rnaid]
        test_rnaid = ['DATA.' + str(i) for i in test_rnaid]

        if not os.path.isfile(self.rnafile):
            rna_zip = self.rnafile+'.zip'
            wget.download(self.rna_url, out=rna_zip)
            with zipfile.ZipFile(rna_zip, "r") as zip_ref:
                zip_ref.extractall(self.PATH)

        rnadata = pd.read_csv(self.rnafile, sep='\t', index_col=0)
        print('ORIGINAL DATA')
        print(rnadata)

        if args is not None:
            if args.use_lincs:
                with open(f"{args.candle_data_dir}/landmark_genes") as f:
                    genes = [str(line.rstrip()) for line in f]
                #genes = ["ge_" + str(g) for g in genes]
                print('Genes!!!')
                print(genes)
                print('Train RNA Columns!!!')
                genes_index = rnadata.index #['GENE_SYMBOLS']
                print(genes_index)
                print(len(set(genes).intersection(set(genes_index))))
                genes = list(set(genes).intersection(set(genes_index)))
                rnadata = rnadata.loc[genes]
                

        train_rnadata = rnadata[train_rnaid]
        test_rnadata = rnadata[test_rnaid]

        return train_rnadata, test_rnadata


if __name__ == '__main__':
    obj = GetData()
