# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-step1_drugGet.py
@time:2021/8/10 15:48 
@从 pubchem数据库中，根据id查找smile
"""

import os,sys
import pandas as pd
import pubchempy as pcp
import pickle5 as pickle

pub_file = sys.argv[1]
pub_df = pd.read_csv(pub_file)
pub_df = pub_df.dropna(subset=['PubCHEM'])
pub_df = pub_df[(pub_df['PubCHEM']!='none') & (pub_df['PubCHEM']!='several')]
# pub_df = pub_df.head(20)
smile_list = []
inchi_list = []
for idx,row in pub_df.iterrows():
    print(idx)
    # drug_id = row['drug_id']
    # drug_name = row['Name']
    # drug_Synonyms = row['Synonyms']
    pubid = row['PubCHEM'].split(',')[0]
    print(pubid)
    compound = pcp.Compound.from_cid(pubid)
    smile = compound.isomeric_smiles
    smile_list.append(smile)
    print(compound.isomeric_smiles)
    inchi = compound.inchi
    inchi_list.append(inchi)
    print(inchi)

pub_df['smiles'] = smile_list
pub_df['inchi'] = inchi_list

# pub_df.to_pickle('smile_inchi.pkl',protocol=4)
pub_df.to_csv('smile_inchi.csv')