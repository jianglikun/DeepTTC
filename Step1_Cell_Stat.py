# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-Step1_stat.py
@time:2021/8/12 9:50 
"""

import sys
import pandas as pd
import numpy as np

cell_line_file = sys.argv[1]
cell_line_df = pd.read_excel(cell_line_file,sheet_name='Cell line details')
print(cell_line_df.head(5))
print(cell_line_df.shape)


print('#'*50)
print('\t 空值统计：')
print(cell_line_df.count())

print('#'*50)
print('1\t 共有细胞系：{}'.format(cell_line_df['COSMIC identifier'].value_counts().shape[0]))
print('2\t 每个细胞系都有一个独立的COSMIC id')
print('3\t 这些细胞系对应：{}个肿瘤类型'.format(
    cell_line_df['Cancer Type\n(matching TCGA label)'].value_counts().shape[0]))
print('\t 具体每个肿瘤对应的细胞系数据量是')
print('\t 其中UNABLE TO CLASSIFY，是不能分类的细胞系数据量')
print(cell_line_df['Cancer Type\n(matching TCGA label)'].value_counts())
