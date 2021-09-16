# DeepTTC
DeepTTC: a transformer-based model for predicting cancer drug response

Identification of new lead molecules for treatment  of cancer is often the result of more than a decade of dedicated efforts. Before these painstakingly selected drug candidates are used in clinic , their anti-cancer activity was generally varified by vitro cell experiments. Therefore, accurately prediction  cancer drug response is the key and challenging task in anti-cancer drugs design and precision medicine. With the development of pharmacogenomics, combining efficient methods of extracting drug features and omic data makes it possible to improve drug response prediction. In this study, we propose DeepTTC, an novel  end-to-end deep learning model to predict the anti-cancer drug response. DeepTTC takes gene expression data and chemical strcutures  of drug for drug response prediction. Specifically, get inspiration from natural language processing, DeepTTC use  transformer for drug representation learning. Compared to existing methods, DeepTTC achieved higher performance in terms of RMSE, Pearson correlation coefficient and Spearman's rank correlation coefficient on multiple test sets. Moreover, we explorated  identify multiple clinical  indications anti-cancer drugs. With the outstanding preformance, DeepTTC is expected to the  most effective in silico  measure  in cancer drug design.

## Install

```          
biopython            1.78                                          
matplotlib           3.1.3                      
numpy                1.18.1                        
pandas               1.0.3                      
Pillow               7.0.0                                       
prettytable          0.7.2                                             
requests             2.23.0                     
scikit-learn         0.22.1                         
scipy                1.4.1              
subword-nmt          0.3.7              
tensorboard          2.1.0              
terminado            0.8.3                           
torch                1.4.0              
torchsummary         1.5.1              
torchvision          0.5.0                         
```

## Run Step

```
step1
根据GDSC给的pubchem id得到smiles inchi
python Step1_PubchemID2smile.py GDSC_data/Drug_listTue_Aug10_2021.csv 


step2 
先写了一个类，用来得到 drug_id 对应 cosmic_id 的rna数据，可调用
python Step1_getData.py


step3 
构建模型
python Step3_model.py

```
