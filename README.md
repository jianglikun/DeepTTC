# DeepTTC
DeepTTC: a transformer-based model for predicting cancer drug response

Identification of new lead molecules for treatment  of cancer is often the result of more than a decade of dedicated efforts. Before these painstakingly selected drug candidates are used in clinic , their anti-cancer activity was generally varified by vitro cell experiments. Therefore, accurately prediction  cancer drug response is the key and challenging task in anti-cancer drugs design and precision medicine. With the development of pharmacogenomics, combining efficient methods of extracting drug features and omic data makes it possible to improve drug response prediction. In this study, we propose DeepTTC, an novel  end-to-end deep learning model to predict the anti-cancer drug response. DeepTTC takes gene expression data and chemical strcutures  of drug for drug response prediction. Specifically, get inspiration from natural language processing, DeepTTC use  transformer for drug representation learning. Compared to existing methods, DeepTTC achieved higher performance in terms of RMSE, Pearson correlation coefficient and Spearman's rank correlation coefficient on multiple test sets. Moreover, we explorated  identify multiple clinical  indications anti-cancer drugs. With the outstanding preformance, DeepTTC is expected to the  most effective in silico  measure  in cancer drug design.

## Installation steps

1. Build Singularity image
```          
./build.sh DeepTTC.build
```
2. Run Singularity image
```
./run.sh writable/DeepTTC_YYYY_MM_DD
```
3. Navigate to DeepTTC folder
```
cd $PATH_TO_WRITABLE$/DeepTTC_YYYY_MM_DD/DeepTTC/
```


## Run
```
python3 DeepTTC_candle.py --config_file DeepTTC.build
```

## Configuration Parameters

```
mode - Execution mode. Available modes are: 'run', 'benchmark
generate_input_data - 'True' for generating input data anew, 'False' for using stored data
benchmark_result_dir - Directory for benchmark output
use_lincs - Whether to use a LINCS subset of genes ONLY

cancer_id - Column name for cancer
drug_id - Column name for drug
target_id - Column name for target
sample_id - Column name for samples/cell lines

train_data_drug - Drug data for training
test_data_drug - Drug data for testing
train_data_rna - RNA data for training
test_data_rna - RNA data for testing

benchmark_dir - Directory with the input data for benchmarking
output_dir - Output directory for saving model
model_name - File name for the model
rng_seed - Seed for the random number generator
vocab_dir - Directory with ESPF vocabulary

input_dim_drug - Input size of the drug transformer
transformer_emb_size_drug - Size of the drug embeddings
dropout - Dropout rate for classifiers
transformer_n_layer_drug - Number of layers for drug transformer
transformer_intermediate_size_drug - Size of the intermediate drug layers
transformer_num_attention_heads_drug - number of attention heads for drug transformer
transformer_attention_probs_dropout - Dropout rate for the attention layers
transformer_hidden_dropout_rate - Dropout rate for the transformer networks' hidden layers
learning_rate - Learning rate of the 
batch_size - Size of the training batches

input_dim_drug_classifier - Input size of the drug classifier
input_dim_gene_classifier - Input size of the gene classifier
```