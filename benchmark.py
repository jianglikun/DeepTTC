import pickle
import sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

sources = ['ccle', 'ctrp', 'gcsi', 'gdsc1', 'gdsc2']

def groupby_src_and_print(df, print_fn=print):
    print_fn(df.groupby('SOURCE').agg({'CancID': 'nunique', 'DrugID': 'nunique'}).reset_index())

def load_source(src, data_dir):
    pretty_indent = '#' * 10
    print(f'{pretty_indent} {src.upper()} {pretty_indent}')
    datadir = f"{data_dir}/ml.dfs/July2020/data.{src}"

    # Load data
    responses = pd.read_csv(f"{datadir}/rsp_{src}.csv")      # Drug response
    gene_expression = pd.read_csv(f"{datadir}/ge_{src}.csv")        # Gene expressions
    mordred_descriptors = pd.read_csv(f"{datadir}/mordred_{src}.csv")  # Mordred descriptors
    morgan_fingerprints = pd.read_csv(f"{datadir}/ecfp2_{src}.csv")    # Morgan fingerprints
    smiles = pd.read_csv(f"{datadir}/smiles_{src}.csv")   # SMILES

    # Use landmark genes
    use_lincs = True
    if use_lincs:
        with open(f"{datadir}/../landmark_genes") as f:
            genes = [str(line.rstrip()) for line in f]
    genes = ["ge_" + str(g) for g in genes]
    print(len(set(genes).intersection(set(gene_expression.columns[1:]))))
    genes = list(set(genes).intersection(set(gene_expression.columns[1:])))
    cols = ["CancID"] + genes
    gene_expression = gene_expression[cols]

    groupby_src_and_print(responses)
    print("Unique cell lines with gene expressions", gene_expression["CancID"].nunique())
    print("Unique drugs with Mordred", mordred_descriptors["DrugID"].nunique())
    print("Unique drugs with ECFP2", morgan_fingerprints["DrugID"].nunique())
    print("Unique drugs with SMILES", smiles["DrugID"].nunique())
    return gene_expression, mordred_descriptors, morgan_fingerprints, smiles, responses


def deep_ttc_preprocess(gene_expression, smiles, responses, model):
    gene_expression, drug_data = model.preprocess(gene_expression, smiles, responses, 'AUC')
    #gene_expression = gene_expression.drop(['index'], axis=1)
    drug_data = drug_data.drop(['index'], axis=1)
    data = pd.merge(gene_expression, drug_data, on='DrugID', how='inner')
    gene_expression = gene_expression.drop(['CancID', 'DrugID'], axis=1)
    gene_expression_columns = gene_expression.columns
    drug_columns = drug_data.columns

    return data, gene_expression_columns, drug_columns

def score(y_true, y_pred):
    scores = {}
    scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
    scores['mean_absolute_error']   = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
    scores['spearmanr'] = spearmanr(y_true, y_pred)[0]
    scores['pearsonr'] = pearsonr(y_true, y_pred)[0]
    print(scores)
    return scores

def run_benchmark(model, data_dir):
    n_splits = 10
    for src in sources:
        datadir = f"{data_dir}/ml.dfs/July2020/data.{src}"
        splitdir = f"{datadir}/splits"
        gene_expression, _, _, smiles, responses = load_source(src, data_dir)
        data, gene_expression_columns, drug_columns = deep_ttc_preprocess(gene_expression, smiles, responses, model)

        # -----------------------------------------------
        #   Train model
        # -----------------------------------------------
        # Example of training a DRP model with gene expression and Mordred descriptors
        print("\nGet the splits.")

        for split_id in range(n_splits):
            with open(f"{splitdir}/split_{split_id}_tr_id") as f:
                train_id = [int(line.rstrip()) for line in f]

            with open(f"{splitdir}/split_{split_id}_te_id") as f:
                test_id = [int(line.rstrip()) for line in f]

            # Train and test data
            train_data = data.loc[train_id]
            test_data = data.loc[test_id]

            # Val data from tr_data
            train_data, validation_data = train_test_split(train_data, test_size=0.12)
            print("Train", train_data.shape)
            print("Val  ", validation_data.shape)
            print("Test ", test_data.shape)

            # Scale
            # Train model
            train_data.index = range(train_data.shape[0])
            validation_data.index = range(validation_data.shape[0])
            test_data.index = range(test_data.shape[0])
            model.train(train_drug=train_data[drug_columns], train_rna=train_data[gene_expression_columns],
              val_drug=validation_data[drug_columns], val_rna=validation_data[gene_expression_columns])

            # Predict
            _, y_pred, _, _, _, _, _, _, _ = model.predict(test_data[drug_columns], test_data[gene_expression_columns])
            y_true = test_data['Label']       
        
            # Scores
            scores = score(y_true, y_pred)
            result = {'y_true': y_true,
                    'y_pred': y_pred}
            pickle.dump(result, open(f'results/predictions_{src}_cv_split_{split_id}.pickle', 'wb'))
            pickle.dump(scores, open(f'results/scores_{src}_cv_split_{split_id}.pickle', 'wb'))



            for test_src in sources:
                if test_src == src:
                    continue
                test_gene_expression, _, _, test_smiles, test_responses = load_source(test_src, data_dir)
                test_src_data, test_gene_expression_columns, test_drug_columns = deep_ttc_preprocess(test_gene_expression, test_smiles, test_responses, model)
                # Subsetting to the current set of expressed genes!!!
                _, y_pred, _, _, _, _, _, _, _ = model.predict(test_src_data[test_drug_columns], test_src_data[gene_expression_columns])
                y_true = test_src_data['Label']
                result = {'y_true': y_true,
                    'y_pred': y_pred}
                scores = score(y_pred, y_true)
                pickle.dump(result, open(f'results/predictions_{src}_split_{split_id}_on_{test_src}.pickle', 'wb'))
                pickle.dump(scores, open(f'results/scores_{src}_split_{split_id}_on_{test_src}.pickle', 'wb'))
