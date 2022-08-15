import os
import wget
import torch
import candle
import subprocess
from Step3_model import *
from Step2_DataEncoding import DataEncoding
from cross_study_validation import run_cross_study_analysis

file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = [
    {
        "name": "use_lincs",
        "type": bool,
        "help": "Whether to use a LINCS subset of genes ONLY",
    },
    {
        "name": "benchmark_dir",
        "type": str,
        "help": "Directory with the input data for benchmarking",
    },
    {
        "name": "benchmark_result_dir",
        "type": str,
        "help": "Directory for benchmark output",
    },
    {
        "name": "generate_input_data",
        "type": bool,
        "help": "'True' for generating input data anew, 'False' for using stored data",
    },
    {
        "name": "mode",
        "type": str,
        "help": "Execution mode. Available modes are: 'run', 'benchmark'",
    },
    {
        "name": "cancer_id",
        "type": str,
        "help": "Column name for cancer",
    },
    {
        "name": "drug_id",
        "type": str,
        "help": "Column name for drug",
    },
    {
        "name": "sample_id",
        "type": str,
        "help": "Column name for samples/cell lines",
    },
    {
        "name": "target_id",
        "type": str,
        "help": "Column name for target",
    },
    {
        "name": "train_data_drug",
        "type": str,
        "help": "Drug data for training",
    },
    {
        "name": "test_data_drug",
        "type": str,
        "help": "Drug data for testing",
    },
    {
        "name": "train_data_rna",
        "type": str,
        "help": "RNA data for training",
    },
    {
        "name": "test_data_rna",
        "type": str,
        "help": "RNA data for testing",
    },
    {
        "name": "vocab_dir",
        "type": str,
        "help": "Directory with ESPF vocabulary",
    },
    {
        "name": "transformer_num_attention_heads_drug",
        "type": int,
        "help": "number of attention heads for drug transformer",
    },
    {
        "name": "input_dim_drug",
        "type": int,
        "help": "Input size of the drug transformer",
    },
    {
        "name": "transformer_emb_size_drug",
        "type": int,
        "help": "Size of the drug embeddings",
    },
    {
        "name": "transformer_n_layer_drug",
        "type": int,
        "help": "Number of layers for drug transformer",
    },
    {
        "name": "transformer_intermediate_size_drug",
        "type": int,
        "help": "Intermediate size of the drug layers",
    },
    {
        "name": "transformer_attention_probs_dropout",
        "type": int,
        "help": "number of layers for drug transformer",
    },
    {
        "name": "transformer_hidden_dropout_rate",
        "type": int,
        "help": "dropout rate for transformer hidden layers",
    },
    {
        "name": "input_dim_drug_classifier",
        "type": int,
        "help": "dropout rate for classifier hidden layers",
    },
    {
        "name": "input_dim_gene_classifier",
        "type": int,
        "help": "dropout rate for classifier hidden layers",
    },
]

required = None


class DeepTTCCandle(candle.Benchmark):
    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def process_data(args):
    train_drug = test_drug = train_rna = test_rna = None
    if not os.path.exists(args.train_data_rna) or \
            not os.path.exists(args.test_data_rna) or \
            args.generate_input_data:
        obj = DataEncoding(args.vocab_dir, args.cancer_id,
                           args.sample_id, args.target_id, args.drug_id)
        train_drug, test_drug = obj.Getdata.ByCancer(random_seed=args.rng_seed)

        train_drug, train_rna, test_drug, test_rna = obj.encode(
            traindata=train_drug,
            testdata=test_drug)
        print('Train Drug:')
        print(train_drug)
        print('Train RNA:')
        print(train_rna)

        pickle.dump(train_drug, open(args.train_data_drug, 'wb'), protocol=4)
        pickle.dump(test_drug, open(args.test_data_drug, 'wb'), protocol=4)
        pickle.dump(train_rna, open(args.train_data_rna, 'wb'), protocol=4)
        pickle.dump(test_rna, open(args.test_data_rna, 'wb'), protocol=4)
    else:
        train_drug = pickle.load(open(args.train_data_drug, 'rb'))
        test_drug = pickle.load(open(args.test_data_drug, 'rb'))
        train_rna = pickle.load(open(args.train_data_rna, 'rb'))
        test_rna = pickle.load(open(args.test_data_rna, 'rb'))
    return train_drug, test_drug, train_rna, test_rna


def initialize_parameters(default_model='DeepTTC.default'):
    # Build benchmark object
    common = DeepTTCCandle(file_path,
                           default_model,
                           'torch',
                           prog='deep_ttc',
                           desc='DeepTTC drug response prediction model')

    # Initialize parameters
    gParameters = candle.finalize_parameters(common)

    return gParameters


def get_model(args):
    net = DeepTTC(modeldir=args.output_dir, args=args)
    return net

def download_gdsc(url):
    OUT_DIR = 'GDSC_data'
    url_length = len(url.split('/'))-4
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    subprocess.run(['wget', '--recursive', '-nH', f'--cut-dirs={url_length}', '--no-parent', f'--directory-prefix={OUT_DIR}', f'{url}'])
    #wget.download(url, out=OUT_DIR)

def run(args):
    download_gdsc(args.default_data_url)
    train_drug, test_drug, train_rna, test_rna = process_data(args)
    modeldir = args.output_dir
    modelfile = os.path.join(modeldir, args.model_name)
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    model = get_model(args)
    model.train(train_drug=train_drug, train_rna=train_rna,
                val_drug=test_drug, val_rna=test_rna)
    model.save_model()
    print("Model Saved :{}".format(modelfile))


def benchmark(args):
    model = get_model(args)
    run_cross_study_analysis(model, args.benchmark_dir,
                             args.benchmark_result_dir, use_lincs=args.use_lincs)


def main():
    gParameters = initialize_parameters()
    args = candle.ArgumentStruct(**gParameters)
    print(args)

    if args.mode == 'run':
        run(args)
    elif args.mode == 'benchmark':
        benchmark(args)
    else:
        raise Exception('Unknown mode. Please check configuration file')


if __name__ == "__main__":
    main()
    try:
        torch.cuda.empty_cache()
    except AttributeError:
        pass
