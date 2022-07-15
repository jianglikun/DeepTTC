import os
import torch
import candle
from Step3_model import *
from Step2_DataEncoding import DataEncoding

file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = [
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
        "help": "intermediate size of the drug layers",
    },
    {
        "name": "transformer_emb_size_drug",
        "type": int,
        "help": "size of the drug embeddings",
    },
    {
        "name": "transformer_n_layer_drug",
        "type": int,
        "help": "number of layers for drug transformer",
    },
    {
        "name": "transformer_intermediate_size_drug",
        "type": int,
        "help": "number of layers for drug transformer",
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
            not os.path.exists(args.test_data_rna):
        obj = DataEncoding(vocab_dir=args.vocab_dir)
        train_drug, test_drug = obj.Getdata.ByCancer(random_seed=args.rng_seed)
        train_drug, train_rna, test_drug, test_rna = obj.encode(
            traindata=train_drug,
            testdata=test_drug)
        train_drug.to_csv(args.train_data_drug)
        test_drug.to_csv(args.test_data_drug)
        train_rna.to_csv(args.train_data_rna)
        test_rna.to_csv(args.test_data_rna)
    else:
        train_drug = pd.read_csv(args.train_data_drug)
        test_drug = pd.read_csv(args.test_data_drug)
        train_rna = pd.read_csv(args.train_data_rna)
        test_rna = pd.read_csv(args.test_data_rna)
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


def run(gParameters):
    args = candle.ArgumentStruct(**gParameters)
    print(args)

    train_drug, test_drug, train_rna, test_rna = process_data(args)

    # step2：构造模型
    modeldir = args.output_dir
    modelfile = os.path.join(modeldir, args.model_name)
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    net = DeepTTC(modeldir=modeldir, args=args)
    net.train(train_drug=train_drug, train_rna=train_rna,
              val_drug=test_drug, val_rna=test_rna)
    net.save_model()
    print("Model Saved :{}".format(modelfile))


def main():
    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == "__main__":
    main()
    try:
        torch.cuda.empty_cache()
    except AttributeError:
        pass
