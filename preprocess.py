from DeepTTC_candle import *
import candle

if __name__ == "__main__":
    gParameters = initialize_parameters()
    args = candle.ArgumentStruct(**gParameters)

    download_gdsc(args.default_data_url)
    train_drug, test_drug, train_rna, test_rna = process_data(args)
