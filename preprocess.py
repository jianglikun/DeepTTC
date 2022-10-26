from DeepTTC_candle import *
import candle


if __name__ == "__main__":
    gParameters = initialize_parameters()
    args = candle.ArgumentStruct(**gParameters)
    loader = DataLoader(args)

    train_drug, test_drug, train_rna, test_rna = loader.load_data()
    loader.save_data(train_drug, test_drug, train_rna, test_rna)
