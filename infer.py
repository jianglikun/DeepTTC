from DeepTTC_candle import *
from .Step3_model import *
import pickle
import sys
import os

if __name__ == "__main__":
    #if len(sys.argv) < 4:
    #    raise Exception(
    #        "Drug data, gene expression data, and output files are required to make predictions")
    #drug_data = sys.argv[1]
    #rna_data = sys.argv[2]
    #output = sys.argv[3]


    gParameters = initialize_parameters()
    args = candle.ArgumentStruct(**gParameters)

    try:
        path = os.path.split(args.predictions_output)
        if len(path) > 1:
            path = os.path.join(path[:-1])
            os.makedirs(path)
    except:
        pass

    model_path = os.path.join(os.getenv('CANDLE_DATA_DIR'), args.model_name)

    model = DeepTTC()
    model.load_pretrained(model_path)
    _, y_pred, _, _, _, _, _, _, _ = model.predict(args.test_data_drug, args.test_rna_data)
    results = pd.DataFrame(y_pred)
    results.to_csv(args.predictions_output)

    #pickle.dump(results, open(f'{output}.pickle', 'wb'), protocol=4)
