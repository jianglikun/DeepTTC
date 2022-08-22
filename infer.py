from DeepTTC_candle import *
from Step3 import *
import pickle
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise Exception("Drug data, gene expression data, and output files are required to make predictions")
    drug_data = sys.argv[1]
    rna_data = sys.argv[2]
    output = sys.argv[3]

    gParameters = initialize_parameters()
    args = candle.ArgumentStruct(**gParameters)
    model_path = os.path.join(output_dir, model_name)

    model = DeepTTC()
    model.load_pretrained(model_path)
    results = model.predict(drug_data, rna_data)
    pickle.dump(results, open(f'{output}.pickle', 'wb'), protocol=4)
