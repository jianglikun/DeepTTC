import pandas as pd
import os

DIR= 'results.DeepTTC'

for fname in os.listdir(DIR):
    if '.csv' not in fname:
        continue
    print(fname)
    path = os.path.join(DIR, fname)
    data = pd.read_csv(path)
    if 'Unnamed' in data.columns[0]:
        data = data.drop(data.columns[0], axis=1)
    data.columns = [data.columns[0], data.columns[1], 'True', 'Pred']
    data.to_csv(path, index=None)
