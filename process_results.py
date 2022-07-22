import os
import pickle
import numpy as np
import pandas as pd

results_dir = 'results'

def process_scores(results_dir):
    results = {}
    for fname in os.listdir(results_dir):
        trained_on, tested_on = None, None
        if 'scores' in fname:
            file_path = os.path.join(results_dir, fname)
            scores = pickle.load(open(file_path, 'rb'))
            splitted_name = fname.split('.')[0].split('_')
            trained_on = splitted_name[1] 
            if 'cv' in fname:
                tested_on = trained_on
            else:
                tested_on = splitted_name[-1]
            if trained_on not in results:
                results[trained_on] = {}
            if tested_on not in results[trained_on]:
                results[trained_on][tested_on] = {}
            for key in scores:
                if key not in results[trained_on][tested_on]:
                    results[trained_on][tested_on][key] = []
                results[trained_on][tested_on][key].append(scores[key])


    processed_results = {}
    for trained_on in results:
        for tested_on in results[trained_on]:
            for key in results[trained_on][tested_on]:
                if key not in processed_results:
                    processed_results[key] = {}
                    processed_results[key]['mean'] = {}
                    processed_results[key]['std'] = {}
                if trained_on not in processed_results[key]['mean']:
                    processed_results[key]['mean'][trained_on] = {}
                    processed_results[key]['std'][trained_on] = {}
                processed_results[key]['mean'][trained_on][tested_on] = np.mean(results[trained_on][tested_on][key])
                processed_results[key]['std'][trained_on][tested_on] = np.std(results[trained_on][tested_on][key])

    indent = '#' * 10
    for key in processed_results:
        print(f'{indent}{indent} {key} {indent}{indent}')
        for statistic in ['mean', 'std']:
            table = pd.DataFrame.from_dict(processed_results[key][statistic])
            for col in table.columns:
                table[col] = table[col].map('{:,.2f}'.format)
            print(f'{indent} {statistic.upper()} {indent}')
            print(table)


if __name__ == '__main__':
    process_scores(results_dir)
