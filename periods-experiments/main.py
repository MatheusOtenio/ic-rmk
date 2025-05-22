import os
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
from ml_experiments import execute

def execute_helper(dataset, algorithm, corr_threshold, const_threshold, seed, line_number):
    try:
        result = execute(dataset, algorithm, corr_threshold, const_threshold, seed, line_number)
        if result is None:
            return None, None
        return result, None
    except Exception as e:
        error = {
            'error': str(e),
            'dataset': dataset,
            'algorithm': algorithm,
            'corr_threshold': corr_threshold,
            'const_threshold': const_threshold,
            'seed': seed,
            'line_number': line_number
        }
        return None, error

def main():
    if __name__ == '__main__':
        directory = 'datasets'
        datasets = os.listdir(directory)

        corr_threshold = [0.8, 0.85, 0.9, 0.95]
        const_threshold = [0.05, 0.1, 0.15, 0.2]
        seed = [145, 278, 392, 49, 203, 411, 89, 356, 27, 489]
        algorithm = ['dt', 'rf', 'neigh', 'nb']

        grid = pd.DataFrame(list(product(datasets, algorithm, corr_threshold, const_threshold, seed)),
                            columns=['dataset', 'algorithm', 'corr_threshold', 'const_threshold', 'seed'])
        grid['line_number'] = grid.reset_index().index + 1

        if not os.path.exists('grid'):
            os.makedirs('grid') 
        grid.to_csv('grid/executions.csv', index=False)

        num_process = cpu_count()
        results = []
        errors = []

        with Pool(processes=num_process) as pool:
            results_list = pool.starmap(execute_helper, zip(grid['dataset'], grid['algorithm'], grid['corr_threshold'], grid['const_threshold'], grid['seed'], grid['line_number']))

        for result, error in results_list:
            if error:
                errors.append(error)
            if isinstance(result, pd.DataFrame) and not result.empty:
                results.append(result)

        if results:
            df_all_results = pd.concat(results, ignore_index=True)
            if not os.path.exists('all_results'):
                os.makedirs('all_results') 
            df_all_results.to_csv('all_results/ALL_results.csv', index=False)
            print("ALL_results.csv has been saved")

        if errors:
            df_errors = pd.DataFrame(errors)
            if not os.path.exists('errors'):
                os.makedirs('errors') 
            df_errors.to_csv('errors/errors.csv', index=False)
            print("errors.csv has been saved")

main()
