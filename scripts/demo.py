#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import time
from mri_fatwater import fatwater, io


def demo_with_param_files():
    root_path = Path(__file__).resolve().parent.parent.absolute()
    data_path = root_path / 'data'
    config_path = root_path / 'configs'

    data_param_file = data_path / '17.yml'
    algo_param_file = config_path / 'algoParams.yml'
    model_param_file = config_path / 'modelParams.yml'

    t = time.time()
    results = fatwater.separate(
        data_param_file=data_param_file, 
        algo_param_file=algo_param_file, 
        model_param_file=model_param_file
        )
    print(f'Fat/water separation took {time.time() - t:.1f} sec')

    io.save(results, data_path / '17_REC')


def demo_with_param_dicts():
    root_path = Path(__file__).resolve().parent.parent.absolute()
    data_path = root_path / 'data'

    data = io.load_numpy_data('17.npy', data_path)
    
    t = time.time()
    results = fatwater.separate(
        data=data,
        data_params={'B0': 1.5, 't': [0.00287, 0.00607, 0.00927]}, 
        algo_params={
            'graphcut': True, 
            'output': ['wat', 'fat', 'ff', 'B0map', 'R2map']}, 
        model_params={
            'watCS': 4.7, 
            'fatCS': [5.3, 4.31, 2.76, 2.1, 1.3, 0.9], 
            'relAmps': [0.048, 0.039, 0.004, 0.128, 0.693, 0.087]}
        )
    print(f'Fat/water separation took {time.time() - t:.1f} sec')

    io.save(results, data_path / '17_REC')


if __name__ == '__main__':
    #demo_with_param_files()
    demo_with_param_dicts()