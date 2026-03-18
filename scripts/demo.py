#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
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
    results = fatwater.separate(data_param_file=data_param_file, algo_param_file=algo_param_file, model_param_file=model_param_file)
    print(f'Fat/water separation took {time.time() - t:.1f} sec')

    io.save(results, data_path / '17_REC')


if __name__ == '__main__':
    demo_with_param_files()