
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

# mri-fatwater

**MRI chemical shift-based fat-water separation with B0-correction using multi-scale graph cuts.**


## Overview

`mri-fatwater` is a Python package for MRI fat-water separation based on the multi-scale graph-cut algorithm described in [Berglund & Skorpil, 2017](https://doi.org/10.1002/mrm.26479). It can be used in Python scripts and as a command line tool.

> ⚠️ **Repository rename notice:** This repository was previously named `fwqpbo`. Existing links remain valid.

## Installation

```
pip install mri-fatwater
```
Alternatively, clone the repository and install dependencies, for instance using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/bretglun/mri-fatwater.git
cd mri-fatwater
uv sync
```

## Usage

### Command-Line Interface
```bash
# Show help
fatwater -h

# Run with parameter files
fatwater -d mri_fatwater/configs/dataParams.yml -a mri_fatwater/configs/algoParams.yml -m mri_fatwater/configs/modelParams.yml -o output_dir/
```

### Python API
Example using config files:

```python
from mri_fatwater import fatwater, io

results = fatwater.separate(
    data_param_file='mri_fatwater/configs/dataParams.yml',
    algo_param_file='mri_fatwater/configs/algoParams.yml',
    model_param_file='mri_fatwater/configs/modelParams.yml'
)
io.save(results, 'output_dir/')
```

Example using config dicts:

```python
from mri_fatwater import fatwater, io

data_path = 'mri_fatwater/data'
data = io.load_numpy_data('17.npy', data_path)

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
io.save(results, data_path + '/17_REC')
```

See [scripts/demo.py](scripts/demo.py) for a complete example.

## Configuration
Input parameters can be provided as:
* Human-readable YAML configuration files
* Python dictionaries

Example configuration files are available in the [mri_fatwater/configs/](mri_fatwater/configs) directory.
Input data should be a complex NumPy array or a `.npy` file specified in the data configuration file with chemical shift encoded "echoes" along the first dimension.

## Dependencies
See [pyproject.toml](pyproject.toml) for the complete list of dependencies.

## Citation
If you use this software in your research, please cite:

> Berglund J and Skorpil M. Multi-scale graph-cut algorithm for efficient water-fat separation. *Magn Reson Med*, 78(3):941-949, 2017. [[doi: 10.1002/mrm.26479](https://doi.org/10.1002/mrm.26479)]

## License
This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0).

See [LICENSE.md](LICENSE.md) for the full license text.

## Contact
Johan Berglund, Ph.D.  
Uppsala University Hospital, Uppsala, Sweden  
📧 johan.berglund@akademiska.se

*Copyright © 2016–2026 Johan Berglund*