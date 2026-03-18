from pathlib import Path
import numpy as np
import yaml


def read_config_file(config_file):
    if not config_file:
        return {}
    config_file = Path(config_file)
    
    if not config_file.exists():
        raise FileNotFoundError(f'Config file not found: {config_file}')

    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f'Invalid YAML in config file {f}: {e}') from e


def load_numpy_data(data_file, filepath):
    if not data_file:
        raise ValueError('No data file specified in parameters')
    
    data_file = Path(filepath / data_file)
    
    if not data_file.is_file():
        raise FileNotFoundError(f'Could not find data file "{data_file}" in path "{filepath}"')
    
    if data_file.suffix != '.npy':
        raise ValueError(f'Data file must be in .npy format, not "{data_file}"')
    
    return np.load(data_file)


def save(output, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for seriesType in output:
        filename = outdir / f'{seriesType}.npy'
        print(f'Writing images to "{filename}"')
        np.save(filename, output[seriesType])