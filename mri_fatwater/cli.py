import argparse
import sys
from mri_fatwater import fatwater, io
from importlib.resources import files
from pathlib import Path


def get_example_param_files():
    path = Path(files('mri_fatwater')).parent / 'configs'
    param_files = {}
    for par in ('data', 'algo', 'model'):
        file = path / f'{par}Params.yml'
        param_files[par] = file if file.is_file() else None
    return param_files


def main():
    param_files = get_example_param_files()
    example_hint = ''
    if all(p is not None for p in param_files.values()):
        example_hint = f'''Example parameter files:\n{param_files['data']}\n{param_files['algo']}\n{param_files['model']}\n\nExample usage:\nfatwater -d {param_files['data']} -a {param_files['algo']} -m {param_files['model']} -o results'''
        
    # Initiate command line parser
    parser = argparse.ArgumentParser(
        prog='fatwater',
        description='MRI fat/water separation of chemical shift encoded data',
        epilog=example_hint,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataParamFile', '-d', default=None,
                        help="File path of data parameter configuration file")
    parser.add_argument('--algoParamFile', '-a', default=None,
                        help="File path of algorithm parameter configuration file")
    parser.add_argument('--modelParamFile', '-m', default=None,
                        help="File path of model parameter configuration file")
    parser.add_argument('--outDir', '-o', default='.',
                        help="Path to save the results")

    # Parse command line
    args = parser.parse_args()
    
    # If no arguments provided, show help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    results = fatwater.separate(
        data_param_file=args.dataParamFile,
        algo_param_file=args.algoParamFile,
        model_param_file=args.modelParamFile
    )
    io.save(results, args.outDir)

    return 0


if __name__ == '__main__':
    exit(main())