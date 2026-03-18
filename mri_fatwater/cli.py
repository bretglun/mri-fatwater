import optparse
from mri_fatwater import fatwater, io


def main():
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--dataParamFile', '-d', default='',  type="string",
                 help="File path of data parameter configuration file")
    p.add_option('--algoParamFile', '-a', default='',  type="string",
                 help="File path of algorithm parameter configuration file")
    p.add_option('--modelParamFile', '-m', default='',  type="string",
                 help="File path of model parameter configuration file")
    p.add_option('--outDir', '-o', default='.',  type="string",
                 help="Path to save the results")

    # Parse command line
    options, arguments = p.parse_args()

    results = fatwater.separate(data_param_file=options.dataParamFile, algo_param_file=options.algoParamFile, model_param_file=options.modelParamFile)
    io.save(results, options.outDir)

    return 0


if __name__ == '__main__':
    exit(main())