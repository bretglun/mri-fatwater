import optparse
from mri_fatwater import fatwater


def CLI():
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--dataParamFile', '-d', default='',  type="string",
                 help="File path of data parameter configuration file")
    p.add_option('--algoParamFile', '-a', default='',  type="string",
                 help="File path of algorithm parameter configuration file")
    p.add_option('--modelParamFile', '-m', default='',  type="string",
                 help="File path of model parameter configuration file")

    # Parse command line
    options, arguments = p.parse_args()

    fatwater.separate(options.dataParamFile, options.algoParamFile, options.modelParamFile)


if __name__ == '__main__':
    CLI()