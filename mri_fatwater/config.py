from pathlib import Path
import yaml
from mri_fatwater import DICOM, MATLAB


# extract data parameter object representing a single slice
def getSliceDataParams(dPar, slice, z):
    sliceDataParams = dict(dPar)
    sliceDataParams['sliceList'] = [slice]
    sliceDataParams['img'] = dPar['img'][:, [z], :, :]
    sliceDataParams['nz'] = 1
    return sliceDataParams


# extract dPar object representing a slab of contiguous slices starting at z
def getSlabDataParams(dPar, slices, z):
    slabDataParams = dict(dPar)
    slabDataParams['sliceList'] = slices
    slabSize = len(slices)
    slabDataParams['img'] = dPar['img'][:, z:z+slabSize, :, :]
    slabDataParams['nz'] = slabSize
    return slabDataParams


# group slices in sliceList in slabs of reconSlab contiguous slices
def getSlabs(sliceList, reconSlab):
    slabs = []
    slices = []
    pos = 0
    for z, slice in enumerate(sliceList):
        # start a new slab
        if slices and (len(slices) == reconSlab or not slice == slices[-1]+1):
            slabs.append((slices, pos))
            slices = [slice]
            pos = z
        else:
            slices.append(slice)
    slabs.append((slices, pos))
    return slabs

    
# Update data param object, set default parameters and read data from files
def setupDataParams(dPar, outDir=None):
    if outDir:
        dPar['outDir'] = Path(outDir)
    elif 'outDir' in dPar:
        dPar['outDir'] = Path(dPar['outDir'])
    else:
        raise Exception('No outDir defined')

    defaults = [
        ('reScale', 1.0),
        ('temperature', None),
        ('clockwisePrecession', False),
        ('offresCenter', 0.),
        ('files', [])
    ]

    for param, defval in defaults:
        if param not in dPar:
            dPar[param] = defval

    if 'files' in dPar:
        dPar['files'] = [dPar['configPath'] / file for file in list(dPar['files']) if Path(dPar['configPath'] / file).is_file()]
    
    if 'dirs' in dPar:
        dPar['dirs'] = [dPar['configPath'] / dir for dir in list(dPar['dirs']) if Path(dPar['configPath'] / dir).is_dir()]
        for path in dPar['dirs']:
            dPar['files'] += [obj for obj in path.iterdir() if obj.is_file()]
    
    validFiles = DICOM.getValidFiles(dPar['files'])
    
    if validFiles:
        DICOM.updateDataParams(dPar, validFiles)
    else:
        if len(dPar['files']) == 1 and dPar['files'][0].suffix == '.mat':
            MATLAB.updateDataParams(dPar, dPar['files'][0])
        else:
            raise Exception('No valid files found')
    
    if 'reconSlab' in dPar:
        dPar['slabs'] = getSlabs(dPar['sliceList'], dPar['reconSlab'])


# Read configuration file
def readConfig(file, section):
    file = Path(file)
    with open(file, 'r') as configFile:
        try:
            config = yaml.safe_load(configFile)
        except yaml.YAMLError as exc:
            raise Exception(f'Error reading config file {file}') from exc
    config['configPath'] = file.parent
    return config