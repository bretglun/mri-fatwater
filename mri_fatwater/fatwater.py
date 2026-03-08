from dataclasses import replace
import numpy as np
from mri_fatwater import algorithm, params, DICOM, MATLAB
from .constants import EPSILON


def save(output, dPar):
    for seriesType in output: # zero pad if was cropped and reshape to row,col,slice
        output[seriesType] = np.moveaxis(output[seriesType], 0, -1)
    
    if dPar.fileType == 'DICOM':
        DICOM.save(output, dPar)
    elif dPar.fileType == 'MATLAB':
        MATLAB.save(output, dPar)
    else:
        raise Exception(f'Unknown filetype: {dPar.fileType}')


# Merge output for slices reconstructed separately
def mergeOutputSlices(outputList):
    mergedOutput = outputList[0]
    for output in outputList[1:]:
        for seriesType in output:
            mergedOutput[seriesType] = np.concatenate((mergedOutput[seriesType], output[seriesType]))
    return mergedOutput


def getFattyAcidComposition(rho):
    nFAC = len(rho) - 2 # Number of Fatty Acid Composition Parameters
    CL, UD, PUD = None, None, None

    if nFAC == 1:
        # UD = F2/F1
        UD = np.abs(rho[2] / (rho[1] + EPSILON))
    elif nFAC == 2:
        # UD = F2/F1
        # PUD = F3/F1
        UD = np.abs(rho[2] / (rho[1] + EPSILON))
        PUD = np.abs(rho[3] / (rho[1] + EPSILON))
    elif nFAC == 3:
        # UD = F2/F1
        # PUD = F3/F1
        # CL = F4/F1
        UD = np.abs(rho[2] / (rho[1] + EPSILON))
        PUD = np.abs(rho[3] / (rho[1] + EPSILON))
        CL = np.abs(rho[4] / (rho[1] + EPSILON))
    else:
        raise Exception(f'Unknown number of Fatty Acid Composition parameters: {nFAC}')

    return CL, UD, PUD


# Get total fat component (for Fatty Acid Composition; trivial otherwise)
def getFat(rho, alpha):
    fat = np.zeros(rho.shape[1:], dtype=complex)
    for m in range(1, alpha.shape[0]):
        fat += sum(alpha[m, 1:])*rho[m]
    return fat


def autocrop(dPar):
    crop = [0] * 6 # [x0, y0, z0, x1, y1, z1]
    abs_img = np.mean(np.abs(dPar.img), axis=0)
    threshold = np.percentile(abs_img, 95) * .01
    
    for dim in range(3):
        profile = abs_img.mean(axis=tuple(i for i in range(3) if i != dim))
        foreground_indices = np.where(profile > threshold)[0]
        crop[2-dim] = int(foreground_indices[0])
        crop[5-dim] = int(foreground_indices[-1] + 1)
    
    if tuple(crop[:3]) == (0, 0, 0) and tuple(crop[3:]) == abs_img.shape[::-1]:
        return dPar
    
    print(f'Auto-cropping to FOV: [x0, y0, z0, x1, y1, z1] = {crop} ({np.prod([crop[3+i] - crop[i] for i in range(3)])/np.prod(abs_img.shape)*100:.1f}% of original FOV)')
    return replace(dPar, crop=crop, pad=True)


# Zero pad cropped data to original shape if prescribed
def padCropped(data, dPar):
    if dPar.pad:
        nz, ny, nx = dPar.original_shape
        x0, y0, z0, x1, y1, z1 = dPar.crop
        return np.pad(data, ((z0, nz-z1), (y0, ny-y1), (x0, nx-x1)))
    else:
        return data


# Perform fat/water separation and return prescribed output
def reconstruct(dPar, aPar, mPar):

    if aPar.autocrop and dPar.crop is None:
        dPar = autocrop(dPar)

    # Do the fat/water separation
    rho, B0map, R2map = algorithm.reconstruct(dPar, aPar, mPar)
    wat = rho[0]
    fat = getFat(rho, mPar.alpha)

    # Prepare prescribed output
    output = {}
    if 'wat' in aPar.output:
        output['wat'] = np.abs(wat)
    if 'fat' in aPar.output:
        output['fat'] = np.abs(fat)
    if 'phi' in aPar.output:
        output['phi'] = np.angle(wat, deg=True) + 180
    if 'ip' in aPar.output: # Calculate synthetic in-phase
        output['ip'] = np.abs(wat+fat)
    if 'op' in aPar.output: # Calculate synthetic opposed-phase
        output['op'] = np.abs(wat-fat)
    if 'ff' in aPar.output: # Calculate the fat fraction
        if aPar.magnitudeDiscrimination:  # to avoid bias from noise
            output['ff'] = 100 * np.real(fat / (wat + fat + EPSILON))
        else:
            output['ff'] = 100 * np.abs(fat)/(np.abs(wat) + np.abs(fat) + EPSILON)
    if 'B0map' in aPar.output:
        output['B0map'] = B0map
    if 'R2map' in aPar.output:
        output['R2map'] = R2map

    # Do any Fatty Acid Composition in a second pass
    if hasattr(mPar, 'pass2'):
        rho = algorithm.reconstruct(dPar, aPar.pass2, mPar.pass2, B0map, R2map)[0]
        CL, UD, PUD = getFattyAcidComposition(rho)
    
        if 'CL' in aPar.output:
            output['CL'] = CL
        if 'UD' in aPar.output:
            output['UD'] = UD
        if 'PUD' in aPar.output:
            output['PUD'] = PUD
    
    for seriesType in output:
        output[seriesType] = padCropped(output[seriesType], dPar)

    return output


def separate(dataParamFile, algoParamFile, modelParamFile, outDir=None):
    # Read configuration files
    dPar = params.DataParams(dataParamFile, outDir=outDir)
    mPar = params.ModelParams(configFile=modelParamFile, temperature=dPar.temperature)
    aPar = params.AlgoParams(configFile=algoParamFile)

    if mPar.nFAC > 0:
        # For Fatty Acid Composition, create algorithm and model params for two passes
        # First pass: use standard fat-water separation to determine B0 and R2*
        # Second pass: use B0- and R2*-maps from first pass and do the Fatty Acid Composition
        aPar.pass2 = replace(aPar, nICMiter=0, graphcut=False, graphcutLevel=None)
        mPar2 = replace(mPar)
        mPar = replace(mPar, nFAC=0, relAmps=None)
        mPar.pass2 = mPar2

    print(dPar)
    print(f't = {' / '.join(f'{t*1e3:.2f}' for t in dPar.t)} msec')
    print(f'(nx, ny, nz) = {dPar.img.shape[-1:0:-1]}')
    print(mPar)
    print(aPar)

    # Run fat/water processing and save output
    if aPar.use3D or dPar.nz == 1:
        output = reconstruct(dPar, aPar, mPar)
        save(output, dPar)
    else:
        output = []
        for slice in range(dPar.nz):
            print(f'Processing slice {slice+1}/{dPar.nz}...')
            sliceDataParams = replace(dPar, sliceList=[slice])
            output.append(reconstruct(sliceDataParams, aPar, mPar))
        save(mergeOutputSlices(output), dPar)