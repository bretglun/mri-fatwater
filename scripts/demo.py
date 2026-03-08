#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import scipy.io
import time
from pathlib import Path
from mri_fatwater import fatwater, MATLAB


def getScore(case, dir, refFile):
    # Read reconstructed MATLAB-file
    file = dir / '0.mat'
    mat = scipy.io.loadmat(file)
    recFF = mat['ff'].flatten(order='F')/100
    recFF.shape = recFF.shape + (1,)
    # Read reference MATLAB-file
    mat = scipy.io.loadmat(refFile)
    refFF = mat['REFCASES'][0, case - 1]
    mask = mat['MASKS'][0, case - 1]
    # Calculate score
    score = 100 * \
        (1 - np.sum((np.abs(refFF - recFF) > 0.1) * mask) / np.sum(mask))
    return score


def save(output, outdir):
    for seriesType in output: # zero pad if was cropped and reshape to row,col,slice
        output[seriesType] = np.moveaxis(output[seriesType], 0, -1)
    MATLAB.save(output, outdir)


if __name__ == '__main__':

    rootPath = Path(__file__).resolve().parent.parent.absolute()
    challengePath = rootPath / 'challenge'

    refFile = challengePath / 'refdata.mat'
    if not refFile.is_file():
        url = 'https://challenge.ismrm.org/u/5131732/ISMRM_Challenge/refdata.mat'
        raise Exception(f'ISMRM 2012 challenge reference data file was not found at: {refFile.absolute()}. '
                        f'Please download from: {url}.')

    modelParamsFile = rootPath / 'configs/modelParams.yml'
    cases = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    for case in cases:
        caseFile = challengePath / f'{str(case).zfill(2)}.mat'
        if not caseFile.is_file():
            url = 'https://challenge.ismrm.org/u/5131732/ISMRM_Challenge/data_matlab.zip'
            raise Exception(f'ISMRM 2012 challenge dataset {case} was not found at: {caseFile.absolute()}. '
                            f'Please download from: {url}.')

    results = []
    for case in cases:
        dataParamsFile = challengePath / f'{case}.yml'
        if case == 9:
            algoParamsFile = rootPath / 'configs/algoParams2D.yml'
        else:
            algoParamsFile = rootPath / 'configs/algoParams3D.yml'
        outdir = challengePath / f'{str(case).zfill(2)}_REC'
        t = time.time()
        output = fatwater.separate(dataParamsFile, algoParamsFile, modelParamsFile)
        save(output, outdir)
        results.append((case, getScore(case, outdir, refFile), time.time() - t))

    print()
    for case, score, recTime in results:
        print(f'Case {case}: score {score:.2f}% in {recTime:.1f} sec')
