from dataclasses import replace
import numpy as np
from mri_fatwater import algorithm, params
from .constants import EPSILON


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


def autocrop(dPar):
    crop = [0] * 6 # [x0, y0, z0, x1, y1, z1]
    abs_img = np.mean(np.abs(dPar.data), axis=0)
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
def pad_cropped(data, dPar):
    if dPar.pad:
        nz, ny, nx = dPar.original_shape
        x0, y0, z0, x1, y1, z1 = dPar.crop
        return np.pad(data, ((z0, nz-z1), (y0, ny-y1), (x0, nx-x1)))
    else:
        return data


# Get total fat component (for Fatty Acid Composition; trivial otherwise)
def getFat(rho, alpha):
    fat = np.zeros(rho.shape[1:], dtype=complex)
    for m in range(1, alpha.shape[0]):
        fat += sum(alpha[m, 1:])*rho[m]
    return fat


def get_prescribed_output(rho, B0map, R2map, alpha, output, magnitude_discrimination=False):
    wat = rho[0]
    fat = getFat(rho, alpha)

    results = {}
    if 'wat' in output:
        results['wat'] = np.abs(wat)
    if 'fat' in output:
        results['fat'] = np.abs(fat)
    if 'phi' in output:
        results['phi'] = np.angle(wat, deg=True) + 180
    if 'ip' in output: # synthetic in-phase
        results['ip'] = np.abs(wat+fat)
    if 'op' in output: # synthetic opposed-phase
        results['op'] = np.abs(wat-fat)
    if 'ff' in output: # fat signal fraction
        if magnitude_discrimination:  # to avoid bias from noise
            results['ff'] = 100 * np.real(fat / (wat + fat + EPSILON))
        else:
            results['ff'] = 100 * np.abs(fat)/(np.abs(wat) + np.abs(fat) + EPSILON)
    if 'B0map' in output:
        results['B0map'] = B0map
    if 'R2map' in output:
        results['R2map'] = R2map
    
    if any(p in output for p in ['CL', 'UD', 'PUD']):
        CL, UD, PUD = getFattyAcidComposition(rho)
        if 'CL' in output:
            results['CL'] = CL
        if 'UD' in output:
            results['UD'] = UD
        if 'PUD' in output:
            results['PUD'] = PUD

    return results


def run_separation_passes(passes):
    B0map, R2map, results = None, None, {}
    for dPar, aPar, mPar in passes:
        rho, B0map, R2map = algorithm.core_fatwater_separation(dPar, aPar, mPar, B0map, R2map)
        results.update(get_prescribed_output(rho, B0map, R2map, mPar.alpha, aPar.output, aPar.magnitudeDiscrimination))
    return results


def run_FAC_passes(dPar, aPar, mPar):
    output = ['UD']
    if (mPar.nFAC > 1):
        output.append('PUD')
    if (mPar.nFAC > 2):
        output.append('CL')
    mPar1 = replace(mPar, nFAC=0, relAmps=None)
    aPar2 = replace(aPar, nICMiter=0, graphcut=False, graphcutLevel=None, output=output)
    passes = [
        (dPar, aPar, mPar1), # First pass: use standard fat-water separation to determine B0 and R2*
        (dPar, aPar2, mPar)  # Second pass: use B0- and R2*-maps from first pass and do the Fatty Acid Composition
    ]
    return run_separation_passes(passes)


def separate_volume(dPar, aPar, mPar):
    if aPar.autocrop and dPar.crop is None:
        dPar = autocrop(dPar)
    
    if mPar.nFAC == 0:
        results = run_separation_passes([(dPar, aPar, mPar)])
    else:
        results = run_FAC_passes(dPar, aPar, mPar)

    for result_type in results:
        results[result_type] = pad_cropped(results[result_type], dPar)
    
    return results


def merge_slice_results(slice_results):
    results = slice_results[0]
    for slice_result in slice_results[1:]:
        for result_type in slice_result:
            results[result_type] = np.concatenate((results[result_type], slice_result[result_type]))
    return results


def separate_slices(dPar, aPar, mPar):
    slice_results = []
    for slice in range(dPar.nz):
        print(f'Processing slice {slice + 1}/{dPar.nz}...')
        slice_dPar = replace(dPar, slices=[slice])
        slice_results.append(separate_volume(slice_dPar, aPar, mPar))
    return merge_slice_results(slice_results)


def separate_with_param_objects(dPar, aPar, mPar):
    if aPar.use3D or dPar.nz == 1:
        return separate_volume(dPar, aPar, mPar)
    return separate_slices(dPar, aPar, mPar)


def separate(data=None, 
             data_params={}, 
             algo_params={}, 
             model_params={}, 
             data_param_file=None, 
             algo_param_file=None, 
             model_param_file=None):
    
    data_params, algo_params, model_params = params.prepare(data, data_params, algo_params, model_params, data_param_file, algo_param_file, model_param_file)
    
    dPar = params.DataParams(**data_params)
    mPar = params.ModelParams(**model_params, temperature=dPar.temperature)
    aPar = params.AlgoParams(**algo_params)

    print(dPar)
    print(f't = {' / '.join(f'{t*1e3:.2f}' for t in dPar.t)} msec')
    print(f'(nx, ny, nz) = {dPar.data.shape[-1:0:-1]}')
    print(mPar)
    print(aPar)

    return separate_with_param_objects(dPar, aPar, mPar)