import scipy.io
import numpy as np


def readISMRMchallengeData(file):
    data_params = {}

    data = scipy.io.loadmat(file)['imDataParams'][0, 0]
    for i in range(0, 4):
        if len(data[i].shape) == 5:
            data_params['img'] = data[i] # Image data (row, col, slice, coil, echo)
        elif data[i].shape[1] > 2:
            data_params['t'] = data[i][0] # Dephasing times [sec]
        else:
            if data[i][0, 0] > 1:
                data_params['B0'] = float(data[i][0, 0]) # Fieldstrength [T]
            else:
                data_params['clockwisePrecession'] = bool(data[i][0, 0])
    
    if 'img' not in data_params:
        raise ValueError('Could not read all required parameters from MATLAB file')

    if data_params['img'].shape[3] > 1:
        raise NotImplementedError('Multiple coil elements not supported yet')
    
    data_params['img'] = data_params['img'][:, :, :, 0, :] # Ignore coil dimension

    # To get data as: (echo, slice, row, col)
    data_params['img'] = np.transpose(data_params['img'])
    data_params['img'] = np.swapaxes(data_params['img'], 2, 3)

    return data_params


# Save output as MATLAB arrays
def save(output, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    filename = outdir / '0.mat'
    print(f'Writing images to "{filename}"')
    scipy.io.savemat(filename, output)
