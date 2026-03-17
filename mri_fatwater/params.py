from dataclasses import dataclass, field, fields
import yaml
import numpy as np
from typing import Optional
from pathlib import Path


def init_dataclass(dataclass_instance, **overrides):
    params = {f.name: f.default for f in fields(dataclass_instance) if f.init}
    params.update(overrides)

    for param in params:
        if not hasattr(dataclass_instance, param):
            raise ValueError(f'Unknown parameter "{param}" passed to {type(dataclass_instance).__name__} constructor')
        setattr(dataclass_instance, param, params[param])


@dataclass
class DataParams:
    data: Optional[np.ndarray] = field(default=None, repr=False)
    t: tuple[float, ...] = field(default=None) # [sec] dephasing times (=TE for gradient echo)
    B0: float = 3.0 # [T]
    dx: float = 1.5 # [mm] TODO: consider voxelsize tuple instead
    dy: float = 1.5 # [mm]
    dz: float = 5.0 # [mm]
    
    crop: Optional[tuple[int, ...]] = None # [x0, y0, z0, x1, y1, z1] (if cropping is desired)
    pad: bool = True # Whether to zero-pad back to original shape after reconstruction (if cropping was applied)

    temperature: Optional[float] = None
    offresCenter: int = 0 # TODO: units of Hz instead of index
    reScale: float = 1.0 # TODO: handle differently
    file: str = field(default=None, repr=False)

    def __init__(self, echoes=None, slices=None, clockwise=True, **overrides):
        init_dataclass(self, **overrides)
        
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 4:
            raise ValueError('DataParams requires a 4D numpy array "data" with dimensions (N, nz, ny, nx)')
        if len(self.t) != self.data.shape[0]:
            raise ValueError(f'Number of time shifts ({len(self.t)}) does not match number of echoes in data ({self.data.shape[0]})')
        if echoes is not None:
            if any(echo not in range(len(self.t)) for echo in echoes):
                raise ValueError(f'Echo indices must be over 0 and smaller than {self.data.shape[0]} (number of echoes in data)')
            self.t = tuple(self.t[i] for i in echoes)
            self.data = self.data[echoes, ...]
        if slices is not None:
            if any(slice not in range(self.data.shape[1]) for slice in slices):
                raise ValueError(f'Slice indices must be over 0 and smaller than {self.data.shape[1]} (number of slices in data)')
            self.data = self.data[:, slices, ...]
        if self.pad:
            if self.crop is None:
                self.pad = False
            else:
                self.original_shape = self.data.shape[1:]
        if self.crop is not None:
            if len(self.crop) != 6:
                raise ValueError('Param "crop" must be a list of six integers: [x0, y0, z0, x1, y1, z1]')
            low = self.crop[2::-1]
            high = self.crop[5:2:-1]
            if any(lo<0 or lo>N or hi<0 or hi>N or hi<=lo for lo, hi, N in zip(low, high, self.data.shape[1:])):
                raise ValueError(f'Param "crop" [x0, y0, z0, x1, y1, z1] values must be within the image dimensions (nx={self.data.shape[3]}, ny={self.data.shape[2]}, nz={self.data.shape[1]})')
            self.data = self.data[:, low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        if not clockwise:
            np.conjugate(self.data, out=self.data)

        self.data *= self.reScale

        if any(s<1 for s in self.data.shape):
            raise ValueError(f'Empty data dims found: data.shape={self.data.shape}')
        
        if self.N < 2:
            raise Exception(f'At least two echoes required, only {self.N} found')
    
    @property
    def N(self):
        return len(self.t)
    
    @property
    def nz(self):
        return self.data.shape[1]
    
    @property
    def ny(self):
        return self.data.shape[2]
    
    @property
    def nx(self):
        return self.data.shape[3]
    
    @property
    def t1(self):
        return self.t[0]
    
    @property
    def dt(self):
        dt = np.diff(self.t)
        if np.max(dt)/np.min(dt) > 1.1:
            raise ValueError(f'Varying inter-echo spacing for t={self.t}')
        return np.mean(dt)


@dataclass
class ModelParams:
    # Default fat spectrum from ISMRM fat-water separation Matlab toolbox
    fatCS: tuple[float, ...] = (5.3, 4.31, 2.76, 2.1, 1.3, 0.9) # [ppm]
    relAmps: tuple[float, ...] = (0.048, 0.039, 0.004, 0.128, 0.693, 0.087,)
    watCS: float = 4.7
    
    def __new__(cls, temperature=None, **overrides):
        if cls is ModelParams and 'nFAC' in overrides:
            from mri_fatwater.FAC import FACmodelParams
            return super().__new__(FACmodelParams)
        return super().__new__(cls)
    
    def __init__(self, temperature=None, **overrides):
        init_dataclass(self, **overrides)
        
        # Temperature dependence according to Hernando et al., MRM 72(2):464–70, 2014
        if temperature is not None:
            self.watCS = 1.3 + 3.748 -.01085 * temperature # Temp in [°C]

        self.CS = np.array([self.watCS] + self.fatCS, dtype=np.float32)

        self.set_alpha()
    
    def set_alpha(self):
        M = 2 # Number of linear components
        P = len(self.CS) # Number of resonance peaks
        self.alpha = np.zeros([M, P], dtype=np.float32)
        self.alpha[0, 0] = 1.  # Water component

        if self.relAmps is not None:
            for (p, amp) in enumerate(self.relAmps):
                self.alpha[1, p+1] = float(amp)
        elif P==2:
            self.alpha[1, 1] = 1. # Single fat peak
        else:
            raise ValueError(f'Relative amplitudes not provided for the {P-1} fat peaks.')
    
    @property
    def M(self):
        return self.alpha.shape[0]
    
    @property
    def P(self):
        return self.alpha.shape[1]


@dataclass
class AlgoParams:
    nR2: int = 145
    R2max: float = 144. # [sec-1]
    R2cand: tuple[float, ...] = (40.,) # [sec-1]
    mu: float = 0.1
    nB0: int = 100
    nICMiter: int = 10
    graphcut: bool = True
    graphcutLevel: int = 0
    multiScale: bool = True
    use3D: bool = True
    magnitudeDiscrimination: bool = False
    offresPenalty: float = 0.
    realEstimates: Optional[bool] = None
    autocrop: bool = True
    output: Optional[tuple[str, ...]] = None

    def __init__(self, N=None, **overrides):
        init_dataclass(self, **overrides)
    
        if self.realEstimates is None:
            self.realEstimates = (N == 2)
        
        self.R2step = self.R2max/(self.nR2-1) if self.nR2 > 1 else 1. # [sec-1]
        self.iR2cand = np.array(list(set([min(self.nR2-1, int(R2/self.R2step)) for R2 in self.R2cand])))
        
        self.maxICMupdate = round(self.nB0/10)

        if self.output is None:
            self.output = ['wat', 'fat', 'ff', 'B0map']
            if self.realEstimates:
                self.output.append('phi')
            if (self.nR2 > 1):
                self.output.append('R2map')


def load_data(data_file, filepath):
    if not data_file:
        raise ValueError('No data file specified in parameters')
    if Path(data_file).suffix != '.npy':
        raise ValueError(f'Data file must be in .npy format, not "{data_file}"')
    if not Path(filepath / data_file).is_file():
        raise FileNotFoundError(f'Could not find data file "{data_file}" in path "{filepath}"')
    data_file = Path(filepath / data_file)
    return np.load(data_file).transpose()


def prepare_data_params(data, data_params, data_param_file):
    params = get_params(data_param_file, data_params)
    data_file = params.pop('file') if 'file' in params else None
    filepath = params.pop('filepath') if 'filepath' in params else (data_param_file.parent if data_param_file else '.')
    params['data'] = data
    if params['data'] is None:
        try:
            params['data'] = load_data(data_file, filepath)
        except Exception as e:
            raise Exception('No data provided and could not load data from file') from e    
    return params


def read_config_file(config_file):
    if config_file:
        with open(config_file, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise Exception(f'Error reading config file {f}') from exc
    else:
        return {}


def get_params(config_file, overrides):
    params = read_config_file(config_file)
    params.update(overrides)
    return params


def prepare(data, data_params, algo_params, model_params, data_param_file, algo_param_file, model_param_file):
    data_params = prepare_data_params(data, data_params, data_param_file)
    model_params = get_params(model_param_file, model_params)
    algo_params = get_params(algo_param_file, algo_params)
    return data_params, algo_params, model_params