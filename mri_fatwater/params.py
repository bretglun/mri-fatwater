from dataclasses import dataclass, field, fields
from mri_fatwater import io
import numpy as np
from typing import Optional, ClassVar
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
    voxelsize: tuple[float, ...] = (1.5, 1.5, 5.0) # [mm]
    cyclic: tuple[bool, ...] = (False, False, False) # which dimensions are cyclic?
    
    crop: Optional[tuple[int, ...]] = None # [x0, y0, z0, x1, y1, z1] (if cropping is desired)
    pad: bool = True # Whether to zero-pad back to original shape after reconstruction (if cropping was applied)

    temperature: Optional[float] = None
    offresCenter: float = 0. # [ppm]

    def __init__(self, echo_dim=0, echoes=None, slices=None, clockwise=True, **overrides):
        init_dataclass(self, **overrides)
        
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 4:
            raise ValueError('DataParams requires a 4D numpy array "data" with dimensions (N, nz, ny, nx)')
        self.data = np.moveaxis(self.data, echo_dim, 0) # Put echo dim first
        if len(self.t) != self.data.shape[0]:
            raise ValueError(f'Number of time shifts ({len(self.t)}) does not match number of echoes in data ({self.data.shape[0]})')
        if echoes is not None:
            if any(echo not in range(len(self.t)) for echo in echoes):
                raise ValueError(f'Echo indices must be over 0 and smaller than {self.data.shape[0]} (number of echoes in data)')
            self.t = tuple(self.t[i] for i in echoes)
            self.data = self.data[echoes, ...]
        if slices is not None:
            if any(slice not in range(self.data.shape[3]) for slice in slices):
                raise ValueError(f'Slice indices must be over 0 and smaller than {self.data.shape[3]} (number of slices in data)')
            self.data = self.data[..., slices]
        if self.pad:
            if self.crop is None:
                self.pad = False
            else:
                self.original_shape = self.data.shape[1:]
        if self.crop is not None:
            if len(self.crop) != 6:
                raise ValueError('Param "crop" must be a list of six integers: [x0, y0, z0, x1, y1, z1]')
            low = self.crop[:3]
            high = self.crop[3:6]
            if any(lo<0 or lo>N or hi<0 or hi>N or hi<=lo for lo, hi, N in zip(low, high, self.data.shape[1:])):
                raise ValueError(f'Param "crop" [x0, y0, z0, x1, y1, z1] values must be within the image dimensions (nx={self.data.shape[1]}, ny={self.data.shape[2]}, nz={self.data.shape[3]}). Got crop={self.crop}')
            self.data = self.data[:, low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        if not clockwise:
            np.conjugate(self.data, out=self.data)

        if any(s<1 for s in self.data.shape):
            raise ValueError(f'Empty data dims found: data.shape={self.data.shape}')
        
        if self.N < 2:
            raise Exception(f'At least two echoes required, only {self.N} found')
    
    @property
    def N(self):
        return len(self.t)
    
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
    watCS: float = 4.7
    fatCS: tuple[float, ...] = (5.3, 4.31, 2.76, 2.1, 1.3, 0.9) # [ppm]
    relAmps: tuple[float, ...] = (0.048, 0.039, 0.004, 0.128, 0.693, 0.087)
    realEstimates: Optional[bool] = None
    
    def __new__(cls, temperature=None, **overrides):
        if cls is ModelParams and 'nFAC' in overrides:
            from mri_fatwater.FAC import FACmodelParams
            return super().__new__(FACmodelParams)
        return super().__new__(cls)
    
    def __init__(self, N=None, temperature=None, **overrides):
        init_dataclass(self, **overrides)
        
        if self.realEstimates is None:
            self.realEstimates = (N == 2)
        
        # Temperature dependence according to Hernando et al., MRM 72(2):464–70, 2014
        if temperature is not None:
            self.watCS = 1.3 + 3.748 -.01085 * temperature # Temp in [°C]

        self.CS = np.array([self.watCS] + list(self.fatCS), dtype=np.float32)

        self.set_alpha()
    
    def set_alpha(self):
        M = 2 # Number of linear components
        P = len(self.CS) # Number of resonance peaks
        self.alpha = np.zeros([M, P], dtype=np.float32)
        self.alpha[0, 0] = 1.  # Water component

        if P==2:
            self.relAmps = tuple([1.]) # Single fat peak
        if self.relAmps is None:
            raise ValueError(f'Relative amplitudes not provided for the {P-1} fat peaks.')
        if len(self.relAmps) != P-1:
            raise ValueError(f'Relative amplitudes provided for {len(self.relAmps)} fat peaks, expected {P-1}.')
        for (p, amp) in enumerate(self.relAmps):
            self.alpha[1, p+1] = float(amp)
    
    @property
    def M(self):
        return self.alpha.shape[0]
    
    @property
    def P(self):
        return self.alpha.shape[1]


@dataclass
class BaseAlgoParams:
    ALGORITHM_NAME: ClassVar[str] = 'pass'
    nR2: int = 145
    R2max: float = 144. # [sec-1]
    R2cand: tuple[float, ...] = (40.,) # [sec-1]
    nB0: int = 100
    use3D: bool = True
    magnitudeDiscrimination: bool = False
    autocrop: bool = True
    output: Optional[tuple[str, ...]] = None

    def __init__(self, realEstimates=False, **overrides):
        init_dataclass(self, **overrides)
        
        self.R2step = self.R2max/(self.nR2-1) if self.nR2 > 1 else 1. # [sec-1]
        self.iR2cand = np.array(list(set([min(self.nR2-1, int(R2/self.R2step)) for R2 in self.R2cand])))
        
        self.maxICMupdate = round(self.nB0/10)

        if self.output is None:
            self.output = ['wat', 'fat', 'ff', 'B0map']
            if realEstimates:
                self.output.append('phi')
            if (self.nR2 > 1):
                self.output.append('R2map')
    
    @property
    def algorithm(self):
        return self.ALGORITHM_NAME


@dataclass
class MRFparams(BaseAlgoParams):
    ALGORITHM_NAME: ClassVar[str] = 'ICM'
    mu: float = 0.1
    offresPenalty: float = 0.
    neighbourhoodRadius: float = 0. # [mm]
    nICMiter: int = 10
    
    def __init__(self, **overrides):
        super().__init__(**overrides)


@dataclass
class QPBOparams(MRFparams):
    ALGORITHM_NAME: ClassVar[str] = 'QPBO'
    multiScale: bool = True
    graphcutLevel: int = 0

    def __init__(self, **overrides):
        super().__init__(**overrides)


ALGORITHM_CLASSES = [BaseAlgoParams, MRFparams, QPBOparams]


def AlgoParams(algorithm, instance=None, realEstimates=False, **overrides):
    algorithm_classes = {cls.ALGORITHM_NAME: cls for cls in ALGORITHM_CLASSES}
    if algorithm not in algorithm_classes:
        raise ValueError(f'Unknown algorithm "{algorithm}". Supported algorithms are {list(algorithm_classes.keys())}.')
    cls = algorithm_classes[algorithm]
    
    if instance is None:
        kwargs = {}
    else:
        kwargs = {f.name: getattr(instance, f.name) for f in fields(cls) if hasattr(instance, f.name)}
    kwargs.update(overrides)
    return cls(realEstimates=realEstimates, **kwargs)
    

def prepare_data_params(data, data_params, data_param_file):
    params = get_params(data_param_file, data_params)
    data_file = params.pop('file') if 'file' in params else None
    filepath = params.pop('filepath') if 'filepath' in params else (Path(data_param_file).parent if data_param_file else '.')
    params['data'] = data
    if params['data'] is None:
        try:
            params['data'] = io.load_numpy_data(data_file, filepath)
        except (ValueError, FileNotFoundError) as e:
            raise RuntimeError(f'No data provided and failed to load data: {e}.') from e
    return params


def get_params(config_file, overrides):
    params = io.read_config_file(config_file)
    params.update(overrides)
    return params


def prepare(data, data_params, algo_params, model_params, data_param_file, algo_param_file, model_param_file):
    data_params = prepare_data_params(data, data_params, data_param_file)
    model_params = get_params(model_param_file, model_params)
    algo_params = get_params(algo_param_file, algo_params)
    return data_params, algo_params, model_params