from dataclasses import dataclass, field, fields, replace
import yaml
import numpy as np
from typing import Optional
from pathlib import Path
from mri_fatwater import DICOM, MATLAB


def read_config_file(config_file):
    if config_file:
        with open(config_file, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise Exception(f'Error reading config file {f}') from exc
    else:
        return {}


# dataclass parameters are set with increasing priority: 
# 1. defaults
# 2. metadata from data files specified in config file
# 3. config file parameters
# 4. excplicit overrides
def init_dataclass(dataclass_instance, configFile, **overrides):
    params = {f.name: f.default for f in fields(dataclass_instance) if f.init}

    config_data = read_config_file(configFile)

    if type(dataclass_instance).__name__ == 'DataParams':
        data_files = overrides.get('files', config_data.get('files', []))
        data_dirs = overrides.get('dirs', config_data.get('dirs', []))
        filepath = overrides.get('filepath', config_data.get('filepath', configFile.parent if configFile else './'))

        if data_files or data_dirs:
            data = load_data(data_files, data_dirs, filepath)
            params.update(data)

    params.update(config_data)

    params.update(overrides)
    params['configFile'] = configFile

    if type(dataclass_instance).__name__ == 'DataParams':
        if 'img' not in params or not isinstance(params['img'], np.ndarray) or params['img'].ndim != 4:
            raise ValueError('DataParams requires a 4D numpy array "img" with dimensions (N, nz, ny, nx)')
        if len(params['t']) != params['img'].shape[0]:
            raise ValueError(f'Number of time shifts ({len(params['t'])}) does not match number of echoes in data ({params['img'].shape[0]})')
        if 'echoes' in params:
            echoes = params.pop('echoes')
            if any(echo not in range(len(params['t'])) for echo in echoes):
                raise ValueError(f'Echo indices must be over 0 and smaller than {params['img'].shape[0]} (number of echoes in data)')
            params['t'] = tuple(params['t'][i] for i in echoes)
            params['img'] = params['img'][echoes, ...]
        if 'sliceList' in params:
            sliceList = params.pop('sliceList')
            if any(slice not in range(params['img'].shape[1]) for slice in sliceList):
                raise ValueError(f'Slice indices must be over 0 and smaller than {params['img'].shape[1]} (number of slices in data)')
            params['img'] = params['img'][:, sliceList, ...]
        if params['pad']:
            if 'crop' not in params or params['crop'] is None:
                params['pad'] = False
            else:
                dataclass_instance.original_shape = params['img'].shape[1:]
        if 'crop' in params and params['crop'] is not None:
            if len(params['crop']) != 6:
                raise ValueError('Param "crop" must be a list of six integers: [x0, y0, z0, x1, y1, z1]')
            if params['crop'][0] < 0 or params['crop'][1] < 0 or params['crop'][2] < 0 or params['crop'][3] > params['img'].shape[3] or params['crop'][4] > params['img'].shape[2] or params['crop'][5] > params['img'].shape[1]:
                raise ValueError(f'Param "crop" values must be within the image dimensions (nx={params['img'].shape[3]}, ny={params['img'].shape[2]}, nz={params['img'].shape[1]})')
            params['img'] = params['img'][:, params['crop'][2]:params['crop'][5], params['crop'][1]:params['crop'][4], params['crop'][0]:params['crop'][3]]
        if 'clockwisePrecession' in params:
            clockwisePrecession = params.pop('clockwisePrecession')
            if not clockwisePrecession:
                np.conjugate(params['img'], out=params['img'])

    for param in params:
        if hasattr(dataclass_instance, param):
            setattr(dataclass_instance, param, params[param])
        else:
            msg = f'Unknown parameter "{param}" passed to {type(dataclass_instance).__name__} constructor'
            if param not in overrides:
                if param in config_data:
                    msg += f' (from config file {configFile})'
                elif param in data:
                    msg += f' (from data file(s) {data_files} and/or data dirs {data_dirs})'
            raise Exception(msg)


def load_data(data_files, data_dirs, base_path):

    if len(data_files) > 0:
        data_files = [base_path / file for file in list(data_files) if Path(base_path / file).is_file()]
    
    if len(data_dirs) > 0:
        data_dirs = [base_path / dir for dir in list(data_dirs) if Path(base_path / dir).is_dir()]
        for path in data_dirs:
            data_files += [obj for obj in path.iterdir() if obj.is_file()]
    
    valid_DICOM_files = DICOM.getValidFiles(data_files)
    
    if valid_DICOM_files:
        return DICOM.readData(valid_DICOM_files)
    elif len(data_files) == 1 and data_files[0].suffix == '.mat':
        return MATLAB.readISMRMchallengeData(data_files[0])
    else:
        raise Exception('No valid files found')


@dataclass
class DataParams:
    img: Optional[np.ndarray] = field(default=None, repr=False)
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
    files: tuple[str, ...] = field(default=(), repr=False)
    dirs: tuple[str, ...] = field(default=(), repr=False)

    configFile: Optional[str] = field(default=None, repr=False)

    def __init__(self, configFile: Optional[str] = None, **overrides):
        init_dataclass(self, configFile, **overrides)
        
        self.img *= self.reScale

        if self.N < 2:
            raise Exception(f'At least two echoes required, only {self.N} found')
    
    @property
    def N(self):
        return len(self.t)
    
    @property
    def nz(self):
        return self.img.shape[1]
    
    @property
    def ny(self):
        return self.img.shape[2]
    
    @property
    def nx(self):
        return self.img.shape[3]
    
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
    nFAC: int = 0
    CL: float = 17.4 # Derived from Lundbom et al., NMR in Biomed 23(5):466–72, 2010
    P2U: float = 0.2 # Derived from Lundbom et al., NMR in Biomed 23(5):466–72, 2010
    UD: float = 2.6  # Derived from Lundbom et al., NMR in Biomed 23(5):466–72, 2010

    configFile: Optional[str] = field(default=None, repr=False)

    def __init__(self, configFile: Optional[str] = None, temperature=None, **overrides):
        init_dataclass(self, configFile, **overrides)
        
        if self.nFAC > 0:
            if self.nFAC > 3:
                raise ValueError(f'Unknown number of FAC parameters: {self.nFAC}')
            if len(self.fatCS) != 10:
                raise ValueError('FAC excpects exactly one water and ten triglyceride resonances')
            self.relAmps = None
        
        # Temperature dependence according to Hernando et al., MRM 72(2):464–70, 2014
        if temperature is not None:
            self.watCS = 1.3 + 3.748 -.01085 * temperature # Temp in [°C]

        self.CS = np.array([self.watCS] + self.fatCS, dtype=np.float32)
        
        self.M = 2 + self.nFAC # Number of linear components
        self.P = len(self.CS) # Number of resonance peaks
        self.alpha = np.zeros([self.M, self.P], dtype=np.float32)
        self.alpha[0, 0] = 1.  # Water component
        CL, UD, P2U = self.CL, self.UD, self.P2U # for readability later
        if self.M == 2:
            if self.relAmps is not None:
                for (p, amp) in enumerate(self.relAmps):
                    self.alpha[1, p+1] = float(amp)
            elif self.P==1:
                self.alpha[1, 1] = 1. # Single fat peak
            elif self.P==11:
                # F = 9A+(6(CL-4)+UD(2P2U-8))B+6C+4UD(1-P2U)D+6E+2UDP2UF+2G+2H+I+2UDJ
                self.alpha[1, 1:] = [9, 6*(CL-4)+UD*(2*P2U-8), 6, 4*UD*(1-P2U), 6, 2*UD*P2U,
                                2, 2, 1, UD*2]
            else:
                raise ValueError(f'Relative amplitudes not provided for the {self.P-1} fat peaks.')
        elif self.M == 3:
            # F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
            # F2 = (2P2U-8)B+4(1-P2U)D+2P2UF+2J
            self.alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
            self.alpha[2, 1:] = [0, 2*P2U-8, 0, 4*(1-P2U), 0, 2*P2U, 0, 0, 0, 2]
        elif self.M == 4:
            # F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
            # F2 = -8B+4D+2J
            # F3 = 2B-4D+2F
            self.alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
            self.alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
            self.alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
        elif self.M == 5:
            # F1 = 9A-24B+6C+6E+2G+2H+I
            # F2 = -8B+4D+2J
            # F3 = 2B-4D+2F
            # F4 = 6B
            self.alpha[1, 1:] = [9, -24, 6, 0, 6, 0, 2, 2, 1, 0]
            self.alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
            self.alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
            self.alpha[4, 1:] = [0, 6, 0, 0, 0, 0, 0, 0, 0, 0]


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

    configFile: Optional[str] = field(default=None, repr=False)

    def __init__(self, configFile: Optional[str] = None, N=None, nFAC=0, **overrides):
        init_dataclass(self, configFile, **overrides)
    
        if self.realEstimates is None:
            self.realEstimates = (N == 2)
        
        self.R2step = self.R2max/(self.nR2-1) if self.nR2 > 1 else 1. # [sec-1]
        self.iR2cand = np.array(list(set([min(self.nR2-1, int(R2/self.R2step)) for R2 in self.R2cand])))
        
        self.maxICMupdate = round(self.nB0/10)

        self.output = ['wat', 'fat', 'ff', 'B0map']
        if self.realEstimates:
            self.output.append('phi')
        if (self.nR2 > 1):
            self.output.append('R2map')
        if (nFAC > 2):
            self.output.append('CL')
        if (nFAC > 1):
            self.output.append('PUD')
        if (nFAC > 0):
            self.output.append('UD')