from dataclasses import dataclass, field, fields, replace
import yaml
import numpy as np
from typing import Optional
from pathlib import Path
from mri_fatwater import DICOM, MATLAB

def init_dataclass(dataclass_instance, configFile, **overrides):
    params = {f.name: f.default for f in fields(dataclass_instance) if f.init}

    if configFile:
        with open(configFile, 'r') as f:
            try:
                params.update(yaml.safe_load(f))
            except yaml.YAMLError as exc:
                raise Exception(f'Error reading config file {f}') from exc

    params.update(overrides)
    params['configFile'] = configFile

    for param in params:
        if hasattr(dataclass_instance, param):
            setattr(dataclass_instance, param, params[param])
        else:
            raise Exception(f'Unknown parameter "{param}" passed to {type(dataclass_instance).__name__} constructor' + (f' (from config file {configFile})' if param not in overrides else ''))


@dataclass
class DataParams:
    reScale: float = 1.0
    temperature: Optional[float] = None
    clockwisePrecession: bool = False
    offresCenter: float = 0.
    files: tuple[str, ...] = ()
    dirs: tuple[str, ...] = ()
    sliceList: tuple[int, ...] = ()
    outDir: Optional[str] = None

    configFile: Optional[str] = field(default=None, repr=False)

    def __init__(self, configFile: Optional[str] = None, **overrides):
        init_dataclass(self, configFile, **overrides)

        if self.outDir is None:
            raise ValueError('No outDir defined')

        filepath = self.configFile.parent if self.configFile else None

        if len(self.files) > 0:
            self.files = [filepath / file for file in list(self.files) if Path(filepath / file).is_file()]
        
        if len(self.dirs) > 0:
            self.dirs = [filepath / dir for dir in list(self.dirs) if Path(filepath / dir).is_dir()]
            for path in self.dirs:
                self.files += [obj for obj in path.iterdir() if obj.is_file()]
        
        validFiles = DICOM.getValidFiles(self.files)
        
        if validFiles:
            DICOM.updateDataParams(self, validFiles)
        else:
            if len(self.files) == 1 and self.files[0].suffix == '.mat':
                MATLAB.updateDataParams(self, self.files[0])
            else:
                raise Exception('No valid files found')
        
        if hasattr(self, 'reconSlab'):
            self.slabs = self.getSlabs(self.sliceList, self.reconSlab)
    
    def getSlabs(self, sliceList, reconSlab):
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

    def __init__(self, configFile: Optional[str] = None, clockwisePrecession=False, temperature=None, **overrides):
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
        if clockwisePrecession:
            self.CS *= -1
        
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