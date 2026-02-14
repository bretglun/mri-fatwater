from dataclasses import dataclass, field, fields, replace
import yaml
import numpy as np
from typing import Optional


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

    def __init__(self, configFile: Optional[str] = None, **overrides):
        params = {f.name: f.default for f in fields(self) if f.init}

        if configFile:
            with open(configFile, 'r') as f:
                try:
                    params.update(yaml.safe_load(f))
                except yaml.YAMLError as exc:
                    raise Exception(f'Error reading config file {f}') from exc

        params.update(overrides)
        params['configFile'] = configFile

        for param in params:
            if hasattr(self, param):
                setattr(self, param, params[param])
            elif param in overrides:
                raise Exception(f'Unknown algorithm parameter "{param}" passed to AlgoParams constructor')
            else:
                raise Exception(f'Unknown algorithm parameter "{param}" in config file {configFile}')

    def setup(self, N, nFAC=0):
        if self.realEstimates is None:
            self.realEstimates = (N == 2)
        
        self.R2step = self.R2max/(self.nR2-1) if self.nR2 > 1 else 1. # [sec-1]
        self.iR2cand = np.array(list(set([min(self.nR2-1, int(R2/self.R2step)) for R2 in self.R2cand])))
        
        self.maxICMupdate = round(self.nB0/10)

        # For Fatty Acid Composition, create algorithmParams for two passes: self and self.pass2
        # First pass: use standard fat-water separation to determine B0 and R2*
        # Second pass: use B0- and R2*-maps from first pass
        if nFAC > 0:
            self.pass2 = replace(self, nICMiter=0, graphcut=False, graphcutLevel=None)
        
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