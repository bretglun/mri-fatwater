# FAC = Fatty Acid Composition
# See Berglund et al. 2012 “Model-Based Mapping of Fat Unsaturation and Chain Length by Chemical Shift Imaging—Phantom Validation and in Vivo Feasibility.” MRM 68(6):1815–27.
import numpy as np
from dataclasses import dataclass, replace
from mri_fatwater import fatwater, params
from mri_fatwater.params import ModelParams
from .constants import EPSILON

output = {'CL', 'UD', 'PUD'}

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


def get_results(rho, output):
    results = {}
    CL, UD, PUD = getFattyAcidComposition(rho)
    if 'CL' in output:
        results['CL'] = CL
    if 'UD' in output:
        results['UD'] = UD
    if 'PUD' in output:
        results['PUD'] = PUD
    return results


def run_FAC_passes(dPar, aPar, mPar):
    if mPar.nFAC==0:
        return fatwater.run_separation_passes([(dPar, aPar, mPar)])
    output = ['UD']
    if (mPar.nFAC > 1):
        output.append('PUD')
    if (mPar.nFAC > 2):
        output.append('CL')
    mPar1 = replace(mPar, nFAC=0, relAmps=None)
    aPar2 = params.AlgoParams(algorithm='pass', instance=aPar, output=output)
    passes = [
        (dPar, aPar, mPar1), # First pass: use standard fat-water separation to determine B0 and R2*
        (dPar, aPar2, mPar)  # Second pass: use B0- and R2*-maps from first pass and do the Fatty Acid Composition
    ]
    return fatwater.run_separation_passes(passes)


@dataclass
class FACmodelParams(ModelParams):
    # Fatty Acid Composition: Default values derived from Lundbom et al., NMR in Biomed 23(5):466–72, 2010
    nFAC: int = 0
    CL: float = 17.4 # Fatty acid carbon Chain Length
    UD: float = 2.6  # Unsaturation Degree (number of double bonds per triglyceride)
    P2U: float = 0.2 # Ratio of PUD to UD (PUD=PolyUnsaturation Degree; number of double bond pairs separated by a single methylene group per triglyceride)

    def __init__(self, **overrides):
        super().__init__(**overrides)
    
    def set_alpha(self):
        M = 2 + self.nFAC # Number of linear components
        P = len(self.CS) # Number of resonance peaks
        self.alpha = np.zeros([M, P], dtype=np.float32)
        self.alpha[0, 0] = 1.  # Water component

        if self.nFAC > 3 or self.nFAC < 0:
            raise ValueError(f'Unknown number of FAC parameters: {self.nFAC}')
        if len(self.fatCS) != 10:
            raise ValueError('FAC excpects exactly one water and ten triglyceride resonances')
        self.relAmps = None
        
        CL, UD, P2U = self.CL, self.UD, self.P2U # for readability later
        if M == 2:
            # F = 9A+(6(CL-4)+UD(2P2U-8))B+6C+4UD(1-P2U)D+6E+2UDP2UF+2G+2H+I+2UDJ
            self.alpha[1, 1:] = [9, 6*(CL-4)+UD*(2*P2U-8), 6, 4*UD*(1-P2U), 6, 2*UD*P2U, 2, 2, 1, UD*2]
        elif M == 3:
            # F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
            # F2 = (2P2U-8)B+4(1-P2U)D+2P2UF+2J
            self.alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
            self.alpha[2, 1:] = [0, 2*P2U-8, 0, 4*(1-P2U), 0, 2*P2U, 0, 0, 0, 2]
        elif M == 4:
            # F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
            # F2 = -8B+4D+2J
            # F3 = 2B-4D+2F
            self.alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
            self.alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
            self.alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
        elif M == 5:
            # F1 = 9A-24B+6C+6E+2G+2H+I
            # F2 = -8B+4D+2J
            # F3 = 2B-4D+2F
            # F4 = 6B
            self.alpha[1, 1:] = [9, -24, 6, 0, 6, 0, 2, 2, 1, 0]
            self.alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
            self.alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
            self.alpha[4, 1:] = [0, 6, 0, 0, 0, 0, 0, 0, 0, 0]