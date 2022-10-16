from utils import *
from curve import *

import numpy as np

class Boundary:
    
    curves: List[Curve]
    n_pts: int
    a: np.ndarray
    da: np.ndarray
    t: np.ndarray
    dt_da: np.ndarray
    dt: np.ndarray
    k: np.ndarray
    z: np.complex128
    
    def __init__(self, curves):
        self.curves = curves
        
    @property
    def a(self): return np.concatenate(
        [c.a + 2*i for i, c in enumerate(self.curves)])
    
    @property
    def n_pts(self): return len(self.a)
    
    @property
    def da(self): return np.concatenate(
        [c.da for c in self.curves])
    
    @property
    def t(self): return np.concatenate(
        [c.t for c in self.curves])
    
    @property
    def dt_da(self): return np.concatenate(
        [c.dt_da for c in self.curves])
    
    @property
    def dt(self): return self.dt_da * self.da
    
    @property
    def k(self): return np.concatenate(
        [c.k for c in self.curves])
    
    @property
    def z(self): return np.mean(self.t)
    
    @property
    def orientation(self):
        orientation = np.sum(self.dt/(self.t-self.z))/(2j*np.pi)
        if np.abs(orientation - 1) < 1e-10:
            return 1
        elif np.abs(orientation + 1) < 1e-10:
            return -1
        else:
            assert False, "Unknown orientation"
    
    def reverse_orientation(self):
        
        for curve in self.curves:
            
            # TODO I should define this locally for each curve. 

            pass
        
    
    def build(self, max_distance=None, legendre_ratio=None):
        [c.build(max_distance, legendre_ratio) for c in self.curves]
    
    