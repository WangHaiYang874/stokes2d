from cProfile import label
from utils import *
from curve import *

import numpy as np
from shapely.ops import polylabel
from shapely.geometry import LineString

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
    def z(self):
        # here I assume that the domain is convex,
        # so I will simply take the average of points on the boundary.
        return np.mean(self.t)
    
        # TODO: the more careful algorithm to handle non-convex domains is to use
        # poles of inaccessibility.
        # PIA has a convenient python implementation at
        # https://github.com/shapely/shapely/blob/main/shapely/algorithms/polylabel.py
        line = LineString(np.array((self.t.real, self.t.imag)).T)
        line = line.buffer(5e-2)
        label = polylabel(line, tolerance=1e-1)
        x, y = label.wkt[0], label.wkt[1]
        
        

    

        
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
            
            # TODO REVERSE ORIENTATION

            pass
        
    
    def build(self, max_distance=None, legendre_ratio=None):
        [c.build(max_distance, legendre_ratio) for c in self.curves]
    
    