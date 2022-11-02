from utils import *
from .curve import Curve
from .cap import Cap
from .corner import Corner
from matplotlib import path
from shapely.geometry import LineString

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
        self.check_correctedness()
        
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
    
    
    @property
    def caps(self):
        return [c for c in self.curves if isinstance(c, Cap)]

    @property
    def panels(self):
        return [p for c in self.curves for p in c.panels]
    
    @property
    def orientation(self):
        orientation = np.sum(self.dt/(self.t-self.z))/(2j*np.pi)
        if np.abs(orientation - 1) < 1e-10:
            return 1
        elif np.abs(orientation + 1) < 1e-10:
            return -1
        else:
            assert False, "Unknown orientation"
    
    @property
    def leftest_nodes(self):
        ret = np.inf
        for c in self.curves:
            ret = min(c.start_pt[0], ret)
            if isinstance(c, Corner):
                ret = min(c.mid_pt[0], ret)
        return ret
        
        
    def plyg_bdr(self,closed=True):
        pts = []
        for c in self.curves:
            pts += [c.start_pt]
            if isinstance(c, Corner):
                pts += [[c.x[i], c.y[i]] for i in range(len(c.a))]
        
        if not closed: return np.array(pts)
        return np.array(pts + [pts[0]])
        
    def smth_plyg_bdr(self,closed=True):
        pts = []
        for c in self.curves:
            pts += [c.start_pt]
            if isinstance(c, Corner) or isinstance(c, Cap):
                pts += [[c.x[i], c.y[i]] for i in range(len(c.a))]
        if not closed: return np.array(pts)
        return np.array(pts + [pts[0]])

    # def reverse_orientation(self):
    #     pass # TODO
    
    def clean_copy(self):
        new_curves = [c.clean_copy() for c in self.curves]
        return Boundary(new_curves)
    
    def transformed(self, shift, ):
        ret = self.clean_copy()
        ret.curves = [c.transformed(shift) for c in ret.curves]
        return ret        
    
    def build(self, required_tol=REQUIRED_TOL, p=None):
        
        if p is None:
            p = (np.ceil(-np.log10(required_tol)) + 2).astype(int)
        
        [c.build(required_tol,p) for c in self.curves]
    
    def inside(self, xs, ys):
        p = path.Path(np.array(LineString(self.plyg_bdr()).buffer(0.002).interiors[0].xy).T)
        return np.array(p.contains_points(np.array([xs, ys]).T))
    
    def outside(self, xs, ys):
        p = path.Path(np.array(LineString(self.plyg_bdr()).buffer(0.002).exterior.xy).T)
        return ~np.array(p.contains_points(np.array([xs, ys]).T))
    
    @property
    def extent(self):
        pts = self.plyg_bdr()
        
        xmin = np.min(pts[:,0])
        xmax = np.max(pts[:,0])
        ymin = np.min(pts[:,1])
        ymax = np.max(pts[:,1])
        
        return (xmin, xmax, ymin, ymax)
    
    def check_correctedness(self):

        for i,c in enumerate(self.curves):
            j = (i+1)%len(self.curves)
            if np.max(c.end_pt - self.curves[j].start_pt) > 1e-12:
                assert False
        
        # TODO should also check self-intersection. 
    