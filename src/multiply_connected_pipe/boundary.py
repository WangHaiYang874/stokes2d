from cProfile import label
from utils import *
from curve import *

import numpy as np
from shapely.ops import polylabel
from shapely.geometry import LineString
from matplotlib.path import Path

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
    
    def clean_copy(self):
        new_curves = [c.clean_copy() for c in self.curves]
        return Boundary(new_curves)
    
    
    
    def inside_mask(self,xs,ys):
        pts = []
        for c in self.curves:
            if isinstance(c, Cap):
                pts += [c.start_pt, c.matching_pt]
            elif isinstance(c, Line):
                pts += [c.start_pt, c.mid_pt]
            elif isinstance(c, Corner):
                pts += [[c.x[i], c.y[i]] for i in range(len(c.a))]
        polygon = np.array(pts)
    
        return Path(polygon).contains_points(np.array([xs, ys]).T)
        
    def near_mask(self,xs,ys,dist):
        
        pts = []
        for c in self.curves:
            if isinstance(c, Line):
                pts += [c.start_pt, c.end_pt]
            elif isinstance(c, Corner) or isinstance(c, Cap):
                pts += [[c.x[i], c.y[i]] for i in range(len(c.a))]
        
        polygon = np.concatenate([pts, [pts[0]]])

        linestring = LineString(polygon).buffer(dist)

        exterior_bdr_pts = np.array(linestring.exterior.coords.xy).T
        in_exterior = Path(np.concatenate((exterior_bdr_pts, [exterior_bdr_pts[0]]))).contains_points(np.array([xs, ys]).T)
        
        not_in_interior = np.full_like(xs, True, dtype=bool)
        
        for interior in linestring.interiors:
            interior_bdr_pts = np.array(interior.coords.xy).T
            
            not_in_interior &= (~np.array(
                Path(np.concatenate((interior_bdr_pts, [interior_bdr_pts[0]])))
                .contains_points(np.array([xs, ys]).T), dtype=bool))

        return in_exterior & not_in_interior
        
        
        
        
        

        