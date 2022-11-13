from utils import *
from .curve import Curve
from .cap import Cap
from .corner import Corner
from matplotlib import path
from shapely.geometry import LineString,Polygon
from shapely.ops import polylabel
from scipy.spatial import KDTree

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
    
    def dense_bdr(self,closed=True):
        pts = []
        for c in self.curves:
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
        
        for ic, c in enumerate(self.curves):
            ip = 0
            
            while ip < len(c.panels):
                
                p = c.panels[ip]
                s = p.arclen
                ip_boundary = ip + sum([len(c_.panels) for c_ in self.curves[:ic]])
                ip_boundary_next = (ip_boundary + 1) % len(self.panels)
                ip_boundary_next2 = (ip_boundary + 2) % len(self.panels)
                ip_boundary_prev = (ip_boundary - 1) % len(self.panels)
                ip_boundary_prev2 = (ip_boundary - 2) % len(self.panels)
                
                adj = [ip_boundary_next, ip_boundary_prev, ip_boundary]
                if p == 0:
                    adj.append(ip_boundary_next2)
                if p == len(c.panels) - 1:
                    adj.append(ip_boundary_prev2)
                else:
                    adj.append(ip_boundary_next2)
                    adj.append(ip_boundary_prev2)
                
                k1 = KDTree(np.array([p.x,p.y]).T)
                pts2 = np.concatenate([p2.t for j,p2 in enumerate(self.panels) if j not in adj])
                pts2 = np.array([pts2.real, pts2.imag]).T
                k2 = KDTree(pts2,compact_nodes=False)
                near = k1.query_ball_tree(k2, r=2.95*s)
                near = np.any([bool(n) for n in near])
                
                if not near: 
                    ip += 1
                    break
                
                
                c.panels.pop(ip)
                p1, p2 = p.refined()
                c.panels.insert(ip, p2)
                c.panels.insert(ip, p1)
                
        try:
            pt = polylabel(Polygon(self.plyg_bdr(1e-2)))
            self.z = pt.x + 1j*pt.y
            # checking it is inside
        except:
            self.z = np.mean(self.t)

        p = path.Path(np.array(LineString(self.dense_bdr()).buffer(0.002).interiors[0].xy).T)
        assert p.contains_point(np.array([self.z.real, self.z.imag]))
    
        for ic, c in enumerate(self.curves):
            ip = 0
            while ip < len(c.panels):
                p = c.panels[ip]
                s = p.arclen
                if np.min(np.abs(p.t - self.z)) > 3*s:
                    ip += 1
                    break
                else:
                    c.panels.pop(ip)
                    p1, p2 = p.refined()
                    c.panels.insert(ip, p2)
                    c.panels.insert(ip, p1)
    
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
    