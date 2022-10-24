import numpy as np
from typing import List, Tuple
from curve import Panel
from utils import *

class MatVec:

    n_pts: int
    t: np.ndarray
    dt: np.ndarray
    da: np.ndarray
    dt_da: np.ndarray
    
    n_interior_boundaries: int
    indices_of_interior_boundary: List[Tuple[int, int]]
    zk: np.ndarray # shape=(n_interior_boundaries), dtype=complex
    
    panels: List[Panel]
    close_panel_interactions = List[np.ndarray]
    
    @property
    def n_interior_boundaries(self):
        return len(self.indices_of_interior_boundary)
    
    @property
    def n_pts(self):
        return len(self.t)
    
    def Ck(self,omega):
        arr = omega*np.abs(self.dt)
        return self.singular_density(arr)

    def bk(self,omega):
        arr = -2*(omega*np.conj(self.dt)).imag
        return self.singular_density(arr)
    
    def singular_density(self, some_density):

        ret = []
        for m in range(self.n_interior_boundaries):
            start, end = self.indices_of_interior_boundary[m]
            ret.append(np.sum(some_density[start:end]))

        return np.array(ret)
    
    def __init__(self,pipe:"MultiplyConnectedPipe") -> None:
        
        self.t = pipe.t
        self.da = pipe.da
        self.dt = pipe.dt
        self.dt_da = pipe.dt_da
        self.k1_diagonal = pipe.k * np.abs(pipe.dt) / (2*np.pi)
        self.k2_diagonal = -pipe.k*pipe.dt_da * \
            pipe.dt/(2*np.pi*np.abs(pipe.dt_da))
        self.zk = np.array([b.z for b in pipe.boundaries[1:]])
        self.indices_of_interior_boundary = pipe.indices_of_boundary[1:]
        
        self.panels = pipe.panels
        self.close_panel_interactions = []
        for p in self.panels:
            l = p.arclen
            s = pt2cplx(p.start_pt)
            e = pt2cplx(p.end_pt)
            
            dist = np.abs(self.t-s) + np.abs(self.t-e)
            self.close_panel_interactions.append(np.where(dist <= 2.5*l)[0])
        
    def __call__(self,omega_sep):
        
        """this direct call will be tranferred into a call in gmres"""
        pass
    
    def velocity(self,x,y, omega):
        pass
    
    def d_phi(self,x,y, omega):
        pass

        # singular_part = np.zeros_like(regular_part, dtype=np.complex128)

        # for k in range(1, self.n_boundaries):
        #     start, end = self.indices_of_boundary[k]
        #     Ck = np.sum(omega[start:end] * np.abs(self.dt[start:end]))
        #     zk = self.boundaries[k].z
        #     singular_part += Ck/(z-zk)

        # return regular_part + singular_part

    def clean(self):
        pass
    
    