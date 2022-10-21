import numpy as np
from typing import List, Tuple

class MatVec:

    n_pts: int
    t: np.ndarray
    dt: np.ndarray
    da: np.ndarray
    dt_da: np.ndarray
    
    n_interior_boundaries: int
    indices_of_interior_boundary: List[Tuple[int, int]]
    zk: np.ndarray # shape=(n_interior_boundaries), dtype=complex
    
    @property
    def n_interior_boundaries(self):
        return len(self.indices_of_interior_boundary)
    
    @property
    def n_pts(self):
        return len(self.t)
    
    def Ck(self,omega):
        return [np.sum(omega[start:end]*np.abs(self.dt[start:end])) 
                for start, end in self.indices_of_interior_boundary]

    def bk(self,omega):
        return [-2*np.sum((omega[start:end]*np.conj(self.dt[start:end])).imag) 
                for start, end in self.indices_of_interior_boundary]
    
    def singular_density(self, some_density):

        ret = []
        for m in range(self.n_interior_boundaries):
            start, end = self.indices_of_interior_boundary[m]
            ret.append(np.sum(some_density[start:end]))

        return np.array(ret)
    
    def __init__(self,) -> None:
        pass
    
    
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

    
    
    