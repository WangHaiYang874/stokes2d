from hashlib import new
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
    
    def K_singular_terms(self, omega):
        
        singular_terms = np.zeros_like(omega, dtype=np.complex128)
        
        for Ck, zk, bk in zip(self.Ck(omega), self.zk, self.bk(omega)):
            diff = self.t - zk
            singular_terms += bk/np.conjugate(diff)
            singular_terms += 2*Ck*np.log(np.abs(diff))
            singular_terms += Ck.conj() * diff / np.conjugate(diff)

        return singular_terms
    
    def velocity_singular_term(self,z, omega):
        
        singular_terms = np.zeros_like(z, dtype=np.complex128)

        for Ck, zk, bk in zip(self.Ck(omega), self.zk, self.bk(omega)):

            phi_singular = Ck * np.log(z-zk)
            d_phi_singular = Ck/(z-zk)
            psi_singular = np.conj(Ck) * np.log(z-zk) + \
                (bk - Ck*np.conj(zk))/(z-zk)

            singular_terms += phi_singular + z*d_phi_singular.conjugate() + \
                psi_singular.conjugate()
        return singular_terms
    
    def d_phi_singular_term(self,z,omega):
        
        singular_term = np.zeros_like(z, dtype=np.complex128)
        
        for Ck, zk in zip(self.Ck(omega), self.zk):
            singular_term += Ck/(z-zk)
            
        return singular_term
        
        
    
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
        
        
        
    def __call__(self,omega_sep):
        
        """this direct call will be tranferred into a call in gmres"""
        pass
    
    def pairs_needing_correction(self, z):
        
        near_panel_interactions = []
        
        for p in self.panels:
            l = p.arclen
            s = p.start_pt
            e = p.end_pt
            
            dist = np.abs(z-s) + np.abs(z-e)
            near_panel_interactions.append(np.where(dist <= 1.5*l)[0])
            
        return near_panel_interactions
    
    def velocity_correction_term(self, z, omega, pairs=None):
        
        return NotImplemented
        
        if pairs is None:
            pairs = self.pairs_needing_correction(z)
            
        correction = np.zeros_like(z, dtype=np.complex128)
        
        source_index_start = 0
        for p, target_indices in zip(self.panels,pairs):
            source_index_end = source_index_start + p.n
            target_curr = z[target_indices:np.newaxis]
            omega_curr = omega[np.newaxis,source_index_start:source_index_end]
            dt = p.dt[np.newaxis,:]
            t  = p.t[np.newaxis, :]
            t_minus_z = t - target_curr
            
            naive_eval = np.sum(
                omega_curr*dt/t_minus_z
                + t_minus_z*omega_curr.conj()*dt.conj()/(t_minus_z.conj()**2)
                - 2*np.real(omega_curr.conj()*dt)/t_minus_z.conj(),
                axis=1)/(2j*np.pi)
            
            precise_eval = None
            
            correction[target_indices] = precise_eval - naive_eval            
            
            source_index_start += p.n
            
    def d_phi_correction_term(self,z,omega,pairs=None):
        if pairs is None:
            pairs = self.pairs_needing_correction(z)                
        correction = np.zeros_like(z, dtype=np.complex128)
        source_index_start = 0
        for p, target_indices in zip(self.panels,pairs):
            source_index_end = source_index_start + p.n
            t_minus_z = p.t[np.newaxis,:] - z[target_indices][:,np.newaxis]
            naive_eval = np.sum(
                (omega[source_index_start:source_index_end]*p.dt)[np.newaxis,:]
                /(t_minus_z**2), 
                axis=1)/(2j*np.pi)
            precise_eval = p.hadamard_integral(z[target_indices]) @ omega[source_index_start:source_index_end]
            correction[target_indices] = precise_eval - naive_eval            
            source_index_start += p.n
        return correction
        
    def velocity(self,x,y, omega):
        pass
    
    def d_phi(self,x,y, omega):
        pass

    def clean(self):
        pass
    
    