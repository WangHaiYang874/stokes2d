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
    
    @property
    def indices_of_panels(self):
        index = np.insert(np.cumsum([p.n for p in self.panels]), 0, 0)
        return [(index[i], index[i+1]) for i in range(len(index)-1)]

    def __init__(self,pipe:"MultiplyConnectedPipe") -> None:
        
        self.t = pipe.t
        self.da = pipe.da
        self.dt = pipe.dt
        self.dt_da = pipe.dt_da
        self.k1_diagonal = pipe.k * np.abs(pipe.dt) / (2*np.pi)
        self.k2_diagonal = -pipe.k*pipe.dt_da**2 * pipe.da / (2*np.pi*np.abs(pipe.dt_da))
        self.zk = np.array([b.z for b in pipe.boundaries[1:]])
        self.indices_of_interior_boundary = pipe.indices_of_boundary[1:]
        
        self.panels = pipe.panels
        self.curves = pipe.curves
        self.z0 = pipe.z0
        
    def b0(self,omega):
        return np.real(np.sum(omega*self.dt/((self.t-self.z0)**2)))/(1j*np.pi)
    
    def Ck(self,omega):
        arr = omega*np.abs(self.dt)
        return self.singular_density(arr)

    def bk(self,omega):
        arr = -2*np.imag(omega*np.conjugate(self.dt))
        return self.singular_density(arr)
    
    def singular_density(self, some_density):

        ret = []
        for m in range(self.n_interior_boundaries):
            start, end = self.indices_of_interior_boundary[m]
            ret.append(np.sum(some_density[start:end]))

        return np.array(ret)

    # def __call__(self,omega_sep):
        
    #     omega = omega_sep[:self.n_pts] + 1j*omega_sep[self.n_pts:]
    #     ret = self.K_non_singular_terms(omega)
    #     if self.n_interior_boundaries > 0:
    #         ret += self.K_singular_terms(omega)
    #     return np.concatenate([ret.real, ret.imag])        
    
    # def K_non_singular_terms(self,omega):
    #     pass
    
    def K_singular_terms(self, omega):

        singular_terms = np.zeros_like(omega, dtype=np.complex128)

        for Ck, zk, bk in zip(self.Ck(omega), self.zk, self.bk(omega)):
            diff = self.t - zk
            singular_terms += np.conjugate(bk/diff)
            singular_terms += 2*Ck*np.log(np.abs(diff))
            singular_terms += np.conjugate(Ck) * diff / np.conjugate(diff)

        return singular_terms
    
    
    def velocity_singular_terms(self,z, omega):
        singular_terms = np.zeros_like(z, dtype=np.complex128)

        for Ck, zk, bk in zip(self.Ck(omega), self.zk, self.bk(omega)):

            phi_singular = Ck * np.log(z-zk)
            d_phi_singular = Ck/(z-zk)
            psi_singular = np.conj(Ck) * np.log(z-zk) + \
                (bk - Ck*np.conj(zk))/(z-zk)

            singular_terms += phi_singular + z*d_phi_singular.conjugate() + \
                psi_singular.conjugate()
        return singular_terms

    def velocity_correction_terms(self, z, omega, pairs=None):
        if pairs is None:
            pairs = self.pairs_needing_correction(z)
            
        correction = np.zeros_like(z, dtype=np.complex128)
        for i,(p, target_indices) in enumerate(zip(self.panels,pairs)):
            s_start, s_end = self.indices_of_panels[i]
            
            dt = p.dt[np.newaxis,:]
            t_minus_z = p.t[np.newaxis,:] - z[target_indices][:,np.newaxis]
            o = omega[s_start:s_end][np.newaxis,:]
            naive_eval = np.sum(o*np.imag(dt/t_minus_z) + o.conj()*np.imag(t_minus_z*dt.conj())/t_minus_z.conj()**2, axis=1)/np.pi
            
            o = omega[s_start:s_end]
            C,H = p.both_integrals(z[target_indices])
            
            precise_eval = 2*np.real(C)@o + z[target_indices]*np.conj(H@o) + np.conj(C@(o*np.conj(p.dt_da)/p.dt_da) - H@(o*np.conj(p.t)))
            correction[target_indices] += (precise_eval - naive_eval)
            
        return correction

    def d_phi_singular_terms(self,z,omega):
        singular_term = np.zeros_like(z, dtype=np.complex128)
        
        for Ck, zk in zip(self.Ck(omega), self.zk):
            singular_term += Ck/(z-zk)
            
        return singular_term
        
    def d_phi_correction_terms(self,z,omega,pairs=None):
        if pairs is None:
            pairs = self.pairs_needing_correction(z)
        correction = np.zeros_like(z, dtype=np.complex128)
        for i,(p, target_indices) in enumerate(zip(self.panels,pairs)):
            s_start, s_end = self.indices_of_panels[i]
            naive_eval = np.sum(
                (omega[s_start:s_end]*p.dt)[np.newaxis,:] /
                (p.t[np.newaxis,:] - z[target_indices][:,np.newaxis])**2, axis=1) / (2j*np.pi)
            precise_eval = p.hadamard_integral(z[target_indices]) @ omega[s_start:s_end]
            correction[target_indices] += (precise_eval - naive_eval)
            
        return correction
        
    def pairs_needing_correction(self, z):
        
        near_panel_interactions = []
        
        for p in self.panels:
            dist = np.abs(z-p.start_pt) + np.abs(z-p.end_pt)
            near_panel_interactions.append(dist <= 3*p.arclen)
            
        return near_panel_interactions
    
    def clean(self):
        pass
    
    