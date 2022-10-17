from typing import List, Tuple
import numpy as np
from fmm2dpy import cfmm2d

FMM_EPS = 5e-16

class A_Fmm:

    t: np.ndarray
    da: np.ndarray
    dt: np.ndarray
    dt_da: np.ndarray
    k: np.ndarray
    boundary_sources: np.ndarray

    n_interior_boundaries: int
    indices_of_interior_boundary: List[Tuple[int, int]]
    zk: np.ndarray # shape=(n_interior_boundaries), dtype=complex
    singular_sources: np.ndarray  # shape=(2, n_interior_boundaries)
    
    @property
    def boundary_sources(self):
        return np.array([self.t.real, self.t.imag])

    @property
    def n_interior_boundaries(self):
        return len(self.indices_of_interior_boundary)

    def __init__(self, pipe: "MultiplyConnectedPipe") -> None:
        self.t = pipe.t
        self.da = pipe.da
        self.dt = pipe.dt
        self.dt_da = pipe.dt_da
        self.k1_diagonal = pipe.k * np.abs(pipe.dt) / (2*np.pi)
        self.k2_diagonal = -pipe.k*pipe.dt_da * \
            pipe.dt/(2*np.pi*np.abs(pipe.dt_da))
        self.zk = np.array([b.z for b in pipe.boundaries[1:]])
        self.singular_sources = np.array([self.zk.real, self.zk.imag])
        self.indices_of_interior_boundary = pipe.indices_of_boundary[1:]

    def singular_density(self, some_density):

        ret = []
        for m in range(self.n_interior_boundaries):
            start, end = self.indices_of_interior_boundary[m]
            ret.append(np.sum(some_density[start:end]))

        return np.array(ret)

    def K1(self, omega: np.ndarray):
        
        # here are the non-singualr terms
        first_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=1,
                            dipstr=self.dt*omega/(-2j*np.pi)).pot
        second_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=1,
                             dipstr=self.dt*(omega.conj())/(-2j*np.pi)).pot.conj()
        diagonal_term = self.k1_diagonal*omega

        non_singular_term = first_term + second_term + diagonal_term

        if self.n_interior_boundaries == 0:
            return non_singular_term

        # here are the singular source terms

        third_term = cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
                            dipstr=self.singular_density(-1j*self.dt*omega.conj())).pottarg.conj()

        fourth_term = cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
                             charges=self.singular_density(2*np.abs(self.dt)*omega)).pottarg

        singular_term = third_term + fourth_term

        return non_singular_term + singular_term

    def K2(self, omega_conj: np.ndarray):

        first_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=1,
                            dipstr=-np.conjugate(self.dt*omega_conj)/(2j*np.pi),).pot.conj()

        second_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=2,
                             dipstr=-self.dt*omega_conj.conj()/(2j*np.pi)).grad.conj()*self.t

        third_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=2,
                            dipstr=self.dt*omega_conj.conj()*self.t.conj()/(2j*np.pi)).grad.conj()

        diagonal_term = self.k2_diagonal*omega_conj

        non_singular_term = first_term + second_term + third_term + diagonal_term

        if self.n_interior_boundaries == 0:
            return non_singular_term

        # here are the singular source terms

        fourth_term_dipstr = []

        for (start, end), zk in zip(self.indices_of_interior_boundary,self.zk):
            dt = self.dt[start:end]
            omega = omega_conj[start:end].conj()
            fourth_term_dipstr.append(np.sum(
                (1j*dt.conj() - np.abs(dt)*np.conj(zk))*omega
            ))

        fourth_term_dipstr = np.array(fourth_term_dipstr)

        fourth_term = cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
                             dipstr=fourth_term_dipstr,).pottarg.conj()

        fifth_term = cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
                            dipstr=self.singular_density(np.abs(self.dt)*omega_conj.conj())).pottarg.conj()*self.t

        singular_term = fourth_term + fifth_term

        return non_singular_term + singular_term

    def Ck(self,omega):
        return [np.sum(omega[start:end]*np.abs(self.dt[start:end])) 
                for start, end in self.indices_of_interior_boundary]

    def bk(self,omega):
        return [-2*np.sum((omega[start:end]*np.conj(self.dt[start:end])).imag) 
                for start, end in self.indices_of_interior_boundary]

    def phi(self, x, y, omega):

        assert x.shape == y.shape
        assert x.ndim == 1

        non_singular_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=1, 
                    dipstr=omega * self.dt).pottarg / (-2j*np.pi)
        
        if self.n_interior_boundaries == 0:
            return non_singular_term
        
        singular_term = np.zeros_like(non_singular_term, dtype=np.complex128)
        
        z = x + 1j*y
        
        for Ck, zk in zip(self.Ck(omega), self.zk):
            singular_term += Ck * np.log(z-zk)
        
        return non_singular_term + singular_term            
        
    def d_phi(self, x, y, omega):
        
        assert x.shape == y.shape
        assert x.ndim == 1

        non_singular_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=2,
                    dipstr=omega * self.dt).gradtarg / (-2j*np.pi)
        
        if self.n_interior_boundaries == 0:
            return non_singular_term
        
        singular_term = np.zeros_like(non_singular_term, dtype=np.complex128)
        
        z = x + 1j*y
        
        for Ck, zk in zip(self.Ck(omega), self.zk):
            singular_term += Ck/(z-zk)
        
        return non_singular_term + singular_term            
        
    def psi(self, x,y, omega):
        
        assert x.shape == y.shape
        assert x.ndim == 1

        fisrt_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=1,
                    dipstr=np.real(omega.conj() * self.dt)).pottarg / (-1j*np.pi)
        
        second_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=2,
                    dipstr=np.real(self.t.conj()*omega*self.dt)).gradtarg / (2j*np.pi)
        
        non_singular_term = fisrt_term + second_term
        
        if self.n_interior_boundaries == 0:
            return non_singular_term
        
        singular_term = np.zeros_like(non_singular_term, dtype=np.complex128)
        
        z = x + 1j*y
        
        for Ck, zk, bk in zip(self.Ck(omega), self.zk, self.bk(omega)):
            singular_term += np.conj(Ck) * np.log(z-zk) + (bk - Ck*np.conj(zk))/(z-zk)
        
        return non_singular_term + singular_term
        
    def __call__(self, omega_sep):
        # print('A_fmm called')
        omega = np.array(omega_sep[:len(self.t)] + 1j*omega_sep[len(self.t):])
        ret = omega + self.K1(omega) + self.K2(omega.conj())
        # print('A_fmm finished')
        return np.concatenate([ret.real, ret.imag])
