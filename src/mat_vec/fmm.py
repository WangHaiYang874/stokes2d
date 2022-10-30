from .mat_vec import MatVec

import numpy as np
from fmm2dpy import cfmm2d, bhfmm2d
from utils import FMM_EPS


class Fmm(MatVec):

    boundary_sources: np.ndarray
    singular_sources: np.ndarray  # shape=(2, n_interior_boundaries)

    @property
    def boundary_sources(self):
        return np.array([self.t.real, self.t.imag])

    def __init__(self, pipe: "MultiplyConnectedPipe") -> None:
        
        super().__init__(pipe)
        self.singular_sources = np.array([self.zk.real, self.zk.imag])
        
    def K_non_singular_terms(self, omega):

        diagonal_term = self.k1_diagonal*omega + \
            (self.k2_diagonal)*omega.conj()
        
        dipoles = np.array([
            -self.dt*omega/(2j*np.pi), 
            (self.dt*omega.conj()).real/(1j*np.pi)])
        
        bh_term = bhfmm2d(
            eps=FMM_EPS, pg=1, sources=self.boundary_sources,
            dipoles=dipoles).pot

        return diagonal_term + bh_term

    def d_phi(self, x, y, omega):

        assert x.shape == y.shape
        assert x.ndim == 1

        ret = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=2,
                        dipstr=omega * self.dt).gradtarg / (-2j*np.pi)

        if self.n_interior_boundaries > 0:
            ret += self.d_phi_singular_term(x + 1j*y, omega) 

        return ret

    def velocity(self, x, y, omega):

        assert x.shape == y.shape
        assert x.ndim == 1

        dipoles = np.array([
            omega*self.dt/(2j*np.pi),
            -(omega.conj()*self.dt).real/(1j*np.pi),
        ])

        ret = bhfmm2d(
            eps=FMM_EPS, pgt=1, sources=self.boundary_sources, targets=np.array([x, y]),
            dipoles=dipoles).pottarg

        if self.n_interior_boundaries > 0:
            ret += self.velocity_singular_term(x + 1j*y, omega)
        
        return ret

    def __call__(self, omega_sep):
        omega = np.array(omega_sep[:len(self.t)] + 1j*omega_sep[len(self.t):])
        ret = omega + self.K_non_singular_terms(omega) 
        if self.n_interior_boundaries > 0:
            ret += self.K_singular_terms(omega)
        return np.concatenate([ret.real, ret.imag])