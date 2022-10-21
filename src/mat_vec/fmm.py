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
        dj1 = self.dt*omega
        dj2 = 2*(self.dt*omega.conj()).real.astype(np.complex128)
        dipoles = np.array([dj1, dj2])
        bh_term = bhfmm2d(
            eps=FMM_EPS, pg=1, sources=self.boundary_sources,
            dipoles=dipoles).pot/(2j*np.pi)

        c_term = cfmm2d(
            eps=FMM_EPS, pg=1, sources=self.boundary_sources,
            dipstr=-2*self.dt*omega).pot/(2j*np.pi)

        return diagonal_term + bh_term + c_term

    def K_singular_terms(self, omega):
        bh_term = bhfmm2d(
            eps=FMM_EPS, pgt=1,
            sources=self.singular_sources,
            targets=self.boundary_sources,
            charges=self.singular_density(np.abs(self.dt)*omega)).pottarg
        
        c_term = cfmm2d(
            eps=FMM_EPS, pgt=1,
            sources=self.singular_sources,
            targets=self.boundary_sources,
            dipstr=self.singular_density(-2*(np.imag(np.conjugate(self.dt)*omega)))).pottarg.conj()
        return bh_term + c_term

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

    def velocity(self, x, y, omega):

        assert x.shape == y.shape
        assert x.ndim == 1

        dipoles = np.array([
            -omega*self.dt/(2j*np.pi),
            (omega.conj()*self.dt).real/(1j*np.pi),
        ])

        non_singular_term = bhfmm2d(
            eps=FMM_EPS, pgt=1, sources=self.boundary_sources, targets=np.array([x, y]),
            dipoles=dipoles).pottarg

        if self.n_interior_boundaries == 0:
            return non_singular_term

        singular_terms = np.zeros_like(non_singular_term, dtype=np.complex128)
        z = x + 1j*y

        for Ck, zk, bk in zip(self.Ck(omega), self.zk, self.bk(omega)):

            phi_singular = Ck * np.log(z-zk)
            d_phi_singular = Ck/(z-zk)
            psi_singular = np.conj(Ck) * np.log(z-zk) + \
                (bk - Ck*np.conj(zk))/(z-zk)

            singular_terms += phi_singular + z*d_phi_singular.conjugate() + \
                psi_singular.conjugate()

        return non_singular_term + singular_terms

    def __call__(self, omega_sep):
        omega = np.array(omega_sep[:len(self.t)] + 1j*omega_sep[len(self.t):])

        ret = omega + self.K_non_singular_terms(omega) 
 
        if self.n_interior_boundaries > 0:
            ret += self.K_singular_terms(omega)
 
        return np.concatenate([ret.real, ret.imag])

    def clean(self):
        super().clean()