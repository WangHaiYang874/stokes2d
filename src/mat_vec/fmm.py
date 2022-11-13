from .mat_vec import MatVec
from curve import Line
import numpy as np
from fmm2dpy import cfmm2d, bhfmm2d
from utils import FMM_EPS


class Fmm(MatVec):

    boundary_sources: np.ndarray

    @property
    def boundary_sources(self):
        return np.array([self.t.real, self.t.imag])

    def __init__(self, pipe: "MultiplyConnectedPipe") -> None:

        super().__init__(pipe)

    def __call__(self,omega_sep):
        
        omega = omega_sep[:self.n_pts] + 1j*omega_sep[self.n_pts:]
        ret = self.K_non_singular_terms(omega)
        if self.n_interior_boundaries > 0:
            ret += self.K_singular_terms(omega)
        
        # I an attempting to add an extra term to let
        # the gmres to handle a linear system with
        # no rank deficiency. 

        flux = np.real(np.sum(omega*np.conjugate(self.dt)))
        extra_term = flux*self.dt
        ret += extra_term
        
        
        # this term is ignored in professor Greengard's paper. 
        # could it be the cause of stagnation?
        
        # b0 = self.b0(omega)
        # extra_term2 = np.conj(b0/(self.t-self.z0))
        # ret += extra_term2
        
        # print(np.abs(b0))
        
        return np.concatenate([ret.real, ret.imag])

    def K_non_singular_terms(self, omega):
        
        diagonal_term = omega \
            + self.k1_diagonal * omega \
            + self.k2_diagonal * np.conj(omega)

        dipoles = np.array([
            -omega*self.dt/(2j*np.pi),
            (omega.conj()*self.dt).real/(1j*np.pi),
        ])
        
        bh_term = bhfmm2d(
            eps=FMM_EPS, pg=1, sources=self.boundary_sources,
            dipoles=dipoles).pot
        
        # correction_term = np.zeros_like(omega,dtype=np.complex128)
        
        # index = np.insert(np.cumsum([len(c.t) for c in self.curves]), 0, 0)
        # index = [(index[i], index[i+1]) for i in range(len(index)-1)]
        
        # for c,(start,end) in zip(self.curves,index):
            
        #     if isinstance(c,Line):
            
        #         dip = dipoles[:,start:end]
        #         src = self.boundary_sources[:,start:end]
                
        #         correction_term[start:end] = bhfmm2d(
        #             eps=FMM_EPS, pg=1, sources=src, dipoles=dip).pot

        return diagonal_term + bh_term # - correction_term

    def velocity(self,x,y, omega):
        ret = self.velocity_non_singular_terms(x, y, omega)
        if self.n_interior_boundaries > 0:
            ret += self.velocity_singular_terms(x + 1j*y, omega)
        ret += self.velocity_correction_terms(x + 1j*y, omega)
        
        return ret

    def d_phi(self,x,y, omega):
        ret = self.d_phi_non_singular_terms(x, y, omega)
        if self.n_interior_boundaries > 0:
            ret += self.d_phi_singular_terms(x + 1j*y, omega)
        ret += self.d_phi_correction_terms(x + 1j*y, omega)
        return ret

    def velocity_non_singular_terms(self, x, y, omega):
        assert x.shape == y.shape
        assert x.ndim == 1

        dipoles = np.array([
            -omega*self.dt/(2j*np.pi),
            (omega.conj()*self.dt).real/(1j*np.pi),
        ])

        return bhfmm2d(
            eps=FMM_EPS, pgt=1, sources=self.boundary_sources, targets=np.array([x, y]),
            dipoles=dipoles).pottarg

    def d_phi_non_singular_terms(self, x, y, omega):
        assert x.shape == y.shape
        assert x.ndim == 1

        return cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=2,
                      dipstr=omega * self.dt).gradtarg / (-2j*np.pi)
