from typing import List, Tuple
import numpy as np
from fmm2dpy import cfmm2d
from utils import FMM_EPS
from scipy.linalg import lu_factor, lu_solve


class Preconditioner:

    def __init__(self, E, F) -> None:
        self.E = E
        self.F = F
        self.S = -np.matmul(E,F)
        self.S_lu = lu_factor(self.S)
        self.n = E.shape[0]//2
        self.m = E.shape[1]//2
        
    def __call__(self, r_omega_sep_and_c_sep) -> np.ndarray:
        r_omega_sep = r_omega_sep_and_c_sep[:2*self.n]
        r_c_sep = r_omega_sep_and_c_sep[2*self.n:]
        z_c_sep = lu_solve(self.S_lu,r_c_sep-self.F@r_omega_sep)
        z_omega_sep = r_omega_sep - self.E@z_c_sep
        
        return np.concatenate((z_omega_sep,z_c_sep))

class A_fmm:

    t: np.ndarray
    da: np.ndarray
    dt: np.ndarray
    dt_da: np.ndarray
    k: np.ndarray
    boundary_sources: np.ndarray

    n_interior_boundaries: int
    indices_of_interior_boundary: List[Tuple[int, int]]
    zk: np.ndarray  # shape=(n_interior_boundaries), dtype=complex
    singular_sources: np.ndarray  # shape=(2, n_interior_boundaries)
    preconditioner: Preconditioner
    
    @property
    def boundary_sources(self):
        return np.array([self.t.real, self.t.imag])

    @property
    def n_interior_boundaries(self):
        return len(self.indices_of_interior_boundary)

    def __init__(self, pipe) -> None:
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
        self.build_preconditioner()

    def build_preconditioner(self):

        E1_complex = 2*np.abs(self.dt[:, np.newaxis]) * \
            np.log(np.abs(self.t[:, np.newaxis] - self.zk[np.newaxis, :]))
        E2_complex = (self.t[:, np.newaxis] - self.zk[np.newaxis,:])/np.conjugate(self.t[:, np.newaxis] - self.zk[np.newaxis,:])
        
        n,m = E1_complex.shape
        E = np.zeros((2*n,2*m),dtype=np.float64)
        E[:n,:m] = (E1_complex+E2_complex).real
        E[n:,:m] = (E1_complex+E2_complex).imag
        E[:n,m:] = (-E1_complex+E2_complex).imag
        E[:n,m:] = (E2_complex-E1_complex).real
        

        F_complex = np.zeros((m,n),dtype=np.complex128)
        
        for i,(start,end) in enumerate(self.indices_of_interior_boundary):
            F_complex[i,start:end] = np.abs(self.dt[start:end])
        
        F = np.zeros((2*m,2*n),dtype=np.float64)
        F[:m,:n] = F_complex.real
        F[m:,:n] = F_complex.imag
        F[:m,n:] = -F_complex.imag
        F[m:,n:] = F_complex.real
        
        self.preconditioner = Preconditioner(E,F)
        
    def singular_density(self, some_density):
        ret = []
        for m in range(self.n_interior_boundaries):
            start, end = self.indices_of_interior_boundary[m]
            ret.append(np.sum(some_density[start:end]))
        return np.array(ret)

    # def K1_total(self, omega: np.ndarray):
    #     return self.K1_0(omega) + self.K1_1(omega) + self.K1_2(omega) + self.K1_3(omega) + self.K1_4(omega)

    def K1_precondition(self, omega: np.ndarray):
        return self.K1_0(omega) + self.K1_1(omega) + self.K1_2(omega) + self.K1_3(omega)

    def K1_0(self, omega: np.ndarray):
        return self.k1_diagonal*omega

    def K1_1(self, omega: np.ndarray):
        return cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=1,
                      dipstr=self.dt*omega/(-2j*np.pi)).pot

    def K1_2(self, omega: np.ndarray):
        return cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=1,
                      dipstr=self.dt*(omega.conj())/(-2j*np.pi)).pot.conj()

    def K1_3(self, omega: np.ndarray):
        return cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
                      dipstr=self.singular_density(-1j*self.dt*omega.conj())).pottarg.conj()

    # def K1_4(self, omega: np.ndarray):
    #     # singular source term
    #     return cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
    #                   charges=self.singular_density(2*np.abs(self.dt)*omega)).pottarg

    # def K2_total(self, omega_conj: np.ndarray):
    #     return self.K2_0(omega_conj) + self.K2_1(omega_conj) + self.K2_2(omega_conj) + self.K2_3(omega_conj) + self.K2_4(omega_conj) + self.K2_5(omega_conj) + self.K2_6(omega_conj)

    def K2_precondition(self, omega_conj: np.ndarray):
        return self.K2_0(omega_conj) + self.K2_1(omega_conj) + self.K2_2(omega_conj) + self.K2_3(omega_conj) + self.K2_4(omega_conj)

    def K2_0(self, omega_conj: np.ndarray):
        return self.k2_diagonal*omega_conj

    def K2_1(self, omega_conj: np.ndarray):
        return cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=1,
                      dipstr=-np.conjugate(self.dt*omega_conj)/(2j*np.pi),).pot.conj()

    def K2_2(self, omega_conj: np.ndarray):
        return cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=2,
                      dipstr=-self.dt*omega_conj.conj()/(2j*np.pi)).grad.conj()*self.t

    def K2_3(self, omega_conj: np.ndarray):
        return cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, pg=2,
                      dipstr=self.dt*omega_conj.conj()*self.t.conj()/(2j*np.pi)).grad.conj()

    def K2_4(self, omega_conj: np.ndarray):
        return cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
                      dipstr=self.singular_density(1j*self.dt.conj()*omega_conj.conj())).pottarg.conj()

    # def K2_5(self, omega_conj: np.ndarray):
    #     return cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
    #                   dipstr=self.singular_density(np.abs(self.dt)*omega_conj.conj())).pottarg.conj()*self.t

    # def K2_6(self, omega_conj: np.ndarray):
    #     sixth_density = []

    #     for (start, end), zk in zip(self.indices_of_interior_boundary, self.zk):
    #         dt = self.dt[start:end]
    #         omega = omega_conj[start:end].conj()
    #         sixth_density.append(-np.sum((np.abs(dt)*np.conj(zk))*omega))

    #     sixth_density = np.array(sixth_density)
    #     return cfmm2d(eps=FMM_EPS, sources=self.singular_sources, targets=self.boundary_sources, pgt=1,
    #                   dipstr=sixth_density,).pottarg.conj()

    def bk(self, omega):
        return [-2*np.sum((omega[start:end]*np.conj(self.dt[start:end])).imag)
                for start, end in self.indices_of_interior_boundary]

    def phi(self, x, y, omega, C):

        assert x.shape == y.shape
        assert x.ndim == 1

        non_singular_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=1,
                                   dipstr=omega * self.dt).pottarg / (-2j*np.pi)

        if self.n_interior_boundaries == 0:
            return non_singular_term

        singular_term = np.zeros_like(non_singular_term, dtype=np.complex128)

        z = x + 1j*y

        for Ck, zk in zip(C, self.zk):
            singular_term += Ck * np.log(z-zk)

        return non_singular_term + singular_term

    def d_phi(self, x, y, omega, C):

        assert x.shape == y.shape
        assert x.ndim == 1

        non_singular_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=2,
                                   dipstr=omega * self.dt).gradtarg / (-2j*np.pi)

        if self.n_interior_boundaries == 0:
            return non_singular_term

        singular_term = np.zeros_like(non_singular_term, dtype=np.complex128)

        z = x + 1j*y

        for Ck, zk in zip(C, self.zk):
            singular_term += Ck/(z-zk)

        return non_singular_term + singular_term

    def psi(self, x, y, omega, C):

        assert x.shape == y.shape
        assert x.ndim == 1

        fisrt_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=1,
                            dipstr=np.real(omega.conj() * self.dt)).pottarg / (-1j*np.pi)

        second_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources, targets=np.array([x, y]), pgt=2,
                             dipstr=self.t.conj()*omega*self.dt).gradtarg / (2j*np.pi)

        non_singular_term = fisrt_term + second_term

        if self.n_interior_boundaries == 0:
            return non_singular_term

        singular_term = np.zeros_like(non_singular_term, dtype=np.complex128)

        z = x + 1j*y

        for Ck, zk, bk in zip(C, self.zk, self.bk(omega)):
            singular_term += np.conj(Ck) * np.log(z-zk) + \
                (bk - Ck*np.conj(zk))/(z-zk)

        return non_singular_term + singular_term

    def __call__(self, omega_sep_and_c_sep):
        # print('A_fmm called')
        
        omega_sep = omega_sep_and_c_sep[:2*len(self.t)]
        c_sep = omega_sep_and_c_sep[2*len(self.t):]
        
        omega = np.array(omega_sep[:len(self.t)] + 1j*omega_sep[len(self.t):])
        
        EC_sep = self.preconditioner.E@c_sep
        EC = EC_sep[:len(self.t)] + 1j*EC_sep[len(self.t):]
        
        h = omega + self.K1_precondition(omega) + self.K2_precondition(omega.conj()) + EC
        F_omega_sep = self.preconditioner.F@omega_sep
        
        return np.concatenate([h.real, h.imag, F_omega_sep])
    
        
