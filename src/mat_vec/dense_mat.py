from .mat_vec import *
from numpy import newaxis, conjugate, pi
import numpy as np


class DenseMat(MatVec):

    K1: np.ndarray
    K2: np.ndarray

    def __init__(self, pipe: "MultiplyConnectedPipe") -> None:
        super().__init__(pipe)
        self.K1 = None
        self.K2 = None
        self.build_k()

    def __call__(self,omega_sep):
        
        omega = omega_sep[:self.n_pts] + 1j*omega_sep[self.n_pts:]
        ret = self.K_non_singular_terms(omega)
        if self.n_interior_boundaries > 0:
            ret += self.K_singular_terms(omega)
        return np.concatenate([ret.real, ret.imag])

    def K_non_singular_terms(self, omega):
        if self.K1 is None or self.K2 is None:
            self.build_k()
        return omega + self.K1 @ omega + self.K2 @ conjugate(omega)

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

        z = x + 1j*y
        z = z[:, newaxis]
        omega = omega[newaxis, :]
        dt = self.dt[newaxis, :]
        t = self.t[newaxis, :]
        t_minus_z = t - z

        return np.sum(omega * np.imag(dt/t_minus_z) +
                      np.conj(omega) * np.imag(t_minus_z*np.conj(dt)) / np.conj(t_minus_z**2), axis=1) / pi

    def d_phi_non_singular_terms(self, x, y, omega):
        assert x.shape == y.shape
        assert x.ndim == 1
        assert omega.shape == (self.n_pts,)
        z = x + 1j*y

        return np.sum((omega * self.dt)[newaxis, :] /
                      (self.t[newaxis, :] - z[:, newaxis])**2, axis=1) / (2j * pi)

    def clean(self):
        self.K1 = None
        self.K2 = None
        super().clean()


    def build_k(self):
        diff_t = self.t[:, newaxis] - self.t[newaxis, :]
        dt = self.dt[newaxis, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            K1 = np.imag(dt/diff_t) / (-pi)
            K2 = np.imag(dt * conjugate(diff_t))/(conjugate(diff_t**2)*pi)

        np.fill_diagonal(K1, self.k1_diagonal)
        np.fill_diagonal(K2, self.k2_diagonal)
        self.K1 = K1
        self.K2 = K2