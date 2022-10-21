from .mat_vec import *
from numpy import newaxis, conjugate, pi
import numpy as np


class DenseMat(MatVec):

    K1: np.ndarray
    K2: np.ndarray

    def __init__(self, pipe: "MultiplyConnectedPipe") -> None:

        self.t = pipe.t
        self.da = pipe.da
        self.dt = pipe.dt
        self.dt_da = pipe.dt_da
        self.zk = np.array([b.z for b in pipe.boundaries[1:]])
        self.indices_of_interior_boundary = pipe.indices_of_boundary[1:]

        # Construct K1 and K2
        diff_t = self.t[:, newaxis] - self.t[newaxis, :]
        dt2 = self.dt[newaxis, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            K1 = np.imag(dt2/diff_t) / (-pi)
            K2 = (dt2 / conjugate(diff_t) - conjugate(dt2)
                  * diff_t/(conjugate(diff_t**2))) / (2j*pi)

        K1_diagonal = pipe.k*np.abs(self.dt)/(2*pi)
        K2_diagonal = -pipe.k*self.dt*self.dt_da / (2*pi*np.abs(self.dt_da))
        np.fill_diagonal(K1, K1_diagonal)
        np.fill_diagonal(K2, K2_diagonal)
        self.K1 = K1
        self.K2 = K2

    def __call__(self, omega_sep):

        assert omega_sep.shape == (2*self.n_pts,)

        omega = omega_sep[:self.n_pts] + 1j*omega_sep[self.n_pts:]
        ret = omega + self.K1 @ omega + self.K2 @ conjugate(omega)
        return np.concatenate((np.real(ret), np.imag(ret)))

    def velocity(self, x, y, omega):
        assert x.shape == y.shape
        assert x.ndim == 1
        assert omega.shape == (self.n_pts,)
        z = x + 1j*y

        z = z[:, newaxis]
        omega = omega[newaxis, :]
        dt = self.dt[newaxis, :]
        t = self.t[newaxis, :]
        t_minus_z = t - z

        non_singular_term = np.sum(
            omega*dt/t_minus_z
            + t_minus_z*omega.conj()*dt.conj()/(t_minus_z.conj()**2)
            - 2*np.real(omega.conj()*dt)/t_minus_z.conj(),
            axis=1)/(2j*pi)

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

    def d_phi(self, x, y, omega):
        assert x.shape == y.shape
        assert x.ndim == 1
        assert omega.shape == (self.n_pts,)
        z = x + 1j*y

        non_singular_term = np.sum((omega * self.dt)[newaxis, :] /
                                   (self.t[newaxis, :] - z[:, newaxis])**2, axis=1) / (2j * pi)

        if self.n_interior_boundaries == 0:
            return non_singular_term

        singular_term = np.zeros_like(non_singular_term, dtype=np.complex128)


        for Ck, zk in zip(self.Ck(omega), self.zk):
            singular_term += Ck/(z-zk)

        return non_singular_term + singular_term
