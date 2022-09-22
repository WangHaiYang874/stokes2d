import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator, aslinearoperator
import numbers
from utility_and_spec import *
import fmm2dpy as fmm
from time import time


class stokes2d:
    def __init__(self, geometry, gmres_tol=1e-12):

        self.geometry = geometry
        self.gmres_tol = gmres_tol
        self.A = None
        self.A_fmm = None
        self.K1_diagonal = None
        self.K2_diagonal = None

    ''' the matrix version of nystorm '''

    def build_A(self):
        '''this builds the matrix for the Nystorm discretization'''

        # compute the kernels
        _, da, t, dt_da, k = self.geometry.get_data()
        dt = t[:, np.newaxis] - t[np.newaxis, :]
        d = dt_da[np.newaxis, :]
        da_ = da[np.newaxis, :]

        # this ignore the error for computing the diagonal elements with 0/0 error
        with np.errstate(divide='ignore', invalid='ignore'):
            K1 = -da_ * np.imag(d / dt) / np.pi
            K2 = -da_ * (-d / np.conjugate(dt) + np.conjugate(d)
                         * dt / (np.conjugate(dt ** 2))) / (2j * np.pi)
        # now we need to fill the diagonal elements
        d = dt_da
        K1_diagonal = k * np.abs(d) * da / (2 * np.pi)
        K2_diagonal = -da * k * (d ** 2) / (np.abs(d) * 2 * np.pi)
        np.fill_diagonal(K1, K1_diagonal)
        np.fill_diagonal(K2, K2_diagonal)

        # building the equation for gmres

        n = len(self.geometry.a)
        self.n = n

        A = np.zeros((2 * n, 2 * n))
        A[:n, :n] = np.identity(n) + (K1 + K2).real
        A[:n, n:] = (-K1 + K2).imag
        A[n:, :n] = (K1 + K2).imag
        A[n:, n:] = np.identity(n) + (K1 - K2).real

        self.A = aslinearoperator(A)

    def clean_A(self):
        self.A = None

    def compute_omega(self, U, if_fmm=True, if_callback=False, if_lgmres=False):
        """compute the omega from Nystorm discretization

        Args:
            U (np.ndarray with shape (n,2) and dtype float64): 
            the velocity condition on the boundary. 

        Returns:
            omega (np.ndarray with shape (n,) and dtype complex128): 
            the omega on the boundary. 
        """
        H = U2H(U)
        rhs = np.concatenate((H.real, H.imag))

        if if_fmm:
            if self.A_fmm is None:
                self.build_A_fmm()
            A = self.A_fmm
        else:
            if self.A is None:
                print("building the Nystorm matrix")
                self.build_A()
            A = self.A

        # print('running gmres')

        callback = None

        if if_callback:
            callback = gmres_callback(A, rhs)

        # lgmres does not help with the convergence.

        omega, _ = gmres(A, rhs,
                         tol=self.gmres_tol, atol=0,
                         maxiter=100, restart=10,
                         callback=callback, callback_type='x')

        if _ > 0:
            print('gmres did not converge after', _, ' iterations')
        if _ < 0:
            print('gmres breakdown')

        n = len(self.geometry.a)
        omega = omega[:n] + 1j * omega[n:]

        if if_callback:
            return omega, callback
        return omega

    def build_A_fmm(self):
        '''this builds the matrix for the Nystorm discretization'''

        # compute the kernels
        _, da, _, dt_da, k = self.geometry.get_data()

        self.K1_diagonal = k * np.abs(dt_da) * da / (2 * np.pi)
        self.K2_diagonal = -da * k * (dt_da ** 2) / (np.abs(dt_da) * 2 * np.pi)

        def A_fmm(omega_sep):
            n = len(omega_sep)
            assert n % 2 == 0
            n = n // 2

            omega = omega_sep[:n] + 1j * omega_sep[n:]

            K1 = self.K1_fmm(omega)
            K2 = self.K2_fmm(omega.conj())

            ret = omega + K1 + K2
            ret_real = np.real(ret)
            ret_imag = np.imag(ret)

            return np.concatenate((ret_real, ret_imag))

        n = len(self.geometry.a)

        self.A_fmm = LinearOperator(
            dtype=np.float64, shape=(2 * n, 2 * n), matvec=A_fmm)

    def K1_fmm(self, omega):

        eps = 1e-17
        _, da, _, dt_da, _ = self.geometry.get_data()
        sources = np.array([self.geometry.x, self.geometry.y])
        dt = dt_da * da

        K11 = fmm.cfmm2d(eps=eps,
                         sources=sources,
                         dipstr=- dt * omega / (2j * np.pi),
                         pg=1
                         ).pot

        K12 = fmm.cfmm2d(eps=eps,
                         sources=sources,
                         dipstr=- dt * omega.conjugate() / (2j * np.pi),
                         pg=1
                         ).pot.conjugate()

        # diagonal elements
        K1_diag = self.K1_diagonal * omega

        return K11 + K12 + K1_diag

        return K21 + K221 + K222 + K2_diag

    def compute_velocity_direct(self, z, omega):

        t = self.geometry.get_t()
        dt = self.geometry.get_dt_da() * self.geometry.da

        if isinstance(z, numbers.Number):
            t_minus_z = t - z
            t_minus_z_sq = t_minus_z ** 2

            phi = np.sum(omega * dt / t_minus_z) / (2j * np.pi)
            d_phi = np.sum(omega * dt / t_minus_z_sq) / (2j * np.pi)

            psi = (1 / (2j * np.pi)) * (
                    2 * np.sum(np.real(np.conjugate(omega) * dt) / t_minus_z)
                    - np.sum(np.conjugate(t) * omega * dt / t_minus_z_sq))

        else:
            assert isinstance(z, np.ndarray)
            shape = z.shape
            z = z.flatten()

            t_minus_z = t[np.newaxis, :] - z[:, np.newaxis]
            t_minus_z_sq = t_minus_z ** 2

            phi = np.sum((omega * dt)[np.newaxis, :] /
                         t_minus_z, axis=1) / (2j * np.pi)
            d_phi = np.sum((omega * dt)[np.newaxis, :] /
                           (t_minus_z_sq), axis=1) / (2j * np.pi)

            psi = (1 / (2j * np.pi)) * (
                    2 * np.sum(np.real((np.conjugate(omega) * dt)
                                       [np.newaxis, :]) / t_minus_z, axis=1)
                    - np.sum((np.conjugate(t) * omega * dt)[np.newaxis, :] / t_minus_z_sq, axis=1))

        ret = phi + z * np.conjugate(d_phi) + np.conjugate(psi)

        ret = ret.reshape(shape)

        return H2U(ret)

    def compute_velocity_fmm(self, z, omega):
        '''
        this only support the case when z is a 1-d numpy array
        '''

        eps = 1e-17
        t = self.geometry.get_t()
        dt = self.geometry.get_dt_da() * self.geometry.da
        charges = np.zeros_like(t)
        sources = np.array([self.geometry.x, self.geometry.y])

        x = z.real
        y = z.imag

        targets = np.array([x, y])

        phi = fmm.cfmm2d(eps=eps,
                         sources=sources,
                         charges=charges,
                         dipstr=omega * dt,
                         targets=targets,
                         pgt=2)

        phi, d_phi = phi.pottarg / (-2j * np.pi), phi.gradtarg / (-2j * np.pi)

        psi1 = fmm.cfmm2d(eps=eps,
                          sources=sources,
                          charges=charges,
                          dipstr=np.real(omega.conjugate() * dt),
                          targets=targets,
                          pgt=1).pottarg / (-1j * np.pi)

        psi2 = fmm.cfmm2d(eps=eps,
                          sources=sources,
                          charges=charges,
                          dipstr=np.real(np.conjugate(t) * omega * dt),
                          targets=targets,
                          pgt=2).gradtarg / (2j * np.pi)

        psi = psi1 + psi2

        H = phi + (x + 1j * y) * d_phi.conjugate() + psi.conjugate()

        return H2U(H)

    def compute_pressure_direct(self, z, omega):
        # see equation (14) from the paper.
        # p = Im phi'(z), up to a constant factor.

        t = self.geometry.get_t()
        dt = self.geometry.get_dt_da() * self.geometry.da

        if isinstance(z, numbers.Number):
            t_minus_z_sq = (t - z) ** 2
            d_phi = np.sum(omega * dt / t_minus_z_sq) / (2j * np.pi)
            return np.imag(d_phi)

        assert (isinstance(z, np.ndarray))
        shape = z.shape
        z = z.flatten()

        t_minus_z_sq = (t[np.newaxis, :] - z[:, np.newaxis]) ** 2
        d_phi = np.sum((omega * dt)[np.newaxis, :] /
                       (t_minus_z_sq), axis=1) / (2j * np.pi)

        pressure = np.imag(d_phi)
        return pressure.reshape(shape)

    def compute_pressure_fmm(self, z, omega):
        eps = 1e-17
        t = self.geometry.get_t()
        dt = self.geometry.get_dt_da() * self.geometry.da
        charges = np.zeros_like(t)
        sources = np.array([self.geometry.x, self.geometry.y])

        x = z.real
        y = z.imag

        targets = np.array([x, y])

        d_phi = fmm.cfmm2d(eps=eps,
                           sources=sources,
                           charges=charges,
                           dipstr=omega * dt,
                           targets=targets,
                           pgt=2).gradtarg / (-2j * np.pi)

        return np.imag(d_phi)


class gmres_callback:
    def __init__(self, A, b):
        self.counter = 0
        self.pr_norm = []
        self.A = A
        self.b = b
        self.norm = []

    def __call__(self, x):
        self.counter += 1
        self.norm.append(np.linalg.norm(self.A.matvec(x) - self.b) / np.linalg.norm(self.b))
        if self.counter > 10:
            if np.mean(self.norm[-5:]) > np.mean(self.norm[-10:-5]): return True


class stokes2dGlobal:
    pass
