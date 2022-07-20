import numpy as np
from scipy.sparse.linalg import gmres
import numbers


class stokes2d:
    def __init__(self, geometry, u, v, rho=1, nu=1):
        self.geometry = geometry
        self.u = u
        self.v = v
        self.h2 = u
        self.h1 = -v
        self.rho = rho
        self.nu = nu
        self.compute_kernels()
        self.solve()

    def compute_kernels(self):
        _, da, t, dt_da, k = self.geometry.get_data()
        dt = t[:, np.newaxis] - t[np.newaxis, :]
        d = dt_da[np.newaxis, :]
        da_ = da[np.newaxis, :]

        # this ignore the error for computing the diagonal elements with 0/0 error

        with np.errstate(divide='ignore', invalid='ignore'):
            K1 = -da_ * np.imag(d/dt) / np.pi
            K2 = -da_ * (-d/np.conjugate(dt) + np.conjugate(d)
                         * dt/(np.conjugate(dt**2))) / (2j*np.pi)

        # now we need to fill the diagonal elements
        d = dt_da
        K1_diagonal = k*np.abs(d)*da/(2*np.pi)
        K2_diagonal = -da*k*(d**2)/(np.abs(d)*2*np.pi)
        np.fill_diagonal(K1, K1_diagonal)
        np.fill_diagonal(K2, K2_diagonal)

        self.K1 = K1
        self.K2 = K2

    def solve(self):
        n = len(self.geometry.a)
        A = np.zeros((2*n, 2*n))
        A[:n, :n] = np.identity(n) + (self.K1+self.K2).real
        A[:n, n:] = (-self.K1+self.K2).imag
        A[n:, :n] = (self.K1+self.K2).imag
        A[n:, n:] = np.identity(n) + (self.K1-self.K2).real
        rhs = np.concatenate((self.h1, self.h2))
        print('gmres starts solving the Nystorm, please wait...')
        omega, _ = gmres(A, rhs, tol=1e-12, maxiter=1000)
        if _ == 0:
            print('gmres converged')
        else:
            print('gmres did not converge')
            print('gmres error:', _)
        omega = omega[:n] + 1j*omega[n:]
        self.omega = omega

    def compute_velocity(self, z):

        t = self.geometry.get_t()
        dt = self.geometry.get_dt_da()*self.geometry.da
        omega = self.omega

        if isinstance(z, numbers.Number):
            t_minus_z = t - z
            t_minus_z_sq = t_minus_z**2

            phi = np.sum(omega*dt/t_minus_z)/(2j*np.pi)
            d_phi = np.sum(omega*dt/t_minus_z_sq)/(2j*np.pi)

            psi = (1/(2j*np.pi))*(
                2*np.sum(np.real(np.conjugate(omega)*dt)/t_minus_z)
                - np.sum(np.conjugate(t)*omega*dt/t_minus_z_sq))

        else:
            assert isinstance(z, np.ndarray) and z.ndim == 1
            t_minus_z = t[np.newaxis, :] - z[:, np.newaxis]
            t_minus_z_sq = t_minus_z**2

            phi = np.sum((omega*dt)[np.newaxis, :] /
                         t_minus_z, axis=1)/(2j*np.pi)
            d_phi = np.sum((omega*dt)[np.newaxis, :] /
                           (t_minus_z_sq), axis=1)/(2j*np.pi)

            psi = (1/(2j*np.pi))*(
                2*np.sum(np.real((np.conjugate(omega)*dt)
                         [np.newaxis, :])/t_minus_z, axis=1)
                - np.sum((np.conjugate(t)*omega*dt)[np.newaxis, :]/t_minus_z_sq, axis=1))

        ret = phi + z*np.conjugate(d_phi) + np.conjugate(psi)
        return ret.imag - 1j*ret.real

    def compute_pressure(self, z):

        t = self.geometry.get_t()
        dt = self.geometry.get_dt_da()*self.geometry.da
        omega = self.omega

        if isinstance(z, numbers.Number):
            dd_phi = np.sum(omega*dt/(t-z)**3)/(1j*np.pi)

        else:
            assert isinstance(z, np.ndarray) and z.ndim == 1
            t_minus_z_cubic = (t[np.newaxis, :] - z[:, np.newaxis])**3
            dd_phi = np.sum((omega*dt)[np.newaxis, :] /
                            t_minus_z_cubic, axis=1)/(1j*np.pi)

        grad_p = -4*self.rho*self.nu*(dd_phi.imag + 1j*dd_phi.real)
        return grad_p
