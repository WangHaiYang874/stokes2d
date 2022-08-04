import numpy as np
from scipy.sparse.linalg import gmres
import numbers
from basic_spec import *


class stokes2d:
    def __init__(self, geometry, gmres_tol=5e-13):
        
        self.geometry = geometry
        self.gmres_tol = gmres_tol
        self.build_A()

    def build_A(self):
        '''this builds the matrix for the Nystorm discretization'''
        
        # compute the kernels
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

        # building the equation for gmres
        
        n = len(self.geometry.a)
        self.n = n
        
        A = np.zeros((2*n, 2*n))
        A[:n, :n] = np.identity(n) + (K1+K2).real
        A[:n, n:] = (-K1+K2).imag
        A[n:, :n] = (K1+K2).imag
        A[n:, n:] = np.identity(n) + (K1-K2).real
        
        self.A = A
    
    def clean_A(self):
        self.A = None
        
    def solve(self,U):

        H = U2H(U)
        
        rhs = np.concatenate((H.real, H.imag))
        
        print('gmres starts solving the Nystorm, please wait...')
        
        omega, _ = gmres(self.A, rhs, tol=self.gmres_tol, atol=0)
        
        if _ == 0:
            print('gmres converged')
        else:
            print('gmres did not converge')
            print('gmres error:', _)
        omega = omega[:self.n] + 1j*omega[self.n:]
        
        return omega

    def compute_velocity(self, z, omega):
        
        t = self.geometry.get_t()
        dt = self.geometry.get_dt_da()*self.geometry.da

        if isinstance(z, numbers.Number):
            t_minus_z = t - z
            t_minus_z_sq = t_minus_z**2

            phi = np.sum(omega*dt/t_minus_z)/(2j*np.pi)
            d_phi = np.sum(omega*dt/t_minus_z_sq)/(2j*np.pi)

            psi = (1/(2j*np.pi))*(
                2*np.sum(np.real(np.conjugate(omega)*dt)/t_minus_z)
                - np.sum(np.conjugate(t)*omega*dt/t_minus_z_sq))

        else:
            assert isinstance(z, np.ndarray)
            shape = z.shape
            z = z.flatten()
            
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
        
        ret = ret.reshape(shape)
        
        return H2U(ret)

    def compute_grad_pressure(self, z, omega):

        t = self.geometry.get_t()
        dt = self.geometry.get_dt_da()*self.geometry.da

        if isinstance(z, numbers.Number):
            dd_phi = np.sum(omega*dt/(t-z)**3)/(1j*np.pi)

        else:
            assert isinstance(z, np.ndarray) and z.ndim == 1
            t_minus_z_cubic = (t[np.newaxis, :] - z[:, np.newaxis])**3
            dd_phi = np.sum((omega*dt)[np.newaxis, :] /
                            t_minus_z_cubic, axis=1)/(1j*np.pi)

        grad_p = -4*np.array((dd_phi.imag,dd_phi.real)).T
        return grad_p
    
    
class stokes2dGlobal:
    pass
