'''
from the Curve, we have figured out how to draw stuff:
the lines, obstructions, and caps, smoothed corners, etc. 
those are going to be just a Panel for a tube.

In this file, I'll write the codes to automate the process of 
drawing tubes from the codes of the Curve. There are certain
things I wish this file can do. 

1. a tube should be a sampler, instead of fixing number of points on 
the Panel, we should be able to sample points on the tube to make sure
our stokes2d solver can have given tolerance and distance away from the 
boundary, as the main numerical error appears only near the boundary. In
my previous experiements, 5h rule works out pretty well for controlling the 
numerical error. 

2. I also want to write a codes for several generic tubes with different specifications
such as the radius, the length, the number of bifurcations, how large should the smoothed
Corner be, etc.
'''

import sys
import warnings
import numpy as np

sys.path.append('.')
from curve import *
from utility_and_spec import *
from scipy.sparse.linalg import gmres, LinearOperator, aslinearoperator
import fmm2dpy as fmm

class Pipe:
    def __init__(self) -> None:

        # geometric data.
        self.curves = None

        # graph data.
        self.lets  = None # stands for inlets and outlets
        self.omegas = None
        self.pressure_drops = None

        # drawing data
        self.extent = None
        self.grids = None
        self.velocity_field = None
        self.vortity = None
        self.pressure = None

    @property
    def a(self): return np.concatenate([c.a + 2 * i for i, c in enumerate(self.curves)])
    @property
    def da(self): return np.concatenate([c.da for c in self.curves])
    @property
    def x(self): return np.concatenate([c.x for c in self.curves])
    @property
    def y(self): return np.concatenate([c.y for c in self.curves])
    @property
    def dx_da(self): return np.concatenate([c.dx_da for c in self.curves])
    @property
    def dy_da(self): return np.concatenate([c.dy_da for c in self.curves])
    @property

    def ddx_dda(self): return np.concatenate([c.ddx_dda for c in self.curves])
    @property
    def ddy_dda(self): return np.concatenate([c.ddy_dda for c in self.curves])
    @property
    def t(self): return np.concatenate([c.t for c in self.curves])
    @property
    def dt_da(self): return np.concatenate([c.dt_da for c in self.curves])
    @property
    def k(self): return np.concatenate([c.k for c in self.curves])

    def build_geometry(self, max_distance=None, legendre_ratio=None):

        self.curves = [c.build(max_distance,legendre_ratio) for c in self.curves]

    def build_graph(self):
        self.lets = [i for i,c in enumerate(self.curves) if isinstance(c,Cap)]
        assert len(self.lets) > 1

        self.inlet = self.lets[0]
        self.outlets = self.lets[1:]
        self.flows = [(self.inlet,o) for o in self.outlets]

    def build_A(self,if_fmm=False):
        """not implementing fmm for now. """

        da = self.da
        t = self.t
        dt_da = self.dt_da
        k = self.k

        dt = t[:, np.newaxis] - t[np.newaxis, :]
        d = dt_da[np.newaxis, :]
        da_ = da[np.newaxis, :]

        # this ignores the error for computing the diagonal elements with 0/0 error
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

        n = len(self.a)

        A = np.zeros((2*n, 2*n))
        A[:n, :n] = np.identity(n) + (K1+K2).real
        A[:n, n:] = (-K1+K2).imag
        A[n:, :n] = (K1+K2).imag
        A[n:, n:] = np.identity(n) + (K1-K2).real

        self.A = aslinearoperator(A)


    def compute_omega(self,U,tol=1e-12):
        H = H2U(U)
        b = np.concatenate((H.real,H.imag))

        omega,_ = gmres(self.A, b,atol=0,tol=tol)

        if _ < 0:
            warnings.warn("gmres is not converging to tolerance. ")

        return omega

    def compute_velocity(self,x,y,omega):
        t = self.t
        dt = self.dt_da * self.da
        z = x + 1j*y
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

    def compute_pressure_and_vorticity(self,x,y,omega):
        z = x + 1j*y
        assert (isinstance(z, np.ndarray))
        shape = z.shape
        z = z.flatten()

        t = self.get_t()
        dt = self.get_dt_da() * self.da

        t_minus_z_sq = (t[np.newaxis, :] - z[:, np.newaxis]) ** 2
        d_phi = np.sum((omega * dt)[np.newaxis, :] /
                       (t_minus_z_sq), axis=1) / (2j * np.pi)

        pressure = np.imag(d_phi)
        vorticity = np.real(d_phi)
        return pressure.reshape(shape), vorticity.reshape(shape)


    def build_pressure_drops(self):
        pass
    
    def build_velocity_fields(self):
        pass
    
    def build_graph(self):
        pass
    
    def build(self):
        pass
    
    def is_inside(self, z):
        pass

class StraightPipe(Pipe):
    def __init__(self,p1,p2,r=1) -> None:
        """
        this creates a simple tube. Why do I create it first? because it serves well as
        a template for other more sophisticated geometries.
        """
        super().__init__()
                
        
        self.angle = np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
        theta = self.angle + np.pi/2
        top_left = p1 + r*np.array([np.cos(theta),np.sin(theta)])
        bottom_left = p1 - r*np.array([np.cos(theta),np.sin(theta)])
        top_right = p2 + r*np.array([np.cos(theta),np.sin(theta)])
        bottom_right = p2 - r*np.array([np.cos(theta),np.sin(theta)])
        
        top_line = Line(top_left,top_right)
        right_line = Line(top_right,bottom_right)
        bottom_line = Line(bottom_right,bottom_left)
        left_line = Line(bottom_left,top_left)
        
        self.curves = [top_line,right_line,bottom_line,left_line]

        up = np.max(top_left[1],top_right[1],bottom_left[1],bottom_right[1])
        low = np.min(top_left[1],top_right[1],bottom_left[1],bottom_right[1])
        left = np.min(top_left[0],top_right[0],bottom_left[0],bottom_right[0])
        right = np.max(top_left[0],top_right[0],bottom_left[0],bottom_right[0])


