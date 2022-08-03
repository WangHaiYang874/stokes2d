import numpy as np
from scipy.special import p_roots as gauss_quad_nodes
from scipy.integrate import quad
from math_functions import *

class geometry:
    def __init__(self) -> None:
        
        # the parametrization, d_parametrization, etc. 
        self.a = None
        self.da = None
        self.x = None
        self.y = None
        self.dx_da = None
        self.dy_da = None
        self.ddx_dda = None
        self.ddy_dda = None
        
        # the boundary velocity condition. 
        self.u = None
        self.v = None
        
    def build(self):
        pass

    def scale(self, scale):
        scale_x, scale_y = scale
        self.x = scale_x*self.x
        self.y = scale_y*self.y
        self.dx_da = scale_x*self.dx_da
        self.dy_da = scale_y*self.dy_da
        self.ddx_dda = scale_x*self.ddx_dda
        self.ddy_dda = scale_y*self.ddy_dda

    def rotate(self, theta):
        t = self.get_t()*np.exp(1j*theta)
        self.x = t.real
        self.y = t.imag

        dt = self.get_dt_da()*np.exp(1j*theta)
        self.dx_da = dt.real
        self.dy_da = dt.imag

        ddt = self.get_ddt_dda()*np.exp(1j*theta)
        self.ddx_dda = ddt.real
        self.ddy_dda = ddt.imag

    def shift(self, shift):
        shift_x, shift_y = shift
        self.x = self.x + shift_x
        self.y = self.y + shift_y

    def get_t(self):
        return self.x + 1j*self.y

    def get_dt_da(self):
        return self.dx_da + 1j*self.dy_da

    def get_ddt_dda(self):
        return self.ddx_dda + 1j*self.ddy_dda

    def get_k(self):
        return (self.dx_da*self.ddy_dda - self.dy_da*self.ddx_dda)/((self.dx_da**2+self.dy_da**2)**(1.5))

    def get_data(self):
        return self.a, self.da, self.get_t(), self.get_dt_da(), self.get_k()
    
    def get_h(self):
        return -self.v + 1j*self.u


class line(geometry):
    
    def __init__(self, p1, p2) -> None:
        
        '''
        this gives the line starts from p1 and ends at p2.
        '''
        
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        
        
    def build(self, max_distance=1e-2):
        
        n = 8
        cond = True
        
        while cond:
            a, da = gauss_quad_nodes(n)
            self.a = a
            self.da = da
            self.x = self.p1[0] + (self.p2[0]-self.p1[0])*((a+1)/2)
            self.y = self.p1[1] + (self.p2[1]-self.p1[1])*((a+1)/2)
            if np.max(np.abs(np.diff(self.get_t()))) < max_distance:
                cond = False
            else:
                n *= 2
        
        self.dx_da = (self.p2[0]-self.p1[0])/2 * np.ones(n)
        self.dy_da = (self.p2[1]-self.p1[1])/2 * np.ones(n)
        self.ddx_dda = np.zeros(n)
        self.ddy_dda = np.zeros(n)

        # because in our geometry, only caps represent the inflow and outflow, which might
        # have non-zero velocity.
        
        self.u = np.zeros(n)
        self.v = np.zeros(n)
        

class cap(geometry):

    def __init__(self, p1=(1,0),p2=(-1,0), max_distance=None) -> None:
        
        '''
        by default, it generates a smooth cap with n points in the counterclockwise direction 
        that goes through points 
            p1 = (1,0)
            p* = (0,1), 
            p2 = (-1,0). 
        Looks like a semi-circle.
        
        Given specified value of p1, p2, this will deduce the p* accordingly. and draw a cap
        that starts at p1 and ends at p2, going in the counterclockwise direction. 
        '''
        
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        
    def build(self, max_distance=1e-2):
        
        n = 8
        cond = True
        scale = np.linalg.norm(self.p1 - self.p2)/2
        
        while cond:

            a, da = gauss_quad_nodes(n)
            self.a = a
            self.da = da
            
            self.x = -8*_Psi(a)
            
            b = _int_Psi_normalizer
            self.y = -b*_int_Psi(a)

            max_distance_ = np.max(np.abs(np.diff(self.get_t())))
            
            if max_distance_*scale < max_distance:
                cond = False
            else:
                n *= 2
            
        self.dx_da = -8*_psi(a)
        self.dy_da = -b*_Psi(a)

        self.ddx_dda = -8*_d_psi(a)
        self.ddy_dda = -b*_psi(a)
        
        self.scale((-scale, -scale))
        
        # now it is the time to set the unit flux condition. 
        
        h = (scale**2 - self.x**2)*3/(4*scale**3)
        
        angle = np.angle(self.p2[0] + 1j*self.p2[1] - self.p1[0] - 1j*self.p1[1])
        
        self.h = h*np.exp(1j*angle)
        
        self.u = self.h.imag
        self.v = -self.h.real
        
        self.rotate(angle)        
        self.shift((self.p1 + self.p2)/2)


class corner(geometry):
    def __init__(self, p, q, r, n=64):
        
        '''
        the points p, q, r on the R^2 describe a corner that is given by line p-q and then q-r. q is the point of intersection. 
        '''
        
        self.p = np.array(p)
        self.q = np.array(q)
        self.r = np.array(r)

        assert(np.linalg.norm(p-q) > 0)
        assert(np.linalg.norm(p-q) == np.linalg.norm(q-r))

    def build(self, max_distance=1e-2):
        
        p = self.p
        q = self.q
        r = self.r
        
        # constructing the transformation matrix. 

        standard_p = np.array([-1, 1])
        standard_q = np.array([0, 0])
        standard_r = np.array([1, 1])

        standard_v1 = standard_q - standard_p
        standard_v2 = standard_r - standard_q

        v1 = q-p
        v2 = r-q

        
        # We need to find a transformation that maps standard_vi to vi.
        standard_v_mat = np.array([[standard_v1[0], standard_v2[0]], [
                                  standard_v1[1], standard_v2[1]]])
        v_mat = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
        
        A = np.matmul(v_mat, np.linalg.inv(standard_v_mat))
        # A * standard_v_mat = v_mat
        
        
        # building the geometry now. 
        
        n = 8
        cond = True
        
        while cond:
            a, da = gauss_quad_nodes(n)
            self.x = a.copy()
            self.y = np.array([convoluted_abs(x_) for x_ in self.x])
            self.x, self.y = np.matmul(A, np.array([self.x, self.y]))
            if np.max(np.abs(np.diff(self.get_t()))) < max_distance:
                cond = False
            else:
                n *= 2

        self.dx_da = np.ones(self.a.shape)
        self.dy_da = np.array([d_convoluted_abs(x_) for x_ in self.x])
        self.ddx_dda = np.zeros(self.a.shape)
        self.ddy_dda = np.array([dd_convoluted_abs(x_) for x_ in self.x])

        self.dx_da, self.dy_da = np.matmul(
            A, np.array([self.dx_da, self.dy_da]))
        self.ddx_dda, self.ddy_dda = np.matmul(
            A, np.array([self.ddx_dda, self.ddy_dda]))
        
        # Now, we need to shift the corner to the correct position.
        self.shift(q)
        
        self.u = np.zeros(self.x.shape)
        self.v = np.zeros(self.x.shape)