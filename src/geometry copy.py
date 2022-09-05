from tracemalloc import start
import numpy as np
from scipy.special import p_roots as gauss_quad_nodes
from scipy.integrate import quad
from math_functions import *
from basic_spec import *


def get_velocity(p1, p2, t, flux=0):
    '''
    notice that this works for a particular case. that is x has been rotated yet. 
    '''

    if flux == 0:
        return np.zeros((len(t), 2))

    vec = p2-p1
    angle = np.arctan2(vec[1], vec[0])
    length = np.linalg.norm(vec)
    r = length/2
    x = ((t - p1[0] - p1[1]*1j)*np.exp(-1j*angle)).real
    x = x - r
    H = (x**2 - r**2)*3/(4*r**3)
    H = flux*np.exp(1j*angle)*H
    return H2U(H)

class panel:
    '''
    each panel is designed by 
        - a 16 points gauss quadrature rule, 
        - have a max distance of 1.     
    '''
    def __init__(self) -> None:
        self.a = None
        self.da = None
        self.x = None
        self.y = None
        self.dx_da = None
        self.dy_da = None
        self.ddx_dda = None
        self.ddy_dda = None

class curve:
    def __init__(self,start_pt,mid_pt,end_pt) -> None:

        # the parametrization, d_parametrization, etc. 

        self.panels = []
        self.start_pt = start_pt
        self.mid_pt = mid_pt
        self.end_pt = end_pt

        # some other important variables.
        out_direction = mid_pt - (start_pt + end_pt)/2
        if np.linalg.norm(out_direction) < 1e-10:
            self.out_normal_direction = None
        else:
            self.out_normal_direction = out_direction/np.linalg.norm(out_direction)
        

    @property
    def a(self):
        return np.concatenate([p.a + 2*i for i,p in enumerate(self.panels)])
    
    @property
    def da(self):
        return np.concatenate([p.da for p in self.panels])

    @property
    def x(self):
        return np.concatenate([p.x for p in self.panels])

    @property
    def y(self):
        return np.concatenate([p.y for p in self.panels])

    @property
    def t(self):
        return self.x + 1j*self.y

    @property
    def dx_da(self):
        return np.concatenate([p.dx_da for p in self.panels])

    @property
    def dy_da(self):
        return np.concatenate([p.dy_da for p in self.panels])
    
    @property
    def dt_da(self):
        return self.dx_da + 1j*self.dy_da

    @property
    def ddx_dda(self):
        return np.concatenate([p.ddx_dda for p in self.panels])

    @property
    def ddy_dda(self):
        return np.concatenate([p.ddy_dda for p in self.panels])

    @property
    def get_k(self):
        return (self.dx_da*self.ddy_dda - self.dy_da*self.ddx_dda)/((self.dx_da**2+self.dy_da**2)**(1.5))

    @property
    def Poiseuille_velocity_condition(self, flux=0):
        pass

    def build_transformation(self):
        '''this is for a template function for each subclass. '''
        pass

    def build(self):
        pass


class line(curve):

    def __init__(self, start_point,) -> None:
        '''
        this gives the line starts from p1 and ends at p2.
        '''

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

        self.p = (self.p1 + self.p2)/2
        self.out_normal_direction = np.arctan2(
            self.p2[1]-self.p1[1], self.p2[0]-self.p1[0]) + np.pi/2
        
    def get_velocity(self, flux=0):
        if flux == 0:
            return np.zeros((len(self.a), 2))
        else:
            return get_velocity(self.p1, self.p2, self.get_t(), flux)


class cap(geometry):

    def __init__(self, p1=(1, 0), p2=(-1, 0), max_distance=None) -> None:
        '''
        by default, it generates a smooth cap with n points in the counterclockwise direction 
        that goes through points 
            p1 = (1,0)
            p* = (0,2), 
            p2 = (-1,0). 
        Looks like a semi-circle.

        Given specified value of p1, p2, this will deduce the p* accordingly. and draw a cap
        that starts at p1 and ends at p2, going in the counterclockwise direction. 
        '''

        self.p1 = np.array(p1)
        self.p2 = np.array(p2)

    def build(self, max_distance=1e-2):

        n = 8
        scale = np.linalg.norm(self.p1 - self.p2)/2

        cond = True
        while cond:

            a, da = gauss_quad_nodes(n)
            self.a = a
            self.da = da

            self.x = -8*_Psi(a)

            b = _int_Psi_normalizer
            self.y = -2*b*_int_Psi(a)

            max_distance_ = np.max(np.abs(np.diff(self.get_t())))

            if max_distance_*scale < max_distance:
                cond = False
            else:
                n *= 2

        self.dx_da = -8*_psi(a)
        self.dy_da = -2*b*_Psi(a)
        self.ddx_dda = -8*_d_psi(a)
        self.ddy_dda = -2*b*_psi(a)

        # minus sign makes it going counterclockwise.
        self.scale((-scale, -scale))
        angle = np.arctan2(self.p2[1]-self.p1[1], self.p2[0]-self.p1[0])
        self.rotate(angle)
        self.shift((self.p1 + self.p2)/2)

        self.out_normal_direction = angle + np.pi/2
        self.p = (self.p1 + self.p2)/2

    def get_velocity(self, flux=1):
        if flux == 0:
            return np.zeros((len(self.a), 2))
        else:
            return get_velocity(self.p1, self.p2, self.get_t(), flux)


class corner(geometry):
    def __init__(self, p1, p_, p2):
        '''
        the points p1, p_, p2 on the R^2 describe a corner that is given by line p1-p_ and then p_-p2. p_ is the point of intersection. 
        '''

        self.p1 = np.array(p1)
        self.p_ = np.array(p_)
        self.p2 = np.array(p2)

        assert(np.linalg.norm(p1-p_) > 0)
        assert(np.abs(np.linalg.norm(p1-p_) - np.linalg.norm(p2-p_))<1e-15)
        # this should be zero, but there is some error of machine precision. 

    def build(self, max_distance=1e-2):

        p = self.p1
        q = self.p_
        r = self.p2

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

        n = 32
        max_distance_cond = True
        legendre_cond = True

        while max_distance_cond or legendre_cond:

            # generating the standard geometry.
            a, da = gauss_quad_nodes(n)
            self.a = a
            self.da = da

            self.x = a.copy()
            self.y = np.array([convoluted_abs(x_) for x_ in self.x])
            self.dx_da = np.ones(self.a.shape)
            self.dy_da = np.array([d_convoluted_abs(x_) for x_ in self.x])
            self.ddx_dda = np.zeros(self.a.shape)
            self.ddy_dda = np.array([dd_convoluted_abs(x_) for x_ in self.x])

            # do the transformation.
            self.x, self.y = np.matmul(A, np.array([self.x, self.y]))
            self.dx_da, self.dy_da = np.matmul(
                A, np.array([self.dx_da, self.dy_da]))
            self.ddx_dda, self.ddy_dda = np.matmul(
                A, np.array([self.ddx_dda, self.ddy_dda]))

            if np.max(np.abs(np.diff(self.get_t()))) < max_distance:
                max_distance_cond = False

            legendre_coef_t = np.polynomial.legendre.Legendre.fit(
                self.a, self.get_t(), deg=len(self.a)-1, domain=[-1, 1]).coef

            legendre_coef_k = np.polynomial.legendre.Legendre.fit(
                self.a, self.get_k(), deg=len(self.a)-1, domain=[-1, 1]).coef

            # m = len(legendre_coef_t)//10
            m = 5

            error1 = np.sum(np.abs(legendre_coef_t[-m:]))/np.sum(np.abs(legendre_coef_t[:m]))
            error2 = np.sum(np.abs(legendre_coef_k[-m:]))/np.sum(np.abs(legendre_coef_k[:m]))
            
            # print(self.a.shape, error1, error2,m)

            legendre_cond = max(error1, error2) > 1e-11
            
            n *= 2

        # Now, we need to shift the corner to the correct position.
        self.shift(q)

        self.p = (self.p1 + self.p2)/2
        self.out_normal_direction = np.arctan2(r[1]-p[1], r[0]-p[0]) + np.pi/2

    def get_velocity(self, flux=0):
        assert(flux == 0)
        return np.zeros((len(self.a), 2))
