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

        # some other important variables.
        self.out_normal_direction = None
        self.p = None
        self.p1 = None
        self.p2 = None

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

    def get_velocity(self, flux=0):
        pass


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
        assert(np.linalg.norm(p1-p_) == np.linalg.norm(p2-p_))

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
