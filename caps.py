import numpy as np
import scipy as sp
from scipy.special import p_roots as gauss_quad_nodes
from scipy.integrate import fixed_quad as quad 

class smooth_cap:

    def __init__(self,n=128,scale=None,rotate=None,shift=None):
        '''
        n: number of points
        scale: (scale_x,scale_y)
        rotate: theta \in [0,2pi]
        shift: (shift_x,shift_y)

        by default, it generates a smooth cap with n points in the counterclockwise direction 
        that goes through points (1,0), (0,1), (-1,0). Looks like a semi-circle. 
        '''
        a,da = gauss_quad_nodes(n)
        self.a = a
        self.da = da
        self.x = -8*_Psi(a)
        b = 1/quad(_Psi,0,1,n=128)[0]
        self.y = -b*_int_Psi(a)
        
        self.dx_da = -8*_psi(a)
        self.dy_da = b*self.x/16

        self.ddx_dda = -8*_d_psi(a)
        self.ddy_dda = b*self.dx_da/16

        if scale is not None:
            self.scale(scale)
        if rotate is not None:
            self.rotate(rotate)
        if shift is not None:
            self.shift(shift)
    
    def scale(self,scale):
        scale_x, scale_y = scale
        self.x = scale_x*self.x
        self.y = scale_y*self.y
        self.dx_da = scale_x*self.dx_da
        self.dy_da = scale_y*self.dy_da
        self.ddx_dda = scale_x*self.ddx_dda
        self.ddy_dda = scale_y*self.ddy_dda
    
    def rotate(self,theta):
        t = self.get_t()*np.exp(1j*theta)
        self.x = t.real
        self.y = t.imag
        
        dt = self.get_dt_da()*np.exp(1j*theta)
        self.dx_da = dt.real
        self.dy_da = dt.imag

        ddt = (self.ddx_dda + 1j*self.ddy_dda)*np.exp(1j*theta)
        self.ddx_dda = ddt.real
        self.ddy_dda = ddt.imag

    def shift(self,shift):
        shift_x,shift_y = shift
        self.x = self.x + shift_x
        self.y = self.y + shift_y
    
    def get_t(self):
        return self.x + 1j*self.y

    def get_dt_da(self):
        '''dt/da'''
        return self.dx_da + 1j*self.dy_da

    def get_k(self):
        return (self.dx_da*self.ddy_dda - self.dy_da*self.ddx_dda)/((self.dx_da**2+self.dy_da**2)**(1.5))

    def get_data(self):
        return self.a, self.da, self.get_t(), self.get_dt_da(), self.get_k()


def _psi(a:np.ndarray) -> np.ndarray:

    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        ret = ((a**2+1)*np.exp(4*a/(a**2-1)))/((a**2-1)*(1+np.exp(4*a/(a**2-1))))**2

    ret[abs(a)>=1] = 0
    ret[np.isnan(ret)] = 0
    ret[ret==np.inf] = 0
    return ret

def _Psi(a:np.ndarray) -> np.ndarray:

    with np.errstate(divide='ignore',over='ignore'):
        ret = -np.tanh(-2*a/(1-a**2))/8
    ret[a>=1] = 1/8
    ret[a<=-1] = -1/8
    nan_mask = np.isnan(ret)
    if np.any(nan_mask):
        ret[nan_mask & a>0.9] = 1/8
        ret[nan_mask & a<-0.9] = -1/8
    return ret

    
def _int_Psi(a:np.ndarray) -> np.ndarray:
    '''the number 128 in the expression below is chosen by numerical experiment'''
    return np.array([quad(_Psi,-1,b,n=128)[0] for b in a])

def _d_psi(a:np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        e = np.exp(4*a/(a**2-1))
        ret = (-2 *e* (2 - 3*a + 4*a**2 + 2*a**3 + 2*a**4 + a**5 + e*(-2 - 3*a - 4*a**2 + 2* a**3 - 2*a**4 + a**5)))/((1 + e)**3 * (-1 + a**2)**4)
    ret[np.isnan(ret)] = 0
    return ret