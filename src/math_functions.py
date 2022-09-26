import numbers
from scipy.integrate import quad, IntegrationWarning
import numpy as np

__all__ = ['_psi', '_Psi', '_int_Psi_normalized', '_d_psi', '_Psi_normalized','_psi_normalized',
           '_bump', '_d_bump', '_d_d_bump',
           'convoluted_abs', 'd_convoluted_abs', 'dd_convoluted_abs',]

ERR = 1e-17

import warnings
warnings.simplefilter("ignore",IntegrationWarning)

def _psi(a: np.ndarray) -> np.ndarray:

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = ((a**2+1)*np.exp(4*a/(a**2-1))) / \
            ((a**2-1)*(1+np.exp(4*a/(a**2-1))))**2

    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    ret[ret == np.inf] = 0

    return -8*ret

def _d_psi(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        e = np.exp(4*a/(a**2-1))
        ret = (-2 * e * (2 - 3*a + 4*a**2 + 2*a**3 + 2*a**4 + a**5 + e*(-2 - 3 *
               a - 4*a**2 + 2 * a**3 - 2*a**4 + a**5)))/((1 + e)**3 * (-1 + a**2)**4)
    ret[np.isnan(ret)] = 0
    return -8*ret

def _Psi(a):

    if isinstance(a, np.ndarray):    
        with np.errstate(divide='ignore', over='ignore'):
            ret = np.tanh(-2*a/(1-a**2))
            
        ret[a >= 1] = -1
        ret[a <= -1] = 1
        nan_mask = np.isnan(ret)
        if np.any(nan_mask):
            ret[nan_mask & a > 0.9] = -1
            ret[nan_mask & a < -0.9] = 1
        assert np.sum(np.isnan(ret)) == 0
        return ret

    if isinstance(a,numbers.Number):
        if a >= 1: return -1
        if a <= -1:return 1
        return np.tanh(-2*a/(1-a**2))
    
_int_Psi_normalizer = -quad(_Psi,0,1,epsabs=ERR,epsrel=ERR)[0]/2

def _int_Psi_normalized(a):
    if isinstance(a, np.ndarray):
        return np.array([quad(_Psi, -1, b, epsabs=ERR,epsrel=ERR)[0] for b in a])/_int_Psi_normalizer
    if isinstance(a, numbers.Number):
        return quad(_Psi, -1, a, epsabs=ERR,epsrel=ERR)[0]/_int_Psi_normalizer
    raise ValueError

def _Psi_normalized(a): return _Psi(a)/_int_Psi_normalizer

def _psi_normalized(a): return _psi(a)/_int_Psi_normalizer


def _bump(a):
    if isinstance(a, numbers.Number):
        if np.abs(a) >= 1:
            return 0
        return np.exp(1/(a**2-1))

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = np.exp(1/(a**2-1))
    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    return ret


def _d_bump(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = -2 * _bump(a) * a / (a**2-1)**2
    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    return ret


def _d_d_bump(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = 2 * _bump(a) * (3*a**4 - 1) / (a**2-1)**4
    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    return ret


bump_def_int = quad(_bump, -1, 1, epsabs=ERR, epsrel=ERR)[0]

# def convoluted_abs(x):
#     if np.abs(x) >= 1:
#         return np.abs(x)
#
#     def b(y): return _bump(y)*np.abs(x-y)
#     return quad(b, -1, 1, epsabs=ERR, epsrel=ERR, full_output=1)[0]/bump_def_int
# this function has too much numerical error. It is replaced with

def convoluted_abs(x):
    if np.abs(x) >= 1:
        return np.abs(x)
    x = np.abs(x)

    a = quad(_bump,
               0,x,
               epsabs=ERR, epsrel=ERR)[0]
    b = quad(lambda y: y*_bump(y),
               x,1,
               epsabs=ERR, epsrel=ERR)[0]
    return  2*(x*a + b)/bump_def_int


def d_convoluted_abs(x):
    if np.abs(x) >= 1:
        return np.sign(x)
    
    return (quad(_bump, -1, x, epsabs=ERR, epsrel=ERR)[0] - quad(_bump, x, 1, epsabs=ERR, epsrel=ERR)[0])/bump_def_int

def dd_convoluted_abs(x):
    if np.abs(x) >= 1:
        return 0
    return 2*_bump(x)/bump_def_int