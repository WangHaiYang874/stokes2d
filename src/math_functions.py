import numbers
from scipy.integrate import quad
import numpy as np

__all__ = ['_psi', '_Psi', '_int_Psi', '_d_psi', 
           '_bump', '_d_bump', '_d_d_bump', '_normalized_bump', 
           'convoluted_abs', 'd_convoluted_abs', 'dd_convoluted_abs', 
           '_int_Psi_normalizer']


def _psi(a: np.ndarray) -> np.ndarray:

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = ((a**2+1)*np.exp(4*a/(a**2-1))) / \
            ((a**2-1)*(1+np.exp(4*a/(a**2-1))))**2

    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    ret[ret == np.inf] = 0
    return ret


def _Psi(a):

    if isinstance(a, np.ndarray):    
        with np.errstate(divide='ignore', over='ignore'):
            ret = -np.tanh(-2*a/(1-a**2))/8
        ret[a >= 1] = 1/8
        ret[a <= -1] = -1/8
        nan_mask = np.isnan(ret)
        if np.any(nan_mask):
            ret[nan_mask & a > 0.9] = 1/8
            ret[nan_mask & a < -0.9] = -1/8
        return ret
    if isinstance(a,numbers.Number):
        if a >= 1:
            return 1/8
        if a <= -1:
            return -1/8
        return -np.tanh(-2*a/(1-a**2))/8
    

_int_Psi_normalizer = 1/quad(_Psi,0,1,epsabs=1e-14,epsrel=1e-14)[0]

def _int_Psi(a):
    '''the number 128 in the expression below is chosen by numerical experiment'''
    if isinstance(a, np.ndarray):
        return np.array([quad(_Psi, -1, b, epsabs=1e-14,epsrel=1e-14)[0] for b in a])
    if isinstance(a, numbers.Number):
        return quad(_Psi, -1, a, epsabs=1e-14,epsrel=1e-14)[0]
    else:
        print('input is not a number or a numpy array')
        assert 0
    


def _d_psi(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        e = np.exp(4*a/(a**2-1))
        ret = (-2 * e * (2 - 3*a + 4*a**2 + 2*a**3 + 2*a**4 + a**5 + e*(-2 - 3 *
               a - 4*a**2 + 2 * a**3 - 2*a**4 + a**5)))/((1 + e)**3 * (-1 + a**2)**4)
    ret[np.isnan(ret)] = 0
    return ret


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


bump_def_int = quad(_bump, -1, 1, epsabs=1e-14,epsrel=1e-14)[0]

def _normalized_bump(a):
    return _bump(a)/bump_def_int

def convoluted_abs(x):
    if np.abs(x) >= 1:
        return np.abs(x)

    def b(y): return _normalized_bump(y)*np.abs(x-y)
    return quad(b, -1, 1, epsabs=1e-15, epsrel=1e-15, full_output=1)[0]

def d_convoluted_abs(x):
    if np.abs(x) >= 1:
        return np.sign(x)

    def b(y): return _bump(y)
    return +quad(b, -1, x, epsabs=1e-15, epsrel=1e-15, full_output=1)[0]\
        - quad(b, x, 1, epsabs=1e-15, epsrel=1e-15, full_output=1)[0]

def dd_convoluted_abs(x):
    if np.abs(x) >= 1:
        return 0
    return 2*_normalized_bump(x)