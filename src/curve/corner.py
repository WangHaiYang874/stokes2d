from utils import *
from .curve import Curve

from scipy.integrate import quad, IntegrationWarning

import numbers
import warnings

warnings.simplefilter("ignore",IntegrationWarning)

class Corner(Curve):
    standard_start_pt = pt(-1, 1)
    standard_mid_pt = pt(0, 0)
    standard_end_pt = pt(1, 1)
    x_fn = lambda _, a: a
    y_fn = lambda _, a: np.array([convoluted_abs(x) for x in a])
    dx_da_fn = lambda _, a: np.ones_like(a)
    dy_da_fn = lambda _, a: np.array([d_convoluted_abs(x) for x in a])
    ddx_dda_fn = lambda _, a: np.zeros_like(a)
    ddy_dda_fn = lambda _, a: np.array([dd_convoluted_abs(x) for x in a])

    def __init__(self, start_pt=pt(-1,1), end_pt=pt(1,1), mid_pt=pt(0,0)):

        assert (np.linalg.norm(start_pt - mid_pt) > 0)
        assert (np.abs(np.linalg.norm(start_pt - mid_pt) - np.linalg.norm(end_pt - mid_pt)) < 1e-15)
        
        super(Corner, self).__init__(start_pt, end_pt, mid_pt)

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
