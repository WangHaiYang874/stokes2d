from utils import *
from .curve import Curve

from scipy.integrate import quad, IntegrationWarning
from scipy.special import expi

import numbers
import warnings

warnings.simplefilter("ignore", IntegrationWarning)


class Corner(Curve):
    standard_start_pt = pt(-1, 1)
    standard_mid_pt = pt(0, 0)
    standard_end_pt = pt(1, 1)
    def x_fn(_, a): return a
    def y_fn(_, a): return np.array([convoluted_abs(x) for x in a])
    def dx_da_fn(_, a): return np.ones_like(a)
    def dy_da_fn(_, a): return np.array([d_convoluted_abs(x) for x in a])
    def ddx_dda_fn(_, a): return np.zeros_like(a)
    def ddy_dda_fn(_, a): return np.array([dd_convoluted_abs(x) for x in a])

    def __init__(self, start_pt=pt(-1, 1), end_pt=pt(1, 1), mid_pt=pt(0, 0)):

        assert (np.linalg.norm(start_pt - mid_pt) > 0)
        assert (np.abs(np.linalg.norm(start_pt - mid_pt) -
                    np.linalg.norm(end_pt - mid_pt)) < 1e-14)
            

        super(Corner, self).__init__(start_pt, end_pt, mid_pt)


def convoluted_abs(x):
    # see https://www.wolframalpha.com/input?i=%5Cint+exp%281%2Fx%29+dx

    if np.abs(x) >= 1:
        return np.abs(x)
    x = np.abs(x)

    a = 2*x*quad(bump,0, x,epsabs=ERR, epsrel=ERR)[0]
    t = (x**2-1)
    b = (expi(1/t) - t*np.exp(1/t))/_bump_def_int
    
    return a + b

def d_convoluted_abs(x):
    sign = np.sign(x)
    x = np.abs(x)
    if x >= 1:
        return sign
    return 2*sign*quad(bump, 0, x, epsabs=ERR, epsrel=ERR)[0]

def dd_convoluted_abs(x):
    if np.abs(x) >= 1:
        return 0
    return 2*bump(x)

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

_bump_def_int = 0.443993816168079437823048921170552663761201789045697497307484553947040996939333945294846408031424708284300177578991350408812926995038438567308381450581

def bump(a):
    return _bump(a)/_bump_def_int