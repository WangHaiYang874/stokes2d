from .curve import Curve
from utils import *
from scipy.integrate import quad, IntegrationWarning

import numbers
import warnings

warnings.simplefilter("ignore", IntegrationWarning)


class Cap(Curve):
    standard_start_pt = pt(1, 0)
    standard_mid_pt = pt(0, 2)
    standard_end_pt = pt(-1, 0)
    def x_fn(_, a): return _Psi(a)
    def dx_da_fn(_, a): return _psi(a)
    def ddx_dda_fn(_, a): return _d_psi(a)

    def y_fn(_, a): return int_Psi(a)
    def dy_da_fn(_, a): return Psi(a)
    def ddy_dda_fn(_, a): return psi(a)

    def __init__(self, start_pt=pt(1, 0), end_pt=pt(-1, 0), mid_pt=pt(0, 2)) -> None:

        super(Cap, self).__init__(start_pt, end_pt, mid_pt)

        # this provides conditions for matching
        self.matching_pt = (start_pt + end_pt) / 2
        out_vec = mid_pt - self.matching_pt
        self.dir = np.arctan2(out_vec[1], out_vec[0])
        self.diameter = np.linalg.norm(end_pt - start_pt)

        # TODO: delete these two assert sentences?
        assert np.linalg.norm(self.matching_pt - mid_pt) > 1e-15
        assert self.diameter > 1e-15

    def boundary_velocity(self):
        """
        tcap is the only thing that will be used as the inlet/outlet.
        this function returns the velocity condition of outward unit flux. 
        """

        r = self.diameter / 2

        t = (self.t - self.matching_pt[0] - self.matching_pt[1] * 1j)
        t = t * np.exp(-1j * (self.dir - np.pi / 2))
        x = t.real
        h = (x ** 2 - r ** 2) * 3 / (4 * r ** 3)
        h = np.exp(1j * (self.dir - np.pi / 2)) * h

        return h


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

    if isinstance(a, numbers.Number):
        if a >= 1:
            return -1
        if a <= -1:
            return 1
        return np.tanh(-2*a/(1-a**2))

_int_Psi = -quad(_Psi, 0, 1, epsabs=ERR, epsrel=ERR)[0]/2

def int_Psi(a):
    if isinstance(a, numbers.Number):
        return quad(_Psi, -1, a, epsabs=ERR, epsrel=ERR)[0]/_int_Psi
    return np.array([int_Psi(ai) for ai in a])

def Psi(a): return _Psi(a)/_int_Psi

def psi(a): return _psi(a)/_int_Psi