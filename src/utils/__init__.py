import numpy as np
import numbers
from typing import List, Tuple, Dict
from .callback import Callback


ERR = 1e-17 # the quadrature error setted for scipy.integrate.quad
THRESHOLD = 1e-10 # for matcthing let. 
FMM_EPS = 5e-16

REQUIRED_TOL = 1e-8
DOMAIN_THRESHOLD = 1e-6 # The minimal size of interval of 16 pts gauss quad rule
MAX_DISTANCE = 0.08 # panel.good_enough test
GMRES_TOL = 1E-10
GMRES_MAX_ITER = 4
RESTART = 500
DENSITY = 50

# TODO: change this 
def gauss_quad_rule(n=16, domain=(-1, 1)):
    assert n > 0
    a, da = np.polynomial.legendre.leggauss(n)
    if domain == (-1, 1):
        return a, da
    left, right = domain
    a = ((right - left) * a + (right + left)) / 2
    da = da * (right - left) / 2
    return a, da


def line_intersect(p1, p2, p3, p4):
    """
    p5 is the intersection of line p1-p2 and line p3-p4. Therefore
        p5 = p1 + t1*(p2-p1) = p3 + t2*(p4-p3)
        p1 - p3 = t1*(p1-p2) + t2*(p4-p3)
    we can use this to solve for (t1,t2)
    """

    t = np.linalg.solve(np.array([p1-p2, p4-p3]).T, p1-p3)
    t1, _ = t
    return p1 + t1*(p2-p1)


def pt(x, y):
    return np.array((x, y))

def pt2cplx(pt):
    return pt[0] + 1j * pt[1]
def cplx2pt(cplx):
    return np.array([cplx.real, cplx.imag])

def U2H(U):
    if U.shape == (2,):
        U = U.reshape((1, 2))
    return -U[:, 1] + 1j * U[:, 0]


def H2U(H):
    if isinstance(H, numbers.Number):
        return np.array([H.real, -H.imag]).reshape((1, 2))
    return np.array([H.imag, -H.real]).T
