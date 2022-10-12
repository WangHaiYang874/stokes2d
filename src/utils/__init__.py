import numpy as np
import numbers
from typing import List, Tuple, Dict


ERR = 1e-17 # the quadrature error setted for scipy.integrate.quad
THRESHOLD = 1e-10 # for matcthing let. 
DOMAIN_THRESHOLD = 1e-8 # The minimal size of interval of 16 pts gauss quad rule
MAX_DISTANCE = 1E-2 # panel.good_enough test
LEGENDRE_RATIO = 1e-14 # same as above
GMRES_TOL = 1E-12
RESTART = 100


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


def U2H(U):
    if U.shape == (2,):
        U = U.reshape((1, 2))
    return -U[:, 1] + 1j * U[:, 0]


def H2U(H):
    if isinstance(H, numbers.Number):
        return np.array([H.real, -H.imag]).reshape((1, 2))
    return np.array([H.imag, -H.real]).T
