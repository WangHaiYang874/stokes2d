import numpy as np
import numbers

'''
This document sets some standard for the representation of the basic variables. 
'''

'''
velocity has two forms:
    1. as a numpy array U of shape (n,2). where u = U[:,0] is the x-component and v = V[:,1] is the y-component.
    2. as a complex numpy array H of shape (n,). where u = H.imag and v = -H.real. 
'''


def U2H(U):
    if U.shape == (2,):
        U = U.reshape((1, 2))
    return -U[:, 1] + 1j * U[:, 0]


def H2U(H):
    if isinstance(H, numbers.Number):
        return np.array([H.real, -H.imag]).reshape((1, 2))
    return np.array([H.imag, -H.real]).T


'''
gradient of pressure has only one form: np array with shape (n,2)
'''

'''
TODO: I should also set the tolerance for quadrature rules here. 
'''


def pt(x, y):
    return np.array((x, y))


def gauss_quad_rule(n=16, domain=(-1, 1)):
    assert n > 0
    a, da = np.polynomial.legendre.leggauss(n)
    if domain == (-1, 1):
        return a, da
    left, right = domain
    a = ((right - left) * a + (right + left)) / 2
    da = da * (right - left) / 2
    return a, da


def distance(pts, a, b):
    """
    Distance of the pts to the line segment ab.
    """

    if np.all(a == b):
        return np.linalg.norm(pts - a, axis=1)

    ba_n = (b - a)/np.linalg.norm(b - a)

    # signed parallel distance components
    sa = np.dot(a - pts, ba_n)
    sb = np.dot(pts - b, ba_n)

    # clamped parallel distance
    h = np.maximum.reduce([sa, sb, np.zeros(len(pts))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(pts - a, ba_n)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)
