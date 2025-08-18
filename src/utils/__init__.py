import numpy as np
import numbers
from typing import List, Tuple, Dict
from .callback import Callback


ERR = 1e-17 # the quadrature error setted for scipy.integrate.quad
THRESHOLD = 1e-10 # for matcthing let. 
FMM_EPS = 5e-14

REQUIRED_TOL = 1e-8
DOMAIN_THRESHOLD = 1e-6 # The minimal size of interval of 16 pts gauss quad rule
MAX_DISTANCE = 0.08 # panel.good_enough test
GMRES_TOL = 1E-10
GMRES_MAX_ITER = 4
RESTART = 500
DENSITY = 50

# TODO: change this 
def gauss_quad_rule(n=16, domain=(-1, 1)):
    """Generate Gaussian quadrature points and weights for numerical integration.
    
    Args:
        n: Number of quadrature points (default 16)
        domain: Integration interval (default (-1, 1))
    
    Returns:
        a: Quadrature points 
        da: Quadrature weights
    """
    assert n > 0
    # Get standard Legendre-Gauss quadrature on [-1, 1]
    a, da = np.polynomial.legendre.leggauss(n)
    if domain == (-1, 1):
        return a, da
    # Transform to arbitrary interval [left, right]
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
    """Create 2D point as numpy array."""
    return np.array((x, y))

def pt2cplx(pt):
    """Convert 2D point to complex number."""
    return pt[0] + 1j * pt[1]

def cplx2pt(cplx):
    """Convert complex number to 2D point."""
    return np.array([cplx.real, cplx.imag])

def U2H(U):
    """Convert velocity field (u,v) to complex representation H = v + iu."""
    if U.shape == (2,):
        U = U.reshape((1, 2))
    return -U[:, 1] + 1j * U[:, 0]

def H2U(H):
    """Convert complex field H to velocity components (u,v)."""
    if isinstance(H, numbers.Number):
        return np.array([H.real, -H.imag]).reshape((1, 2))
    return np.array([H.imag, -H.real]).T

def rotation_matrix_2d(theta):
    """Create 2D rotation matrix for counter-clockwise rotation by angle theta.
    
    Args:
        theta: Rotation angle in radians [0, 2π)
    
    Returns:
        2x2 rotation matrix
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta) 
    return np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])

def rotate_point_2d(point, theta):
    """Rotate a 2D point counter-clockwise by angle theta.
    
    Args:
        point: 2D point as numpy array [x, y]
        theta: Rotation angle in radians
        
    Returns:
        Rotated 2D point
    """
    if theta == 0:
        return point.copy()
    R = rotation_matrix_2d(theta)
    return R @ point

def rotate_vector_field_2d(u, v, theta):
    """Rotate velocity field components counter-clockwise by angle theta.
    
    Args:
        u, v: Velocity field components (same shape)
        theta: Rotation angle in radians
        
    Returns:
        u_rot, v_rot: Rotated velocity components
    """
    if theta == 0:
        return u.copy(), v.copy()
        
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Apply rotation transformation: [u_rot, v_rot] = R @ [u, v]
    u_rot = cos_theta * u - sin_theta * v
    v_rot = sin_theta * u + cos_theta * v
    
    return u_rot, v_rot
