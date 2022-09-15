"""
from the Curve, we have figured out how to draw stuff:
the lines, obstructions, and caps, smoothed corners, etc.
those are going to be just a Panel for a tube.

In this file, I'll write the codes to automate the process of
drawing tubes from the codes of the Curve. There are certain
things I wish this file can do.

1. a tube should be a sampler, instead of fixing number of points on
the Panel, we should be able to sample points on the tube to make sure
our stokes2d solver can have given tolerance and distance away from the
boundary, as the main numerical error appears only near the boundary. In
my previous experiements, 5h rule works out pretty well for controlling the
numerical error.

2. I also want to write a codes for several generic tubes with different specifications
such as the radius, the length, the number of bifurcations, how large should the smoothed
Corner be, etc.
"""

import sys
import warnings

from joblib import Parallel, delayed
from matplotlib.path import Path

sys.path.append('.')
from curve import *
from utility_and_spec import *
from scipy.sparse.linalg import gmres, LinearOperator


# import numpy as np
# import fmm2dpy as fmm


class Pipe:
    def __init__(self) -> None:

        # geometric data.
        self.curves: [Curve] = None

        # solver data
        self.K1 = None
        self.K2 = None
        self.A: LinearOperator = None
        self.omegas = None
        self.pressure_drops = None
        # self.velocity_field = None
        # self.pressure = None
        # self.vorticity = None

        # drawing data
        self.grid = None


    @property
    def a(self):
        return np.concatenate([c.a + 2 * i for i, c in enumerate(self.curves)])

    @property
    def da(self):
        return np.concatenate([c.da for c in self.curves])

    @property
    def x(self):
        return np.concatenate([c.x for c in self.curves])

    @property
    def y(self):
        return np.concatenate([c.y for c in self.curves])

    @property
    def dx_da(self):
        return np.concatenate([c.dx_da for c in self.curves])

    @property
    def dy_da(self):
        return np.concatenate([c.dy_da for c in self.curves])

    @property
    def ddx_dda(self):
        return np.concatenate([c.ddx_dda for c in self.curves])

    @property
    def ddy_dda(self):
        return np.concatenate([c.ddy_dda for c in self.curves])

    @property
    def t(self):
        return np.concatenate([c.t for c in self.curves])

    @property
    def dt_da(self):
        return np.concatenate([c.dt_da for c in self.curves])

    @property
    def k(self):
        return np.concatenate([c.k for c in self.curves])

    @property
    def lets(self):
        return [i for i, c in enumerate(self.curves) if isinstance(c, Cap)]

    @property
    def inlet(self):
        return self.lets[0]

    @property
    def outlets(self):
        return self.lets[1:]

    @property
    def flows(self):
        return [(self.inlet, o) for o in self.outlets]

    @property
    def polygon(self):
        return np.concatenate([c.contour_polygon() for c in self.curves])

    @property
    def extent(self):
        """
        (left, right, bottom, top)
        """

        x = self.polygon[:, 0]
        y = self.polygon[:, 1]
        return min(x), max(x), min(y), max(y)

    def build_geometry(self, max_distance=None, legendre_ratio=None):
        """
        default max distance = 1e-2
        default legendre coef = 1e-14"""
        self.curves = [c.build(max_distance, legendre_ratio) for c in self.curves]

    def build_kernel(self):

        da = self.da
        t = self.t
        dt_da = self.dt_da
        k = self.k

        dt = t[:, np.newaxis] - t[np.newaxis, :]
        d = dt_da[np.newaxis, :]
        da_ = da[np.newaxis, :]

        # this ignores the error for computing the diagonal elements with 0/0 error
        with np.errstate(divide='ignore', invalid='ignore'):
            K1 = -da_ * np.imag(d / dt) / np.pi
            K2 = -da_ * (-d / np.conjugate(dt) + np.conjugate(d)
                         * dt / (np.conjugate(dt ** 2))) / (2j * np.pi)

        # now we need to fill the diagonal elements
        d = dt_da
        K1_diagonal = k * np.abs(d) * da / (2 * np.pi)
        K2_diagonal = -da * k * (d ** 2) / (np.abs(d) * 2 * np.pi)
        np.fill_diagonal(K1, K1_diagonal)
        np.fill_diagonal(K2, K2_diagonal)

        self.K1 = K1
        self.K2 = K2

        def A(omega_sep):
            assert len(omega_sep) % 2 == 0
            n = len(omega_sep) // 2
            omega = omega_sep[:n] + 1j * omega_sep[n:]
            h = omega + K1 @ omega + K2 @ omega
            h_sep = np.concatenate([h.real, h.imag])
            return h_sep

        n = len(self.a)
        self.A = LinearOperator(dtype=np.float64, shape=(2 * n, 2 * n), matvec=A)

    def compute_omega(self, u, tol=1e-12):
        h = H2U(u)
        b = np.concatenate((h.real, h.imag))

        omega, _ = gmres(self.A, b, atol=0, tol=tol)

        if _ < 0:
            warnings.warn("gmres is not converging to tolerance. ")

        return omega

    def get_boundary_velocity_condition(self, i):
        inlet, outlet = self.flows[i]
        ret = []
        for j, c in enumerate(self.curves):
            if j == inlet:
                ret.append(-c.boundary_velocity())
            if j == outlet:
                ret.append(c.boundary_velocity())
            else:
                ret.append(np.zeros_like(c.a))
        return np.concatenate(ret)

    def build_omega(self, tol=None, n_jobs=1):

        tol = 1e-12 if tol is None else tol
        compute_omega = lambda i: self.compute_omega(self.get_boundary_velocity_condition(i), tol)
        self.omegas = np.array(Parallel(n_jobs=n_jobs)(
            delayed(compute_omega)(i)
            for i in range(len(self.flows))))

    def build_pressure_drops(self):
        pts = np.array([self.curves[i].matching_pt for i in self.lets])
        x = pts[:, 0]
        y = pts[:, 1]

        pressure_drops = []

        for omega in self.omegas:
            pressure = self.compute_pressure_and_vorticity(self, x, y, omega)[0]
            pressure_drop = pressure[1:] - pressure[0]
            pressure_drops.append(pressure_drop)

        self.pressure_drops = np.array(pressure_drops).T

    def compute_velocity(self, x, y, omega):
        """
        x,y should be 1d np array with dtype float.
        """
        z = x + 1j * y
        t = self.t
        dt = self.dt_da * self.da
        t_minus_z = t[np.newaxis, :] - z[:, np.newaxis]
        t_minus_z_sq = t_minus_z ** 2

        phi = np.sum((omega * dt)[np.newaxis, :] /
                     t_minus_z, axis=1) / (2j * np.pi)
        d_phi = np.sum((omega * dt)[np.newaxis, :] /
                       t_minus_z_sq, axis=1) / (2j * np.pi)

        psi = (1 / (2j * np.pi)) * (
                2 * np.sum(np.real((np.conjugate(omega) * dt)
                                   [np.newaxis, :]) / t_minus_z, axis=1)
                - np.sum((np.conjugate(t) * omega * dt)[np.newaxis, :] / t_minus_z_sq, axis=1))

        ret = phi + z * np.conjugate(d_phi) + np.conjugate(psi)
        return H2U(ret)

    def compute_pressure_and_vorticity(self, x, y, omega):
        z = x + 1j * y
        assert (isinstance(z, np.ndarray))

        t = self.get_t()
        dt = self.get_dt_da() * self.da

        t_minus_z_sq = (t[np.newaxis, :] - z[:, np.newaxis]) ** 2
        d_phi = np.sum((omega * dt)[np.newaxis, :] /
                       t_minus_z_sq, axis=1) / (2j * np.pi)

        pressure = np.imag(d_phi)
        vorticity = np.real(d_phi)
        return pressure, vorticity

    def build(self):
        # building the solver
        self.build_geometry()
        self.build_kernel()
        self.build_omega()
        self.build_pressure_drops()

        # building the graph theory components
        pass

        # building the data for drawing the pictures

    def build_pic(self, density=100):
        density = np.sqrt(density)
        left, right, bottom, top = self.extent
        x = np.linspace(left, right, np.ceil((right - left) * density))
        y = np.linspace(bottom, top, np.ceil((top - bottom) * density))
        self.grid = np.meshgrid(x,y,sparse=True)

        pass

    def build_graph(self):
        n = len(self.lets)
        assert n > 1

        # this graph has n vertices and (n-1) edges

    def is_inside(self, z):
        pass


class StraightPipe(Pipe):
    def __init__(self, p1, p2, r=1) -> None:
        """
        this creates a simple tube. Why do I create it first? because it serves well as
        a template for other more sophisticated geometries.
        """
        super().__init__()

        self.angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        theta = self.angle + np.pi / 2
        top_left = p1 + r * np.array([np.cos(theta), np.sin(theta)])
        bottom_left = p1 - r * np.array([np.cos(theta), np.sin(theta)])
        top_right = p2 + r * np.array([np.cos(theta), np.sin(theta)])
        bottom_right = p2 - r * np.array([np.cos(theta), np.sin(theta)])

        top_line = Line(top_left, top_right)
        right_line = Line(top_right, bottom_right)
        bottom_line = Line(bottom_right, bottom_left)
        left_line = Line(bottom_left, top_left)

        self.curves = [top_line, right_line, bottom_line, left_line]

        up = np.max(top_left[1], top_right[1], bottom_left[1], bottom_right[1])
        low = np.min(top_left[1], top_right[1], bottom_left[1], bottom_right[1])
        left = np.min(top_left[0], top_right[0], bottom_left[0], bottom_right[0])
        right = np.max(top_left[0], top_right[0], bottom_left[0], bottom_right[0])


class auto_tube_constructor(Pipe):
    """
    given points
        p1, p2, p3, ... , pn
    and the description (type) of the Line
        l1 = p1-p2
        l2 = p2-p3
        ...
        ln = pn-p1
    this class should firstly create a cornered Curve with the given specification of points and lines
    then it should automatically smooth the corners.
    """
    def __init__(self,pts,lines,corner_size=1e-1):

        assert len(pts) == len(lines)
        if not np.all([i in [Line, Cap] for i in lines]):
            raise ValueError("invalid Curve type")

        self.curves = []
        self.corner_size = corner_size

        # n =

