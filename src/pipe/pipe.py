from curve import *
from utils import *
from .mat_vec import MatVec

from numpy import ndarray, concatenate, pi, conjugate, array, newaxis
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.interpolate import griddata, NearestNDInterpolator
from shapely.geometry import LineString, Polygon
from matplotlib.path import Path

import warnings
from joblib import Parallel, delayed, cpu_count
from typing import List, Tuple
import pickle


class Pipe:
    ### geometric data ###

    curves: List[Curve]  # the parametrized curve.
    n_pts: int          # number of points on the boundary curve
    a: ndarray          # the parameter of the boundary curve
    da: ndarray         # the quadrature weights
    t: ndarray          # the complex coordinates of boundary points
    dt_da: ndarray      # the derivative of t with respect to the parameter
    dt: ndarray         # dt = dt_da * da
    k: ndarray          # the curvature of the boundary curve

    # single solver, Matrix from Nyestorm's discretization
    A: LinearOperator

    ### flows ###

    # the index of the inlet/outlet in the list of curves
    let_index2curve_index: List[int]
    lets: List[Cap]
    n_lets: int
    n_flows: int            # number of flows
    omegas: ndarray         # shape=(n_flows, n_pts), dtype=complex128
    pressure_drops: ndarray  # shape=(nfluxes, nflows), dtype=float64

    # picture data
    h: float
    boundary: ndarray           # shape=(*, 2), dtype=float64.
    open_bdr: ndarray
    smooth_boundary: ndarray
    smooth_closed_boundary: ndarray
    closed_boundary: ndarray    # shape=(*, 2), dtype=float64.
    interior_boundary: ndarray  # shape=(*, 2), dtype=float64.
    closed_interior_boundary: ndarray     # shape=(*, 2), dtype=float64.
    # list[ndarray with shape (*,2) and dtype float64].
    extent: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)

    xs: ndarray
    ys: ndarray
    u_fields: ndarray  # shape=(n_flows, x, y)
    v_fields: ndarray  # shape=(n_flows, x, y)
    pressure_field: ndarray  # shape=(n_flows, x, y)
    vorticity_field: ndarray  # shape=(n_flows, x, y)

    def __init__(self) -> None:
        pass

    ### GEOMETRIC ###

    @property
    def a(self): return concatenate(
        [c.a + 2*i for i, c in enumerate(self.curves)])

    @property
    def n_pts(self): return len(self.a)
    @property
    def da(self): return concatenate([c.da for c in self.curves])
    @property
    def t(self): return concatenate([c.t for c in self.curves])
    @property
    def dt_da(self): return concatenate([c.dt_da for c in self.curves])
    @property
    def dt(self): return self.dt_da * self.da
    @property
    def k(self): return concatenate([c.k for c in self.curves])

    def build(self, max_distance=None, legendre_ratio=None, tol=None, density=None, h_mult=None, n_jobs=1):
        self.build_geometry(max_distance, legendre_ratio, n_jobs)
        self.build_A()
        self.build_omegas(tol=tol, n_jobs=n_jobs)
        # self.A = None  # free memory
        self.build_pressure_drops()
        self.build_plotting_data(h_mult, density)
        # self.omegas = None  # free memory
        """
        # TODO: what fields should be kept?
        - lets2curve_index,
        - n_flows,
        - pressure_drops,
        - boundary, closed_boundary, open_bdr,
        - extent,
        - xs, ys
        - u_fields, v_fields,
        - pressure_field, voricity_field        
        """

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def build_geometry(self, max_distance=None, legendre_ratio=None, n_jobs=1):

        if n_jobs == 1:
            [c.build(max_distance, legendre_ratio) for c in self.curves]
        else:
            def build_curve(c):
                c.build(max_distance, legendre_ratio)
                return c
            self.curves = Parallel(n_jobs=min(n_jobs, len(self.curves), cpu_count()//2))(
                delayed(build_curve)(c) for c in self.curves)

    ### SINGLE SOLVER ###

    def build_A(self, fmm=False):

        if fmm:
            # TODO Probably implement later.
            return NotImplemented

        # diff_t[i, j] = t[i] - t[j]
        diff_t = self.t[:, newaxis] - self.t[newaxis, :]
        # dt2[*,i] = dt[i] = dt_da[i] * da[i]
        dt2 = self.dt[newaxis, :]

        # this ignores the error for computing the diagonal elements with 0/0 error
        with np.errstate(divide='ignore', invalid='ignore'):
            K1 = np.imag(dt2/diff_t) / (-pi)
            K2 = (dt2 / conjugate(diff_t) - conjugate(dt2)
                  * diff_t/(conjugate(diff_t**2))) / (2j*pi)

        # now we need to fill the diagonal elements
        K1_diagonal = self.k*np.abs(self.dt)/(2*pi)
        K2_diagonal = (self.k*self.dt*self.dt_da) / \
            (-2*pi*np.abs(self.dt_da)**2)
        np.fill_diagonal(K1, K1_diagonal)
        np.fill_diagonal(K2, K2_diagonal)

        self.A = LinearOperator(matvec=MatVec(K1, K2),
                                dtype=np.float64,
                                shape=(2*self.n_pts, 2*self.n_pts))

    def compute_omega(self, H, tol=None):

        tol = GMRES_TOL if tol is None else tol
        b = concatenate((H.real, H.imag))

        omega_sep, _ = gmres(self.A, b, atol=0, tol=tol)

        if _ < 0:
            warnings.warn("gmres is not converging to tolerance. ")
            assert False

        omega = omega_sep[:self.n_pts] + 1j*omega_sep[self.n_pts:]
        return omega

    ### FLOWS ###
    @property
    def let_index2curve_index(self):
        return [i for i, c in enumerate(self.curves) if isinstance(c, Cap)]

    @property
    def lets(self): return [self.curves[i] for i in self.let_index2curve_index]
    @property
    def n_lets(self): return len(self.lets)
    @property
    def n_flows(self): return self.n_lets - 1

    def boundary_value(self, flow_index):
        """Flow goes in the inlet and out the outlet. """
        o = self.let_index2curve_index[flow_index+1]  # outlet
        i = self.let_index2curve_index[0]             # inlet
        ret = []
        for j, c in enumerate(self.curves):
            if j == i:
                ret.append(-c.boundary_velocity())
            elif j == o:
                ret.append(c.boundary_velocity())
            else:
                ret.append(np.zeros_like(c.a))
        return concatenate(ret)

    def build_omegas(self, tol=None, n_jobs=1):
        assert n_jobs > 0
        if n_jobs == 1:
            self.omegas = array(
                [self.compute_omega(self.boundary_value(i), tol) for i in range(self.n_flows)])
        else:
            self.omegas = array(Parallel(n_jobs=min(n_jobs, self.n_flows, cpu_count()//2))(
                delayed(lambda i: self.compute_omega(
                    self.boundary_value(i), tol))(i)
                for i in range(self.n_flows)))

    def build_pressure_drops(self):
        pts = array([let.matching_pt for let in self.lets])
        pressure_drops = []
        for omega in self.omegas:
            pressure = self.pressure(pts[:, 0], pts[:, 1], omega)
            pressure_drops.append(pressure[1:] - pressure[0])

        self.pressure_drops = array(pressure_drops)

    ### COMPUTE PHYSICS QUANTITIES ###

    def phi(self, z, omega):
        assert z.ndim == 1
        return np.sum((omega * self.dt)[newaxis, :] /
                      (self.t[newaxis, :] - z[:, newaxis]), axis=1) / (2j * pi)

    def d_phi(self, z, omega):
        assert z.ndim == 1
        return np.sum((omega * self.dt)[newaxis, :] /
                      (self.t[newaxis, :] - z[:, newaxis])**2, axis=1) / (2j * pi)

    def psi(self, z, omega):
        assert z.ndim == 1

        first_term = np.sum(
            np.real((conjugate(omega) * self.dt)[newaxis, :])
            / (self.t[newaxis, :] - z[:, newaxis]),
            axis=1) / (1j*pi)

        second_term = np.sum(
            (conjugate(self.t) * omega * self.dt)[newaxis, :]
            / (self.t[newaxis, :] - z[:, newaxis])**2,
            axis=1) / (-2j * pi)

        return first_term + second_term

    def velocity(self, x, y, omega):

        z = x + 1j*y
        assert isinstance(z, ndarray)
        assert z.ndim == 1

        return H2U((self.phi(z, omega) + z * conjugate(self.d_phi(z, omega)) + conjugate(self.psi(z, omega))))

    def pressure_and_vorticity(self, x, y, omega):

        # TODO verify
        z = x + 1j*y
        assert (isinstance(z, ndarray))
        assert (z.ndim == 1)

        d_phi = self.d_phi(z, omega)
        d_phi_init = self.d_phi(
            array([self.lets[0].matching_pt[0]+1j*self.lets[0].matching_pt[1]]), omega)

        # this serves an normalization purpose.
        pressure = np.imag(d_phi) - np.imag(d_phi_init)
        vorticity = np.real(d_phi)

        return pressure, vorticity

    def pressure(self, x, y, omega):
        return self.pressure_and_vorticity(x, y, omega)[0]

    def vorticity(self, x, y, omega):
        return self.pressure_and_vorticity(x, y, omega)[1]

    ### PLOTTING ###

    @property
    def boundary(self):
        pts = []
        for c in self.curves:
            if isinstance(c, Cap):
                pts += [c.start_pt, c.matching_pt]
            elif isinstance(c, Line):
                pts += [c.start_pt, c.mid_pt]
            elif isinstance(c, Corner):
                pts += [[c.x[i], c.y[i]] for i in range(len(c.a))]
        return array(pts)

    @property
    def open_bdr(self):

        bdrs = []

        for i in range(self.n_lets):
            cap1 = self.let_index2curve_index[i]
            cap2 = self.let_index2curve_index[(i+1) % self.n_lets]
            cap1 = cap1 - len(self.curves) if cap1 > cap2 else cap1

            bdr = []
            for curve_index in range(cap1+1, cap2):
                c = self.curves[curve_index % len(self.curves)]
                if isinstance(c, Cap):
                    assert False
                elif isinstance(c, Line):
                    bdr += [c.start_pt, c.mid_pt]
                elif isinstance(c, Corner):
                    bdr += [[c.x[i], c.y[i]] for i in range(len(c.a))]
            bdr.append(c.end_pt)
            bdrs.append(array(bdr))

        return bdrs

    @property
    def smooth_boundary(self):
        pts = []
        for c in self.curves:
            if isinstance(c, Line):
                pts += [c.start_pt, c.mid_pt]
            elif isinstance(c, Corner) or isinstance(c, Cap):
                pts += [[c.x[i], c.y[i]] for i in range(len(c.a))]
        return array(pts)

    @property
    def smooth_closed_boundary(self):
        return concatenate((self.smooth_boundary, self.smooth_boundary[:1]))

    @property
    def closed_boundary(self):
        return concatenate((self.boundary, self.boundary[:1]))

    @property
    def h(self):
        return np.max(np.abs(np.diff(self.t)))

    def interior_boundary(self, h_mult=None):

        distance = 4*self.h if h_mult is None else h_mult*self.h
        # this constant 4 here is tested to be good.
        # this is a heuristic similar to the 5h-rule for BIM of harmonic equation.

        p1 = Polygon(LineString(concatenate((self.smooth_boundary, self.smooth_boundary[:1]))).buffer(
            distance).interiors[0])
        p2 = Polygon(self.closed_boundary)
        x, y = p1.intersection(p2).boundary.xy

        return array([x, y]).T[:-1]

    @property
    def extent(self):
        left = np.min(self.boundary[:, 0])
        right = np.max(self.boundary[:, 0])
        bottom = np.min(self.boundary[:, 1])
        top = np.max(self.boundary[:, 1])
        return (left, right, bottom, top)

    def masks(self, x, y, h_mult=None):

        inside = Path(self.boundary).contains_points(
            array([x, y]).T)
        interior = Path(self.interior_boundary(h_mult)).contains_points(
            array([x, y]).T)
        near_boundary = inside & ~interior

        return inside, interior, near_boundary

    def grid_pts(self, density=None):

        density = 100 if density is None else density

        left, right, bottom, top = self.extent
        nx = np.ceil((right - left) * density).astype(int)
        ny = np.ceil((top - bottom) * density).astype(int)

        xs = np.linspace(left, right, nx)
        ys = np.linspace(bottom, top, ny)

        xs, ys = np.meshgrid(xs, ys)
        return xs.ravel(), ys.ravel()

    def build_plotting_data(self, h_mult=None, density=None, n_jobs=1):

        xs, ys = self.grid_pts(density)

        inside, interior, near_boundary = self.masks(xs, ys, h_mult)

        xs = xs[inside]
        ys = ys[inside]
        interior = interior[inside]
        near_boundary = near_boundary[inside]

        u_fields = []
        v_fields = []
        pressure_fields = []
        vorticity_fields = []

        # base point of pressure and vorticity
        base_x, base_y = self.lets[0].matching_pt
        base_x = np.array([base_x])
        base_y = np.array([base_y])

        if n_jobs != 1:
            # TODO Implement
            raise NotImplementedError(
                "Parallel computation is not implemented yet.")

        for omega in self.omegas:

            u_field = np.zeros_like(xs)
            v_field = np.zeros_like(xs)
            pressure_field = np.zeros_like(xs)
            vorticity_field = np.zeros_like(xs)

            # interior can be directly calculated

            v = self.velocity(xs[interior], ys[interior], omega)
            pressure, vorticity = self.pressure_and_vorticity(
                xs[interior], ys[interior], omega)
            base_pressure, base_vorticity = self.pressure_and_vorticity(
                base_x, base_y, omega)
            base_pressure = base_pressure[0]
            base_vorticity = base_vorticity[0]

            pressure -= base_pressure
            vorticity -= base_vorticity

            u_field[interior] = v[:, 0]
            v_field[interior] = v[:, 1]
            pressure_field[interior] = pressure
            vorticity_field[interior] = vorticity

            # near boundary data need to be interpolated/extrapolated.

            for field in [u_field, v_field, pressure_field, vorticity_field]:
                field[near_boundary] = griddata(
                    np.array([xs[interior], ys[interior]]).T, field[interior],
                    np.array([xs[near_boundary], ys[near_boundary]]).T, method='linear')

                if np.any(np.isnan(field[near_boundary])):
                    nearest_extrapolate = NearestNDInterpolator(
                        np.array([xs[interior], ys[interior]]).T, field[interior])
                    nan_mask = np.isnan(field) & near_boundary
                    field[nan_mask] = nearest_extrapolate(
                        xs[nan_mask], ys[nan_mask])

            # store the fields

            u_fields.append(u_field)
            v_fields.append(v_field)
            pressure_fields.append(pressure_field)
            vorticity_fields.append(vorticity_field)

        self.xs = xs
        self.ys = ys

        self.u_fields = np.array(u_fields)
        self.v_fields = np.array(v_fields)
        self.pressure_fields = np.array(pressure_fields)
        self.vorticity_fields = np.array(vorticity_fields)

    def fields_with_fluxes(self, fluxes, base_let_index, base_pressure):
        # TODO verify this is correct.
        assert isinstance(fluxes, np.ndarray)
        assert fluxes.ndim == 1
        assert len(fluxes) == self.n_flows

        u = fluxes@self.u_fields
        v = fluxes@self.v_fields
        p = fluxes@self.pressure_fields
        o = fluxes@self.vorticity_fields

        if base_let_index == 0:
            curr_pressure = 0
        else:
            curr_pressure = fluxes@self.pressure_drops[base_let_index-1]

        p = p - curr_pressure + base_pressure

        return u, v, p, o
