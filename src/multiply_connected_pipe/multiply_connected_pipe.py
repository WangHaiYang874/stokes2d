from curve import *
from utils import *
from mat_vec import MatVec
from curve import Boundary
from mat_vec import MatVec, mat_vec_constructor

from numpy import ndarray, concatenate, array
from scipy.sparse.linalg import gmres, LinearOperator

import warnings
from joblib import Parallel, delayed


class MultiplyConnectedPipe:
    """interior stokes problem with multiply-connected domain"""

    ### geometric data ###

    boundaries: List[Boundary]
    exterior_boundary: Boundary
    interior_boundaries: List[Boundary]
    n_boundaries: int
    indices_of_boundary: List[Tuple[int, int]]
    curves: List[Curve]  # there should be a better way to merge pipe systems.
    panels: List[Panel]

    n_pts: int          # number of points on the boundary curve
    da: ndarray         # the quadrature weights
    t: ndarray          # the complex coordinates of boundary points
    dt_da: ndarray      # the derivative of t with respect to the parameter
    dt: ndarray         # dt = dt_da * da
    k: ndarray          # the curvature of the boundary curve
    z: np.complex128    # this is just a point inside the domain.
    A: LinearOperator
    mat_vec: MatVec

    ### flows ###

    # the index of the inlet/outlet in the list of curves
    let_index2curve_index: List[int]
    lets: List[Cap]
    n_lets: int
    n_flows: int            # number of flows
    omegas: ndarray         # shape=(n_flows, n_pts), dtype=complex128
    pressure_drops: ndarray  # shape=(nfluxes, nflows), dtype=float64

    def __init__(self) -> None:
        pass

    # graphic
    extent: Tuple[float, float, float, float]
    xs: ndarray
    ys: ndarray
    u_fields: ndarray
    v_fields: ndarray
    p_fields: ndarray
    o_fields: ndarray

    ### GEOMETRIC ###

    @property
    def exterior_boundary(self): return self.boundaries[0]
    @property
    def interior_boundaries(self): return self.boundaries[1:]
    @property
    def n_boundaries(self): return len(self.boundaries)
    @property
    def is_simply_connected(self): return self.n_boundaries == 1
    @property
    def n_pts(self): return len(self.da)
    @property
    def da(self): return concatenate([b.da for b in self.boundaries])
    @property
    def t(self): return concatenate([b.t for b in self.boundaries])
    @property
    def dt_da(self): return concatenate([b.dt_da for b in self.boundaries])
    @property
    def dt(self): return self.dt_da * self.da
    @property
    def k(self): return concatenate([c.k for c in self.boundaries])

    @property
    def indices_of_boundary(self):
        index = np.insert(np.cumsum([b.n_pts for b in self.boundaries]), 0, 0)
        return [(index[i], index[i+1]) for i in range(len(index)-1)]

    @property
    def curves(self):
        return [c for b in self.boundaries for c in b.curves]

    @property
    def panels(self):
        return [p for c in self.curves for p in c.panels]

    ### FLOWS ###

    @property
    def let_index2curve_index(self):
        return [i for i, c in enumerate(self.exterior_boundary.curves) if isinstance(c, Cap)]

    @property
    def lets(self): return [self.exterior_boundary.curves[i]
                            for i in self.let_index2curve_index]

    @property
    def n_lets(self): return len(self.lets)
    @property
    def n_flows(self): return self.n_lets - 1
    @property
    def extent(self): return self.exterior_boundary.extent

    def build(self, required_tol=REQUIRED_TOL, n_jobs=1, fmm=None):
        self.build_geometry(required_tol=required_tol)
        self.build_A(fmm=fmm)
        self.build_omegas(tol=required_tol, n_jobs=n_jobs)
        self.build_pressure_drops()
        self.build_plotting_data()

    def build_geometry(self, required_tol=REQUIRED_TOL):

        p = (np.ceil(-np.log10(required_tol)) + 2).astype(int)

        # this is not enough for handling corners, but we don't have any corners.
        # bent panel refinement is ignored for our solver.

        # building each curve separately
        [b.build(required_tol, p) for b in self.boundaries]

        # refining panels to handle the close panel interaction
        for ib, b in enumerate(self.boundaries):
            for ic, c in enumerate(b.curves):
                ip = 0
                while ip < len(c.panels):
                    p = c.panels[ip]
                    s = p.arclen
                    ip_boundary = ip + sum([len(c_.panels)
                                           for c_ in b.curves[:ic]])
                    ip_boundary_next = (ip_boundary + 1) % len(b.panels)
                    ip_boundary_prev = (ip_boundary - 1) % len(b.panels)
                    boundary_offset = sum([len(b_.panels)
                                          for b_ in self.boundaries[:ib]])
                    adj = [boundary_offset + ip_boundary_next, boundary_offset +
                           ip_boundary_prev, boundary_offset + ip_boundary]

                    j = 0
                    good = True

                    while j < len(self.panels):
                        if j in adj:
                            j += 1
                            continue
                        p2 = self.panels[j]
                        if s < 3*np.min(np.abs(p.t[:, np.newaxis] - p2.t[np.newaxis, :])):
                            j += 1
                            continue
                        # need to refine
                        c.panels.pop(ip)
                        p1, p2 = p.refined()
                        c.panels.insert(ip, p2)
                        c.panels.insert(ip, p1)
                        good = False
                        break

                    if good:
                        ip += 1

    def build_A(self, fmm=None):
        self.mat_vec = mat_vec_constructor(self, fmm=fmm)
        self.A = LinearOperator(
            matvec=self.mat_vec, dtype=np.float64, shape=(2*self.n_pts, 2*self.n_pts))

    def build_omegas(self, tol=None, n_jobs=1):
        assert n_jobs > 0
        n_jobs = min(n_jobs, self.n_flows)

        if n_jobs == 1:
            self.omegas = array([
                self.compute_omega(self.boundary_value(i), tol)
                for i in range(self.n_flows)])
        else:
            self.omegas = array(Parallel(n_jobs=n_jobs)(
                delayed(lambda i: self.compute_omega(
                    self.boundary_value(i), tol))(i)
                for i in range(self.n_flows)))

    def build_pressure_drops(self):
        pts = array([let.matching_pt for let in self.lets])
        pressure_drops = []
        for omega in self.omegas:
            pressure, _ = self.pressure_and_vorticity(
                pts[:, 0], pts[:, 1], omega)
            pressure_drops.append(pressure[1:] - pressure[0])
        self.pressure_drops = array(pressure_drops)

    def compute_omega(self, H, tol=None, max_iter=None):

        tol = GMRES_TOL if tol is None else tol
        max_iter = GMRES_MAX_ITER if max_iter is None else max_iter
        b = concatenate((H.real, H.imag))

        omega_sep, _ = gmres(self.A, b, atol=0, tol=tol,
                             restart=RESTART, maxiter=max_iter)

        if _ < 0:
            warnings.warn("gmres is not converging to tolerance. ")
            assert False

        omega = omega_sep[:self.n_pts] + 1j*omega_sep[self.n_pts:]
        return omega

    def boundary_value(self, flow_index):
        """Flow goes in the inlet and out the outlet. """
        o = self.let_index2curve_index[flow_index+1]  # outlet
        i = self.let_index2curve_index[0]             # inlet
        ret = []
        for j, c in enumerate(self.exterior_boundary.curves):
            if j == i:
                ret.append(-c.boundary_velocity())
            elif j == o:
                ret.append(c.boundary_velocity())
            else:
                ret.append(np.zeros_like(c.a))

        ret = concatenate(ret)
        rest = np.zeros(self.n_pts - len(ret))
        return np.concatenate((ret, rest))

    def d_phi(self, x, y, omega):
        return self.mat_vec.d_phi(x, y, omega)

    def velocity(self, x, y, omega: np.ndarray):
        return H2U(self.mat_vec.velocity(x, y, omega))

    def pressure_and_vorticity(self, x, y, omega):

        d_phi = self.d_phi(x, y, omega)
        d_phi_init = self.d_phi(
            array([self.lets[0].matching_pt[0]]),
            array([1j*self.lets[0].matching_pt[1]]), omega)

        pressure = np.imag(d_phi) - np.imag(d_phi_init)
        vorticity = np.real(d_phi)

        return pressure, vorticity

    def clear_geometry(self):
        self.boundaries = [b.clean_copy() for b in self.boundaries]

    def inside(self, xs, ys):
        m = self.exterior_boundary.inside(xs, ys)
        for b in self.interior_boundaries:
            m = m & ~b.inside(xs, ys)
        return m

    def build_plotting_data(self, density=20):

        # points
        xmin, xmax, ymin, ymax = self.extent
        xs = np.linspace(xmin, xmax, (density*(xmax-xmin)+1).astype(int))
        ys = np.linspace(ymin, ymax, (density*(ymax-ymin)+1).astype(int))
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.flatten()
        ys = ys.flatten()
        m = self.inside(xs, ys)
        xs = xs[m]
        ys = ys[m]
        self.xs = xs
        self.ys = ys

        u_fields = []
        v_fields = []
        p_fields = []
        o_fields = []

        base_x, base_y = self.lets[0].matching_pt
        base_x = np.array([base_x])
        base_y = np.array([base_y])

        for omega in self.omegas:

            u_field = np.zeros_like(xs)
            v_field = np.zeros_like(xs)
            p_field = np.zeros_like(xs)
            o_field = np.zeros_like(xs)

            velocity = self.velocity(xs, ys, omega)
            pressure, vorticity = \
                self.pressure_and_vorticity(xs, ys, omega)
            base_pressure = self.pressure_and_vorticity(
                base_x, base_y, omega)[0][0]

            pressure -= base_pressure

            u_field = velocity[:, 0]
            v_field = velocity[:, 1]
            p_field = pressure
            o_field = vorticity

            u_fields.append(u_field)
            v_fields.append(v_field)
            p_fields.append(p_field)
            o_fields.append(o_field)

        self.u_fields = np.array(u_fields)
        self.v_fields = np.array(v_fields)
        self.p_fields = np.array(p_fields)
        self.o_fields = np.array(o_fields)

    def fields_with_fluxes(self, fluxes, base_let_index, base_pressure):
        assert isinstance(fluxes, np.ndarray)
        assert fluxes.ndim == 1
        assert len(fluxes) == self.n_flows

        u = fluxes@self.u_fields
        v = fluxes@self.v_fields
        p = fluxes@self.p_fields
        o = fluxes@self.o_fields

        if base_let_index == 0:
            curr_pressure = 0
        else:
            curr_pressure = (fluxes@self.pressure_drops)[base_let_index-1]

        p = p - curr_pressure + base_pressure

        return u, v, p, o
