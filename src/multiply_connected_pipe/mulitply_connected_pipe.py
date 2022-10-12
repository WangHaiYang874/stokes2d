from abstract_pipe.let import BoundaryLet
from curve import *
from utils import *
from pipe.mat_vec import MatVec
from .boundary import Boundary
from pipe_system import PipeSystem

from numpy import ndarray, concatenate, pi, conjugate, array, newaxis
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.interpolate import griddata, NearestNDInterpolator
import networkx as nx

import warnings
from joblib import Parallel, delayed
from copy import deepcopy


class MultiplyConnectedPipe:
    """interior stokes problem with multiply-connected domain"""

    ### geometric data ###

    boundaries: List[Boundary]
    exterior_boundary: Boundary
    interior_boundaries: List[Boundary]
    n_boundaries: int
    indices_of_boundary: List[Tuple[int, int]]

    n_pts: int          # number of points on the boundary curve
    da: ndarray         # the quadrature weights
    t: ndarray          # the complex coordinates of boundary points
    dt_da: ndarray      # the derivative of t with respect to the parameter
    dt: ndarray         # dt = dt_da * da
    k: ndarray          # the curvature of the boundary curve
    z: np.complex128    # this is just a point inside the domain.
    # single solver, Matrix from Nyestorm's discretization. this is the unconstrained version. Implementing the constrained version can speed up the convergence of gmres. But I don't care that much about this...
    A: LinearOperator

    ### flows ###

    # the index of the inlet/outlet in the list of curves
    let_index2curve_index: List[int]
    lets: List[Cap]
    n_lets: int
    n_flows: int            # number of flows
    omegas: ndarray         # shape=(n_flows, n_pts), dtype=complex128
    pressure_drops: ndarray  # shape=(nfluxes, nflows), dtype=float64

    def __init__(self, pipe_sys:PipeSystem) -> None:
        
        caps_to_keep = []
        for v in pipe_sys.vertices:
            if v.atBdr:
                l = v.l1 if isinstance(v.l1, BoundaryLet) else v.l2
                caps_to_keep.append((l.pipeIndex,l.letIndex))

        curves = []

        for pipe_index, pipe in enumerate(pipe_sys.pipes):
            
            shift = pipe.shift
            pipe = pipe.prototye
            c_index2l_index = {c:l for l,c in enumerate(pipe.let_index2curve_index)}
            
            for curve_index, curve in enumerate(pipe.curves):

                if isinstance(curve, Cap):
                    let_index = c_index2l_index[curve_index]
                    if (pipe_index, let_index) not in caps_to_keep:
                        continue
                
                c = deepcopy(curve)
                for p in c.panels:
                    p.x += shift[0]
                    p.y += shift[1]

                c.start_pt += shift
                c.end_pt += shift
                c.mid_pt += shift
                
                if isinstance(c, Cap):
                    c.matching_pt += shift

                curves.append(c)

        G = nx.Graph()

        for c in curves:
            G.add_edge(pt2tuple(c.start_pt),pt2tuple(c.end_pt), curve=c)

        pts = np.array(list(G.nodes))
        pts_cplx = pts[:,0] + 1j*pts[:,1]
        distance = np.abs(pts_cplx[:,None] - pts_cplx[None,:])
        need_to_merge = (distance < 1e-10) & (distance > 0)

        while np.any(need_to_merge):
            i,j = np.array(np.where(need_to_merge)).T[0]
            
            node1 = list(G.nodes)[i]
            node2 = list(G.nodes)[j]

            nx.contracted_nodes(G,node1,node2, self_loops=False, copy=False)
            
            pts = np.array(list(G.nodes))
            pts_cplx = pts[:,0] + 1j*pts[:,1]
            distance = np.abs(pts_cplx[:,None] - pts_cplx[None,:])
            need_to_merge = (distance < 1e-9) & (distance > 0)
            
        assert len(G.nodes) == len(set(G.edges))

        boundaries = []
        
        for c in nx.cycle_basis(G):
            curves = []
            for node1,node2 in zip(c, c[1:] + c[:1]):
                curves.append(G.edges[node1,node2]['curve'])
            boundaries.append(curves)
            
        boundaries = [Boundary(b) for b in boundaries]
        self.boundaries = sorted(boundaries, key=lambda boundary: np.min(boundary.t.real))
        
                    
    @property
    def exterior_boundary(self):
        return self.boundaries[0]

    @property
    def interior_boundaries(self):
        return self.boundaries[1:]

    @property
    def n_boundaries(self):
        return len(self.boundaries)

    ### GEOMETRIC ###
    @property
    def n_pts(self): return len(self.da)
    @property
    def da(self): return concatenate([c.da for c in self.boundaries])
    @property
    def t(self): return concatenate([c.t for c in self.boundaries])
    @property
    def dt_da(self): return concatenate([c.dt_da for c in self.boundaries])
    @property
    def dt(self): return self.dt_da * self.da
    @property
    def k(self): return concatenate([c.k for c in self.boundaries])

    @property
    def z(self):
        # here I assume that the domain is convex,
        # so I will simply take the average of points on the boundary.
        # TODO: the more careful algorithm to handle non-convex domains is to use
        # poles of inaccessibility.
        # PIA has a convenient python implementation at
        # https://github.com/shapely/shapely/blob/main/shapely/algorithms/polylabel.py

        return np.mean(self.t)

    def build(self, max_distance=None, legendre_ratio=None, tol=None, n_jobs=1):
        self.build_geometry(max_distance, legendre_ratio, n_jobs)
        self.build_A()
        self.build_omegas(tol=tol, n_jobs=n_jobs)
        self.A = None  # free memory
        self.build_pressure_drops()

    def build_geometry(self, max_distance=None, legendre_ratio=None, n_jobs=1):

        if n_jobs == 1:
            [b.build(max_distance, legendre_ratio) for b in self.boundaries]
        else:
            def build_boundary(b):
                b.build(max_distance, legendre_ratio)
                return b
            self.boundaries = Parallel(n_jobs=n_jobs)(
                delayed(build_boundary)(b) for b in self.boundaries)

    ### SINGLE SOLVER ###

    @property
    def indices_of_boundary(self):
        index = np.insert(np.cumsum([b.n_pts for b in self.boundaries]), 0, 0)
        return [(index[i], index[i+1]) for i in range(len(index)-1)]

    def build_A(self):
        # TODO: test this function. If this has a large condition number, then it's probably correct.
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

        for k, m in zip(range(self.n_boundaries), range(1, self.n_boundaries)):

            k_start, k_end = self.indices_of_boundary[k]
            m_start, m_end = self.indices_of_boundary[m]

            t_minus_z = (self.boundaries[k].t -
                         self.boundaries[m].z)[:, newaxis]
            dt = self.boundaries[m].dt[newaxis, :]
            da = self.boundaries[m].da[newaxis, :]

            K1[k_start:k_end, m_start:m_end] += \
                1j*np.conjugate(dt/(t_minus_z)) \
                + 2*da*np.log(np.abs(t_minus_z))

            K2[k_start:k_end, m_start:m_end] += \
                (da*t_minus_z-1j*dt)/(np.conjugate(t_minus_z))

        self.A = LinearOperator(matvec=MatVec(K1, K2),
                                dtype=np.float64,
                                shape=(2*self.n_pts, 2*self.n_pts))

    def compute_omega(self, H, tol=None):

        tol = GMRES_TOL if tol is None else tol
        b = concatenate((H.real, H.imag))

        omega_sep, _ = gmres(self.A, b, atol=0, tol=tol, restart=RESTART)

        if _ < 0:
            warnings.warn("gmres is not converging to tolerance. ")
            assert False

        omega = omega_sep[:self.n_pts] + 1j*omega_sep[self.n_pts:]
        return omega

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

    def build_omegas(self, tol=None, n_jobs=1):
        assert n_jobs > 0
        if n_jobs == 1:
            self.omegas = array(
                [self.compute_omega(self.boundary_value(i), tol) for i in range(self.n_flows)])
        else:
            self.omegas = array(Parallel(n_jobs=n_jobs)(
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

    def build_plotting_data(self, xs, ys, interior):

        near_boundary = ~interior

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

            # interior can be directly calculated

            velocity = self.velocity(xs[interior], ys[interior], omega)
            pressure, vorticity = self.pressure_and_vorticity(
                xs[interior], ys[interior], omega)
            base_pressure = self.pressure_and_vorticity(
                base_x, base_y, omega)[0][0]

            pressure -= base_pressure

            u_field[interior] = velocity[:, 0]
            v_field[interior] = velocity[:, 1]
            p_field[interior] = pressure
            o_field[interior] = vorticity

            # near boundary data need to be interpolated/extrapolated.

            for field in [u_field, v_field, p_field, o_field]:
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
            p_fields.append(p_field)
            o_fields.append(o_field)

        self.xs = xs
        self.ys = ys
        self.interior = interior

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


def pt2tuple(pt):
    assert pt.shape == (2,)
    return (pt[0],pt[1])
