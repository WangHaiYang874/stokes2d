from curve import *
from utils import *
from pipe.mat_vec import MatVec
from .boundary import Boundary
from .fmm import A_fmm


from numpy import ndarray, concatenate, pi, conjugate, array, newaxis
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.interpolate import griddata, NearestNDInterpolator

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

    n_pts: int          # number of points on the boundary curve
    da: ndarray         # the quadrature weights
    t: ndarray          # the complex coordinates of boundary points
    dt_da: ndarray      # the derivative of t with respect to the parameter
    dt: ndarray         # dt = dt_da * da
    k: ndarray          # the curvature of the boundary curve
    z: np.complex128    # this is just a point inside the domain.
    # single solver, Matrix from Nyestorm's discretization. this is the unconstrained version. Implementing the constrained version can speed up the convergence of gmres. But I don't care that much about this...
    A: LinearOperator
    fmm: A_fmm

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
    

    def build(self, max_distance=None, legendre_ratio=None, tol=None, n_jobs=1, fmm=True):
        assert fmm
        self.build_geometry(max_distance, legendre_ratio, n_jobs)
        self.build_A_fmm()
        if not fmm:
            self.build_A()
        self.build_omegas(tol=tol, n_jobs=n_jobs)
        if not fmm:
            self.A = None # free up memory. 
        
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

    def build_A_fmm(self):
        self.fmm = A_fmm(self)
        self.A = LinearOperator(matvec=self.fmm,
                                dtype=np.float64,
                                shape=(2*self.n_pts, 2*self.n_pts))

    def build_A(self):

        ### the first part is the non-singular part of the matrix. ###

        # diff_t[i, j] = t[i] - t[j]
        diff_t = self.t[:, newaxis] - self.t[newaxis, :]
        # dt2[*,i] = dt[i] = dt_da[i] * da[i]
        dt2 = self.dt[newaxis, :]

        K1 = np.zeros(shape=(self.n_pts, self.n_pts), dtype=np.complex128)
        K2 = np.zeros(shape=(self.n_pts, self.n_pts), dtype=np.complex128)

        # this ignores the error for computing the diagonal elements with 0/0 error
        with np.errstate(divide='ignore', invalid='ignore'):
            K1 -= np.imag(dt2/diff_t) / pi
            K2 += np.imag(dt2*np.conj(diff_t)) / (np.conj(diff_t**2)*pi)
        # now we need to fill the diagonal elements
        K1_diagonal = self.k*np.abs(self.dt)/(2*pi)
        K2_diagonal = (self.k*self.dt*self.dt_da) / \
            (-2*pi*np.abs(self.dt_da))
        np.fill_diagonal(K1, K1_diagonal)
        np.fill_diagonal(K2, K2_diagonal)

        ### the second part is the singular part of the matrix. ###

        for m in range(1, self.n_boundaries):

            m_start, m_end = self.indices_of_boundary[m]

            t_minus_z = (self.t -
                         self.boundaries[m].z)[:, newaxis]
            dt = self.boundaries[m].dt[newaxis, :]

            K1[:, m_start:m_end] += \
                1j*np.conj(dt)/np.conj(t_minus_z) + \
                2*np.abs(dt)*np.log(np.abs(t_minus_z))

            K2[:, m_start:m_end] += \
                -1j*dt/np.conj(t_minus_z) + \
                np.abs(dt)*t_minus_z/np.conj(t_minus_z)

        self.A = LinearOperator(matvec=MatVec(K1, K2),
                                dtype=np.float64,
                                shape=(2*self.n_pts, 2*self.n_pts))

    def compute_omega(self, H, tol=None,max_iter=None):

        tol = GMRES_TOL if tol is None else tol
        max_iter = GMRES_MAX_ITER if max_iter is None else max_iter
        b = concatenate((H.real, H.imag))

        omega_sep, _ = gmres(self.A, b, atol=0, tol=tol, restart=RESTART, maxiter=max_iter)

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

        if self.n_flows == 1:
            n_jobs = 1

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

    # def phi(self, z, omega, fmm=True):
    #     assert z.ndim == 1

    #     if fmm: return self.fmm.phi(z.real, z.imag, omega)

    #     regular_part = np.sum((omega * self.dt)[newaxis, :] /
    #                         (self.t[newaxis, :] - z[:, newaxis]), axis=1) / (2j * pi)

    #     singular_part = np.zeros_like(regular_part, dtype=np.complex128)

    #     for k in range(1, self.n_boundaries):

    #         start, end = self.indices_of_boundary[k]
    #         Ck = np.sum(omega[start:end] * np.abs(self.dt[start:end]))
    #         zk = self.boundaries[k].z

    #         singular_part += Ck * np.log(z-zk)

    #     return regular_part + singular_part

    def d_phi(self, z, omega, fmm=True):
        assert z.ndim == 1
        assert fmm
        if fmm: return self.fmm.d_phi(z.real, z.imag, omega)
        
        regular_part = np.sum((omega * self.dt)[newaxis, :] /
                              (self.t[newaxis, :] - z[:, newaxis])**2, axis=1) / (2j * pi)

        singular_part = np.zeros_like(regular_part, dtype=np.complex128)

        for k in range(1, self.n_boundaries):
            start, end = self.indices_of_boundary[k]
            Ck = np.sum(omega[start:end] * np.abs(self.dt[start:end]))
            zk = self.boundaries[k].z
            singular_part += Ck/(z-zk)

        return regular_part + singular_part

    # def psi(self, z, omega, fmm=True):
    #     assert z.ndim == 1

    #     if fmm: return self.fmm.psi(z.real, z.imag, omega)
        
    #     first_term = np.sum(
    #         np.real((conjugate(omega) * self.dt)[newaxis, :])
    #         / (self.t[newaxis, :] - z[:, newaxis]),
    #         axis=1) / (1j*pi)

    #     second_term = np.sum(
    #         (conjugate(self.t) * omega * self.dt)[newaxis, :]
    #         / (self.t[newaxis, :] - z[:, newaxis])**2,
    #         axis=1) / (-2j * pi)

    #     regular_part = first_term + second_term

    #     singular_part = np.zeros_like(regular_part, dtype=np.complex128)

    #     for k in range(1, self.n_boundaries):
    #         start, end = self.indices_of_boundary[k]
    #         Ck = np.sum(omega[start:end] * np.abs(self.dt[start:end]))
    #         zk = self.boundaries[k].z
    #         bk = -2*np.sum((omega[start:end] *
    #                        np.conj(self.dt[start:end])).imag)
    #         singular_part += np.conj(Ck)*np.log(z-zk) + \
    #             (bk - Ck*np.conj(zk))/(z-zk)

    #     return regular_part + singular_part

    def velocity(self, x, y, omega, fmm=True):

        z = x + 1j*y
        assert isinstance(z, ndarray)
        assert z.ndim == 1
        assert fmm
        
        return H2U(self.fmm.velocity(x, y, omega))

    def pressure_and_vorticity(self, x, y, omega, fmm=True):

        z = x + 1j*y
        assert fmm
        assert (isinstance(z, ndarray))
        assert (z.ndim == 1)

        d_phi = self.d_phi(z, omega, fmm)
        d_phi_init = self.d_phi(
            array([self.lets[0].matching_pt[0]+1j*self.lets[0].matching_pt[1]]), omega, fmm)

        # this serves an normalization purpose.
        pressure = np.imag(d_phi) - np.imag(d_phi_init)
        vorticity = np.real(d_phi)

        return pressure, vorticity

    def pressure(self, x, y, omega, fmm=True):
        return self.pressure_and_vorticity(x, y, omega,fmm)[0]

    def vorticity(self, x, y, omega, fmm=True):
        return self.pressure_and_vorticity(x, y, omega,fmm)[1]

    def build_plotting_data(self, xs, ys, interior,fmm=True):

        # TODO assert xs, ys are inside the domain.
        # this could be waited on merging the multiply_connected_pipe with the regular pipe. 
        

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

            velocity = self.velocity(xs[interior], ys[interior], omega, fmm)
            pressure, vorticity = \
                self.pressure_and_vorticity(xs[interior], ys[interior], omega, fmm)
            base_pressure = self.pressure_and_vorticity(base_x, base_y, omega, fmm)[0][0]

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

    def build_pressure_drops(self):
        pts = array([let.matching_pt for let in self.lets])
        pressure_drops = []
        for omega in self.omegas:
            pressure = self.pressure(pts[:, 0], pts[:, 1], omega,fmm=False)
            pressure_drops.append(pressure[1:] - pressure[0])

        self.pressure_drops = array(pressure_drops)
        

        
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

    def clear_geometry(self):
        self.boundaries = [b.clean_copy() for b in self.boundaries]
        
        