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
from curve import *
from utility_and_spec import *
import warnings
from joblib import Parallel, delayed, cpu_count
from typing import List, Tuple
from scipy.sparse.linalg import gmres, LinearOperator
from matplotlib.path import Path
from shapely.geometry import LineString
from numpy import ndarray, concatenate, pi, conjugate, array, newaxis
from numpy.linalg import norm


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
    flows: List[Tuple[int, int]]    # [(inlet_index,outlet_index),...]
    n_flows: int            # number of flows
    omegas: ndarray         # shape=(n_flows, n_pts), dtype=complex128
    pressure_drops: ndarray  # shape=(nfluxes, nflows), dtype=float64

    # picture data
    boundary: ndarray           # shape=(*, 2), dtype=float64.
    interior_boundary: ndarray  # shape=(*, 2), dtype=float64.
    closed_boundary: ndarray    # shape=(*, 2), dtype=float64.
    closed_interior_boundary:ndarray     # shape=(*, 2), dtype=float64.
    extent: Tuple[float,float,float,float] # (xmin, xmax, ymin, ymax)
    
    velocity_field: ndarray      # TODO
    pressure_field: ndarray      # TODO
    vorticity_field: ndarray     # TODO

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

    def build(self):
        self.build_geometry()
        self.build_A()
        self.build_all_boundary_velocity_conditions()
        self.build_omegas()
        self.build_pressure_drops()

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

        def A(omega_sep):
            omega = omega_sep[:self.n_pts] + 1j*omega_sep[self.n_pts:]
            h = omega + K1@omega + K2@(omega.conjugate())
            return concatenate([h.real, h.imag])

        self.A = LinearOperator(
            matvec=A,
            dtype=np.float64,
            shape=(2*self.n_pts, 2*self.n_pts))

    def compute_omega(self, H, tol=None):

        tol = 1e-12 if tol is None else tol
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
    def flows(self): return [(0, i) for i in range(self.n_lets - 1)]
    @property
    def n_flows(self): return self.n_lets - 1

    def boundary_value(self, i):
        """Flow goes in the inlet and out the outlet. """
        inlet, outlet = self.flows[i]
        i = self.let_index2curve_index[inlet]
        o = self.let_index2curve_index[outlet]
        ret = []
        for j, c in enumerate(self.curves):
            if j == i:
                ret.append(-c.boundary_velocity())
            elif j == o:
                ret.append(c.boundary_velocity())
            else:
                ret.append(np.zeros_like(c.a))
        return U2H(concatenate(ret))

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
        pts = array(
            [let.matching_pt for let in self.lets])
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
        shape = z.shape
        z = z.flatten()

        return H2U((self.phi(z, omega) + z * conjugate(self.d_phi(z, omega)) + conjugate(self.psi(z, omega))).reshape(shape))

    def pressure_and_vorticity(self, x, y, omega):

        # TODO : not verified yet.
        z = x + 1j*y
        assert (isinstance(z, ndarray))
        shape = z.shape
        z = z.flatten()

        d_phi = self.d_phi(z, omega)
        pressure = np.imag(d_phi)
        vorticity = np.real(d_phi)
        return pressure.reshape(shape), vorticity.reshape(shape)

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
    def closed_boundary(self):
        return concatenate((self.boundary, self.boundary[:1]))

    def interior_boundary(self, distance=None):
        
        distance = 0.06 if distance is None else distance
        
        x, y = LineString(concatenate((self.boundary, self.boundary[:1]))).buffer(
            distance).interiors[0].coords.xy
        return array([x, y]).T[:-1]


    @property
    def extent(self):
        left = np.min(self.boundary[:, 0])
        right = np.max(self.boundary[:, 0])
        bottom = np.min(self.boundary[:, 1])
        top = np.max(self.boundary[:, 1])
        return (left, right, bottom, top)

    def contains_points(self, x, y):
        return Path(self.boundary).contains_points(array([x, y]).T)

    def contains_points_interior(self, x, y, distance=None):
        return Path(self.interior_boundary(distance)).contains_points(array([x, y]).T)

    def grid(self, density=100):
        left, right, bottom, top = self.extent
        nx = np.ceil((right - left) * density).astype(int)
        ny = np.ceil((top - bottom) * density).astype(int)

        xs = np.linspace(left, right, nx)
        ys = np.linspace(bottom, top, ny)

        xs, ys = np.meshgrid(xs, ys)
        shape = xs.shape

        mask1 = self.contains_points(xs.flatten(), ys.flatten()).reshape(shape)
        mask2 = self.contains_points_interior(
            xs.flatten(), ys.flatten()).reshape(shape)

        return xs, ys, mask1, mask2

    def build_plotting_data(self):
        # TODO
        pass
        xs, ys, mask = self.grid()
        u, v = self.velocity(xs[mask], ys[mask], self.omegas[0])


class StraightPipe(Pipe):
    # TODO
    pass


class SmoothPipe(Pipe):
    """
    given points
        p1, p2, p3, ... , pn
    and the description of the Line
        l1 = p1-p2
        l2 = p2-p3
        ...
        ln = pn-p1
        the descriptions can be:
            - Line
            - Cap
    this class should firstly create a cornered Curve with
    the given specification of points and lines,
    and then it should automatically smooth the corners.
    """

    def __init__(self, points, lines, corner_size=1e-1) -> None:

        super().__init__()
        assert len(points) == len(lines)

        if not np.all([i in [Line, Cap] for i in lines]):
            raise TypeError(
                'invalid Curve type, only Line and Cap are permitted here. ')

        self.curves = []
        self.corner_size = corner_size

        n = len(lines)
        for i in range(n):

            j = (i + 1) % len(lines)
            if lines[i] == Line:
                self.curves.append(lines[i](points[i], points[j]))
            else:  # I need to inference the mid point.

                vec = points[i] - points[i-1]
                vec = vec/norm(vec)
                mid_pt = (points[i] + points[j])/2 + vec
                self.curves.append(lines[i](points[i], points[j], mid_pt))

        self.smooth_corners()

    def smooth_corners(self):

        i = self.next_corner()

        while i is not None:

            l1 = self.curves.pop(i)
            l2 = self.curves.pop(i)

            p = l1.start_pt
            q = l1.end_pt
            r = l2.end_pt

            corner_size = min(self.corner_size, norm(
                p - q) / 2, norm(r - q) / 2)
            assert (corner_size > 1e-2)

            start_pt = q + (((p - q) / norm(p - q)) * corner_size)
            end_pt = q + (((r - q) / norm(r - q)) * corner_size)

            self.curves.insert(i, Line(end_pt, r))
            self.curves.insert(i, Corner(start_pt, end_pt, q))
            self.curves.insert(i, Line(p, start_pt))

            i = self.next_corner()

    def next_corner(self):
        """
        if there are two consecutive Line, they will have a Corner.
        this function return the index of the lines.
        """

        for i in range(len(self.curves)):
            j = (i + 1) % len(self.curves)
            if isinstance(self.curves[i], Line) and isinstance(self.curves[j], Line):
                return i
        return None


class NLets(SmoothPipe):
    def __init__(self, ls, rs, corner_size=1e-1):

        assert len(ls) == len(rs)
        assert np.all(rs > 0)

        thetas = np.arctan2(ls[:, 1], ls[:, 0])
        thetas[thetas == pi] = -pi

        assert np.all(np.diff(thetas) > 0)

        n = len(ls)

        pts = []
        curves = []

        for i in range(n):
            j = (i + 1) % n
            tangential_dir = (thetas[i] + pi/2)
            x = np.cos(tangential_dir)
            y = np.sin(tangential_dir)
            tangential_unit = pt(x, y)

            p1 = ls[i] - tangential_unit*rs[i]
            p2 = ls[i] + tangential_unit*rs[i]

            tangential_dir = (thetas[j] + pi/2)
            x = np.cos(tangential_dir)
            y = np.sin(tangential_dir)
            tangential_unit = pt(x, y)
            q1 = ls[j] - tangential_unit*rs[j]

            p3 = line_intersect(p2, p2+ls[i], q1, q1+ls[j])

            pts = pts + [p1, p2, p3]
            curves = curves + [Cap, Line, Line]

        super().__init__(pts, curves, corner_size)


class Cross(NLets):
    def __init__(self, length, radius, corner_size=0.2):

        l1 = pt(-length, 0)
        l2 = pt(0, -length)
        l3 = pt(length, 0)
        l4 = pt(0, length)
        ls = array([l1, l2, l3, l4])
        rs = array([radius, radius, radius, radius])

        super().__init__(ls, rs, corner_size)
