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
from joblib import Parallel, delayed, cpu_count
from itertools import chain
from typing import List, Tuple
from scipy.sparse.linalg import gmres, LinearOperator
from matplotlib.path import Path

sys.path.append('.')
from utility_and_spec import *
from curve import *
nax = np.newaxis

class Pipe:
    # geometric data.
    curves: List[Curve]

    # solver data
    lets: List[int]  # stands for inlets/outlets
    flows: List[Tuple[int, int]]
    A: LinearOperator
    omegas: List[np.ndarray]
    pressure_drops: np.ndarray

    # picture data
    velocity_field: List[np.ndarray]
    pressure: List[np.ndarray]
    vorticity: List[np.ndarray]

    def __init__(self) -> None:
        # self.grid_density = grid_density
        pass

    ### GEOMETRIC ###

    @property
    def a(self): return np.concatenate([c.a + 2*i for i,c in enumerate(self.curves)])
    @property
    def da(self): return np.concatenate([c.da for c in self.curves])
    @property
    def x(self): return np.concatenate([c.x for c in self.curves])
    @property
    def y(self): return np.concatenate([c.y for c in self.curves])
    @property
    def dx_da(self): return np.concatenate([c.dx_da for c in self.curves])
    @property
    def dy_da(self): return np.concatenate([c.dy_da for c in self.curves])
    @property
    def ddx_dda(self): return np.concatenate([c.ddx_dda for c in self.curves])
    @property
    def ddy_dda(self): return np.concatenate([c.ddy_dda for c in self.curves])
    @property
    def t(self): return np.concatenate([c.t for c in self.curves])
    @property
    def dt_da(self): return np.concatenate([c.dt_da for c in self.curves])
    @property
    def dt(self): return self.dt_da * self.da
    @property
    def k(self): return np.concatenate([c.k for c in self.curves])
    @property
    def panels(self): return chain(*([c.panels for c in self.curves]))

    def build_geometry(self, max_distance=None, legendre_ratio=None,n_jobs=1):
        
        if n_jobs == 1:
            for c in self.curves:
                c.build(max_distance, legendre_ratio)
            return 
        
        def build_curve(c):
            c.build(max_distance,legendre_ratio)
            return c
        
        self.curves = Parallel(n_jobs=min(n_jobs, len(self.curves), cpu_count()//2))(
            delayed(build_curve)(c) for c in self.curves)

    ### SINGLE SOLVER ###

    def build_A(self, fmm=False):

        if fmm: return NotImplemented

        da = self.da
        k = self.k

        dt = self.t[:, nax] - self.t[nax, :]
        d = self.dt_da[nax, :]
        da_ = da[nax, :]

        # this ignores the error for computing the diagonal elements with 0/0 error
        with np.errstate(divide='ignore', invalid='ignore'):
            K1 = -da_ * np.imag(d/dt) / np.pi
            K2 = -da_ * (-d/np.conjugate(dt) + np.conjugate(d)
                         * dt/(np.conjugate(dt**2))) / (2j*np.pi)

        # now we need to fill the diagonal elements
        d = self.dt_da
        K1_diagonal = k*np.abs(d)*da/(2*np.pi)
        K2_diagonal = -da*k*(d**2)/(np.abs(d)*2*np.pi)
        np.fill_diagonal(K1, K1_diagonal)
        np.fill_diagonal(K2, K2_diagonal)

        self.K1 = K1
        self.K2 = K2
        
        n = len(self.a)
        
        def A(omega_sep):
            assert len(omega_sep) == 2*n
            omega = omega_sep[:n] + 1j*omega_sep[n:]
            h = omega + K1@omega + K2@(omega.conjugate())
            h_sep = np.concatenate([h.real, h.imag])
            return h_sep

        self.A = LinearOperator(dtype=np.float64, shape=(2*n, 2*n), matvec=A)

    def compute_omega(self, U, tol=None):

        tol = 1e-12 if tol is None else tol
        H = U2H(U)
        b = np.concatenate((H.real, H.imag))

        omega_sep, _ = gmres(self.A, b, atol=0, tol=tol)

        if _ < 0:
            warnings.warn("gmres is not converging to tolerance. ")

        n = len(H)
        omega = omega_sep[:n] + 1j*omega_sep[n:]

        return omega

    ### SOLVE FOR FLOWS ###

    @property
    def lets(self):
        ret = [i for i, c in enumerate(self.curves) if isinstance(c, Cap)]
        if not ret:
            raise ValueError("There is no flow for this pipe. ")
        return ret
    @property
    def flows(self): return [(self.lets[0], o) for o in self.lets[1:]]
    @property
    def nflows(self): return len(self.flows)

    def get_bounadry_velocity_condition(self, i):
        inlet, outlet = self.flows[i]
        ret = []
        for j, c in enumerate(self.curves):
            if j == inlet:
                ret.append(-c.boundary_velocity())
            elif j == outlet:
                ret.append(c.boundary_velocity())
            else:
                ret.append(0*c.boundary_velocity())
        return np.concatenate(ret)

    def solve(self, tol=None, n_jobs=1):

        self.omegas = Parallel(n_jobs=min(n_jobs, self.nflows, cpu_count()//2))(delayed(
            lambda i:
                self.compute_omega(self.get_bounadry_velocity_condition(i), tol))
            (i) for i in range(self.nflows))

    ### COMPUTE PHYSICS QUANTITIES ###

    def phi(self, z, omega):
        assert z.ndim == 1
        return np.sum((omega * self.dt)[nax, :] /
                      (self.t[nax, :] - z[:, nax]), axis=1) / (2j * np.pi)

    def d_phi(self, z, omega):
        assert z.ndim == 1
        return np.sum((omega * self.dt)[nax, :] /
                      (self.t[nax, :] - z[:, nax])**2, axis=1) / (2j * np.pi)

    def psi(self, z, omega):
        assert z.ndim == 1
        
        first_term = np.sum(
            np.real((np.conjugate(omega) * self.dt)[nax, :])
            / (self.t[nax, :] - z[:, nax]),
            axis=1) / (1j*np.pi)

        second_term = np.sum(
            (np.conjugate(self.t) * omega * self.dt)[nax, :]
            / (self.t[nax, :] - z[:, nax])**2,
            axis=1) / (-2j * np.pi)

        return first_term + second_term

    def velocity(self, x, y, omega):

        z = x + 1j*y
        assert isinstance(z, np.ndarray)
        shape = z.shape
        z = z.flatten()

        return H2U((self.phi(z,omega) + z * np.conjugate(self.d_phi(z, omega)) + np.conjugate(self.psi(z, omega))).reshape(shape))

    def pressure_and_vorticity(self, x, y, omega):
        
        # TODO : not verified yet. 
        z = x + 1j*y
        assert (isinstance(z, np.ndarray))
        shape = z.shape
        z = z.flatten()

        d_phi = self.d_phi(z,omega)
        pressure = np.imag(d_phi)
        vorticity = np.real(d_phi)
        return pressure.reshape(shape), vorticity.reshape(shape)
    
    
    def pressure(self,x,y,omega):
        return self.pressure_and_vorticity(x,y,omega)[0]
    
    def vorticity(self,x,y,omega):
        return self.pressure_and_vorticity(x,y,omega)[1]
    
    ### CONNECTING SOLVER ###
    
    def build_pressure_drops(self):    
        # TODO
        pass
    
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
        return np.array(pts)

    @property
    def extent(self):
        left = np.min(self.boundary[:, 0])
        right = np.max(self.boundary[:, 0])
        bottom = np.min(self.boundary[:, 1])
        top = np.max(self.boundary[:, 1])
        return (left, right, bottom, top)
    
    @property
    def as_polygon(self):
        return Path(self.boundary)
    
    def contains_points(self,x,y):
        return self.as_polygon.contains_points(np.array([x,y]).T)
    
    def grid(self,density=100):
        left, right, bottom, top = self.extent
        nx = np.ceil((right - left) * density).astype(int)
        ny = np.ceil((top - bottom) * density).astype(int)
        
        xs = np.linspace(left, right, nx)
        ys = np.linspace(bottom, top, ny)
        shape = xs.shape
        
        xs, ys = np.meshgrid(xs, ys)
        
        mask = self.contains_points(np.column_stack((xs.flatten(),ys.flatten()))).reshape(shape)
        
        return xs, ys, mask
    
    def build_plotting_data(self):
        xs,ys,mask = self.grid()
        u,v = self.velocity(xs,ys,self.omegas[0])
        # TODO
        pass
    
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
                vec = vec/np.linalg.norm(vec)
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

            corner_size = min(self.corner_size, np.linalg.norm(
                p - q) / 2, np.linalg.norm(r - q) / 2)
            assert (corner_size > 1e-2)

            start_pt = q + (((p - q) / np.linalg.norm(p - q)) * corner_size)
            end_pt = q + (((r - q) / np.linalg.norm(r - q)) * corner_size)

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
        thetas[thetas == np.pi] = -np.pi

        assert np.all(np.diff(thetas) > 0)

        n = len(ls)

        pts = []
        curves = []

        for i in range(n):
            j = (i + 1) % n
            tangential_dir = (thetas[i] + np.pi/2)
            x = np.cos(tangential_dir)
            y = np.sin(tangential_dir)
            tangential_unit = pt(x, y)

            p1 = ls[i] - tangential_unit*rs[i]
            p2 = ls[i] + tangential_unit*rs[i]

            tangential_dir = (thetas[j] + np.pi/2)
            x = np.cos(tangential_dir)
            y = np.sin(tangential_dir)
            tangential_unit = pt(x, y)
            q1 = ls[j] - tangential_unit*rs[j]

            p3 = line_intersect(p2, p2+ls[i], q1, q1+ls[j])

            pts = pts + [p1, p2, p3]
            curves = curves + [Cap, Line, Line]

        super().__init__(pts, curves, corner_size)

class Cross(SmoothPipe):
    def __init__(self, length, radius, corner_size=0.2):

        p1 = np.array([-length, -radius])
        p2 = np.array([-radius, -radius])
        p3 = np.array([-radius, -length])
        p4 = np.array([radius, -length])
        p5 = np.array([radius, -radius])
        p6 = np.array([length, -radius])
        p7 = np.array([length, radius])
        p8 = np.array([radius, radius])
        p9 = np.array([radius, length])
        p10 = np.array([-radius, length])
        p11 = np.array([-radius, radius])
        p12 = np.array([-length, radius])

        points = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
        curves = [Line, Line, Cap,
                  Line, Line, Cap,
                  Line, Line, Cap,
                  Line, Line, Cap]

        super().__init__(points, curves, corner_size)

    # def get_flows(self):
    #     self.caps_index = [i for i, j in enumerate(self.curves) if isinstance(j, Cap)]
    #     self.inflow = self.caps[0]
    #     self.outflows = self.caps[1:]
    #
    # def get_all_boundary_velocity_conditions(self):
    #
    #     velocities = []
    #
    #     for j in self.outflows:
    #         velocity = []
    #         for i, c in enumerate(self.curves):
    #             if i == self.inflow:
    #                 velocity.append(c.get_boundary_velocity_condition(c.get_velocity(flux=1)))
    #             elif i == j:
    #                 velocity.append(c.get_boundary_velocity_condition(c.get_velocity(flux=-1)))
    #             else:
    #                 velocity.append(np.zeros_like(c.a))
    #         velocities.append(np.concatenate(velocity))
    #
    #     self.velocities = np.array(velocities)
    #
    # def compute_pressure_drops(self):
    #     pressure_drops = []
    #
    #     for i, o in enumerate(self.outflows):
    #         omega = self.omegas[i]
    #         pressure_drop = []
    #
    #         p1 = Cross.curves[self.inflow].p
    #         p1_cplx = p1[0] + 1j * p1[1]
    #         p1_pressure = self.solver.compute_pressure(p1_cplx, omega)
    #
    #         for j, o2 in enumerate(self.outflows):
    #             p2 = Cross.curves[o2].p
    #             p2_cplx = p2[0] + 1j * p2[1]
    #             p2_pressure = self.solver.compute_pressure(p2_cplx, omega)
    #             pressure_drop.append(p2_pressure - p1_pressure)
    #
    #         pressure_drops.append(pressure_drop)
    #
    #     self.pressure_drops = pressure_drops
    #
    #
    #
    #
    #
    #
