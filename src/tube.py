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

sys.path.append('.')
from curve import *
from utility_and_spec import *
from scipy.sparse.linalg import gmres, LinearOperator
from typing import List, Tuple
from itertools import chain
from joblib import Parallel, delayed

from joblib import Parallel, delayed, cpu_count

class Pipe:
    # geometric data.
    curves: List[Curve]

    # graph data.
    lets: List[int]  # stands for inlets and outlets
    flows: List[Tuple[int,int]]
    outlets: List[int]
    inlet: int

    # solver data
    A: LinearOperator
    omegas: List[np.ndarray]
    pressure_drops: np.ndarray
    velocity_field: List[np.ndarray]
    pressure: List[np.ndarray]
    vorticity: List[np.ndarray]

    def __init__(self,grid_density=100) -> None:
        self.grid_density = grid_density

    @property
    def boundary(self):
        pts = []
        for c in self.curves:
            if isinstance(c,Cap):
                pts += [c.start_pt, c.matching_pt]
            elif isinstance(c,Line):
                pts += [c.start_pt, c.mid_pt]
            elif isinstance(c,Corner):
                pts += [[c.x[i], c.y[i]] for i in range(len(c.a))]
        pts = np.array(pts)
    
    @property
    def extent(self):
        left = np.min(self.boundary[:,0])
        right = np.max(self.boundary[:,0])
        up = np.max(self.boundary[:,0])
        down = np.min(self.boundary[:,0])
        return (left,right,up,down)
    
    @property
    def grid(self):
        left,right,up,down = self.extent
        n = np.sqrt(self.grid_density)
        xs = np.linspace(left,right,np.ceil(n*(right-left)))
        ys = np.linspace(down,up,np.ceil(n*(up-down)))
        xs,ys = np.meshgrid(xs,ys)
        
        
        
    
    @property
    def grid(self):
        pass

    @property
    def a(self): return np.concatenate([c.a + 2 * i for i, c in enumerate(self.curves)])
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
    def k(self): return np.concatenate([c.k for c in self.curves])
    @property
    def panels(self): 
        return chain(*([ c.panels for c in self.curves]))
        

    def build_geometry(self, max_distance=None, legendre_ratio=None, n_jobs=1):
        
        n_jobs = min(n_jobs,cpu_count(),8, len(self.curves))
        
        if n_jobs == 1:
            for c in self.curves:
                c.build_affine_transform()
                c.build(max_distance,legendre_ratio)
        else:
            Parallel(n_jobs=n_jobs) (
                delayed(lambda c: c.build_affine_transform(), c.build(max_distance,legendre_ratio)) 
            )
            

    def build_graph(self):
        self.lets = [i for i,c in enumerate(self.curves) if isinstance(c,Cap)]
        assert len(self.lets) > 1

        self.inlet = self.lets[0]
        self.outlets = self.lets[1:]
        self.flows = [(self.inlet,o) for o in self.outlets]
        self.nflows = len(self.flows)

    # noinspection PyPep8Naming,DuplicatedCode
    def build_A(self):
        """not implementing fmm for now. """

        da = self.da
        t = self.t
        dt_da = self.dt_da
        k = self.k

        dt = t[:, np.newaxis] - t[np.newaxis, :]
        d = dt_da[np.newaxis, :]
        da_ = da[np.newaxis, :]

        # this ignores the error for computing the diagonal elements with 0/0 error
        with np.errstate(divide='ignore', invalid='ignore'):
            K1 = -da_ * np.imag(d/dt) / np.pi
            K2 = -da_ * (-d/np.conjugate(dt) + np.conjugate(d)
                         * dt/(np.conjugate(dt**2))) / (2j*np.pi)

        # now we need to fill the diagonal elements
        d = dt_da
        K1_diagonal = k*np.abs(d)*da/(2*np.pi)
        K2_diagonal = -da*k*(d**2)/(np.abs(d)*2*np.pi)
        np.fill_diagonal(K1, K1_diagonal)
        np.fill_diagonal(K2, K2_diagonal)

        self.K1 = K1
        self.K2 = K2

        def A(omega_sep):
            assert len(omega_sep)%2 == 0
            n = len(omega_sep) // 2
            omega = omega_sep[:n] + 1j*omega_sep[n:]
            h = omega + K1@omega + K2@omega
            h_sep = np.concatenate([h.real,h.imag])
            return h_sep

        n = len(self.a)
        self.A = LinearOperator(dtype=np.float64, shape=(2*n,2*n),matvec=A)

    def compute_omega(self,U,tol=1e-13):
        H = U2H(U)
        b = np.concatenate((H.real,H.imag))

        omega_sep,_ = gmres(self.A, b,atol=0,tol=tol)

        if _ < 0:
            warnings.warn("gmres is not converging to tolerance. ")

        n = len(omega_sep)
        omega = omega_sep[:n//2] + 1j*omega_sep[n//2:]

        return omega

    def get_bounadry_velocity_condition(self,i):
        inlet,outlet = self.flows[i]
        ret = []
        for j,c in enumerate(self.curves):
            if j == inlet:
                ret.append(-c.boundary_velocity())
            elif j == outlet:
                ret.append(c.boundary_velocity())
            else:
                ret.append(0*c.boundary_velocity())
        return np.concatenate(ret)

    def solve(self,tol=None):

        tol = 1e-12 if tol is None else tol

        
        self.omegas = Parallel(n_jobs=min(4,self.nflows))(delayed(
            lambda i: self.compute_omega(self.get_bounadry_velocity_condition(i),tol)
            )(i) for i in range(self.nflows))


    # noinspection DuplicatedCode
    def compute_velocity(self,x,y,omega):
        t = self.t
        dt = self.dt_da * self.da
        z = x + 1j*y
        assert isinstance(z, np.ndarray)
        shape = z.shape
        z = z.flatten()

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
        ret = ret.reshape(shape)
        return H2U(ret)

    # noinspection DuplicatedCode
    def compute_pressure_and_vorticity(self,x,y,omega):
        z = x + 1j*y
        assert (isinstance(z, np.ndarray))
        shape = z.shape
        z = z.flatten()

        t = self.t
        dt = self.dt_da * self.da

        t_minus_z_sq = (t[np.newaxis, :] - z[:, np.newaxis]) ** 2
        d_phi = np.sum((omega * dt)[np.newaxis, :] /
                       t_minus_z_sq, axis=1) / (2j * np.pi)

        pressure = np.imag(d_phi)
        vorticity = np.real(d_phi)
        return pressure.reshape(shape), vorticity.reshape(shape)


    def build_pressure_drops(self):
        pass
    
    def build_velocity_fields(self):
        pass
    
    def build(self):
        pass
    
    def is_inside(self, z):
        pass
#
# class StraightPipe(Pipe):
#     def __init__(self,p1,p2,r=1) -> None:
#         """
#         this creates a simple tube. Why do I create it first? because it serves well as
#         a template for other more sophisticated geometries.
#         """
#         super().__init__()
#
#
#         self.angle = np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
#         theta = self.angle + np.pi/2
#         top_left = p1 + r*np.array([np.cos(theta),np.sin(theta)])
#         bottom_left = p1 - r*np.array([np.cos(theta),np.sin(theta)])
#         top_right = p2 + r*np.array([np.cos(theta),np.sin(theta)])
#         bottom_right = p2 - r*np.array([np.cos(theta),np.sin(theta)])
#
#         top_line = Line(top_left,top_right)
#         right_line = Line(top_right,bottom_right)
#         bottom_line = Line(bottom_right,bottom_left)
#         left_line = Line(bottom_left,top_left)
#
#         self.curves = [top_line,right_line,bottom_line,left_line]
#
#         up = np.max(top_left[1],top_right[1],bottom_left[1],bottom_right[1])
#         low = np.min(top_left[1],top_right[1],bottom_left[1],bottom_right[1])
#         left = np.min(top_left[0],top_right[0],bottom_left[0],bottom_right[0])
#         right = np.max(top_left[0],top_right[0],bottom_left[0],bottom_right[0])
#

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
            raise TypeError('invalid Curve type, only Line and Cap are permitted here. ')

        self.curves = []
        self.corner_size = corner_size

        n = len(lines)
        for i in range(n):

            j = (i + 1) % len(lines)
            if lines[i] == Line:
                self.curves.append(lines[i](points[i], points[j]))
            else: # I need to inference the mid point.

                vec = points[i] - points[i-1]
                vec = vec/np.linalg.norm(vec)
                mid_pt = (points[i] + points[j])/2 + vec
                self.curves.append(lines[i](points[i], points[j],mid_pt))

        self.smooth_corners()

    def smooth_corners(self):

        i = self.next_corner()

        while i is not None:

            l1 = self.curves.pop(i)
            l2 = self.curves.pop(i)

            p = l1.start_pt
            q = l1.end_pt
            r = l2.end_pt

            corner_size = min(self.corner_size, np.linalg.norm(p - q) / 2, np.linalg.norm(r - q) / 2)
            assert (corner_size > 1e-2)

            start_pt = q + (((p - q) / np.linalg.norm(p - q)) * corner_size)
            end_pt = q + (((r - q) / np.linalg.norm(r - q)) * corner_size)

            self.curves.insert(i,Line(end_pt,r))
            self.curves.insert(i,Corner(start_pt, end_pt, q))
            self.curves.insert(i,Line(p,start_pt))

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


class NLets(SmoothPipe):
    def __init__(self,ls,rs, corner_size=1e-1):

        assert len(ls) == len(rs)
        assert np.all(rs > 0)

        thetas = np.arctan2(ls[:,1], ls[:,0])
        thetas[thetas==np.pi] = -np.pi

        assert np.all(np.diff(thetas) > 0)

        n = len(ls)

        pts = []
        curves = []

        for i in range(n):
            j = (i + 1)%n
            tangential_dir = (thetas[i] + np.pi/2)
            x = np.cos(tangential_dir)
            y = np.sin(tangential_dir)
            tangential_unit = pt(x,y)

            p1 = ls[i] - tangential_unit*rs[i]
            p2 = ls[i] + tangential_unit*rs[i]

            tangential_dir = (thetas[j] + np.pi/2)
            x = np.cos(tangential_dir)
            y = np.sin(tangential_dir)
            tangential_unit = pt(x, y)
            q1 = ls[j] - tangential_unit*rs[j]

            p3 = line_intersect(p2,p2+ls[i],q1,q1+ls[j])

            pts = pts + [p1,p2,p3]
            curves = curves + [Cap,Line,Line]

        super().__init__(pts,curves, corner_size)
