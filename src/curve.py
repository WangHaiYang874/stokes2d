import numpy as np

from utility_and_spec import *
from linear_transform import affine_transformation
from math_functions import *

class Panel:
    """
    each Panel is designed by
        - a 16 points gauss quadrature rule,
        - have a max distance of 1.
    """

    def __init__(self, a, da, x, y, domain) -> None:
        self.a = a
        self.da = da
        self.x = x
        self.y = y
        self.domain = domain

        self.dx_da = None
        self.dy_da = None
        self.ddx_dda = None
        self.ddy_dda = None

    def distance(self, pts):
        a = pt(self.x[0], self.y[0])
        b = pt(self.x[-1], self.y[-1])
        return distance(pts, a, b)

    @property
    def t(self):
        return self.x + 1j * self.y

    @property
    def dt_da(self):
        return self.dx_da + 1j * self.dy_da

    @property
    def k(self):
        return (self.dx_da * self.ddy_dda - self.dy_da * self.ddx_dda) / \
               ((self.dx_da ** 2 + self.dy_da ** 2) ** 1.5)

    @property
    def max_distance(self):
        return np.max(np.linalg.norm(np.diff(np.array([self.x, self.y])), axis=0))

    @property
    def legendre_coef_ratio(self):
        legendre_coef = np.polynomial.legendre.Legendre.fit(
            self.a, self.t, deg=len(self.a) - 1, domain=self.domain).coef
        return np.sum(np.abs(legendre_coef[-2:])) / np.sum(np.abs(legendre_coef[:2]))


    def good_enough(self, max_distance=None, legendre_ratio=None,domain_threhold=1e-8):

        if self.domain[1] - self.domain[0] < domain_threhold:
            return True

        max_distance = max_distance if max_distance else 1e-2
        legendre_ratio = legendre_ratio if legendre_ratio else 1e-14

        return self.max_distance < max_distance and self.legendre_coef_ratio < legendre_ratio


class Curve:
    standard_start_pt = None
    standard_mid_pt = None
    standard_end_pt = None
    x_fn = None
    y_fn = None
    dx_da_fn = None
    dy_da_fn = None
    ddx_dda_fn = None
    ddy_dda_fn = None

    def __init__(self, start_pt, end_pt, mid_pt) -> None:

        assert np.linalg.norm(start_pt - end_pt) > 1e-15

        self.panels = []
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.mid_pt = mid_pt

        self.aff_trans = None

    def build(self, max_distance=None, legendre_ratio=None):
        
        # building affine transformation
        self.aff_trans = affine_transformation(
            self.standard_start_pt, self.standard_mid_pt, self.standard_end_pt,
            self.start_pt, self.mid_pt, self.end_pt)
        
        # initialize panel
        if not len(self.panels):
            a, da = gauss_quad_rule()
            x = self.x_fn(a)
            y = self.y_fn(a)
            x, y = self.aff_trans(x, y, with_affine=True)
            p = Panel(a, da, x, y, (-1, 1))
            self.panels.append(p)

        # refine the panels. 
        i = 0
        while i < len(self.panels):
            if self.panels[i].good_enough(max_distance=max_distance, legendre_ratio=legendre_ratio):
                i += 1
                continue
            
            # the panel is not good enough, so we want to refine it. 
            p = self.panels.pop(i)
            p1, p2 = self.split_a_panel(p)
            self.panels.insert(i, p2)
            self.panels.insert(i, p1)

        # calculating other needed quantities of the curve. 
        for p in self.panels:
            dx_da = self.dx_da_fn(p.a)
            dy_da = self.dy_da_fn(p.a)
            ddx_dda = self.ddx_dda_fn(p.a)
            ddy_dda = self.ddy_dda_fn(p.a)

            dx_da, dy_da = self.aff_trans(dx_da, dy_da)
            ddx_dda, ddy_dda = self.aff_trans(ddx_dda, ddy_dda)

            p.dx_da = dx_da
            p.dy_da = dy_da
            p.ddx_dda = ddx_dda
            p.ddy_dda = ddy_dda

    def split_a_panel(self, p):
        left, right = p.domain
        mid = (left + right) / 2.
        domain1 = (left, mid)
        domain2 = (mid, right)

        ret = []

        for domain in [domain1, domain2]:

            a, da = gauss_quad_rule(domain=domain)
            x = self.x_fn(a)
            y = self.y_fn(a)
            x, y = self.aff_trans(x, y, with_affine=True)
            p = Panel(a, da, x, y, domain)

            ret.append(p)

        return ret

    def boundary_velocity(self):
        return np.zeros((len(self.a),2))

    @property
    def a(self):
        return np.concatenate([p.a for p in self.panels])

    @property
    def da(self):
        return np.concatenate([p.da for p in self.panels])

    @property
    def x(self):
        return np.concatenate([p.x for p in self.panels])

    @property
    def y(self):
        return np.concatenate([p.y for p in self.panels])

    @property
    def t(self):
        return np.concatenate([p.t for p in self.panels])

    @property
    def dx_da(self):
        return np.concatenate([p.dx_da for p in self.panels])

    @property
    def dy_da(self):
        return np.concatenate([p.dy_da for p in self.panels])

    @property
    def dt_da(self):
        return np.concatenate([p.dt_da for p in self.panels])

    @property
    def ddx_dda(self):
        return np.concatenate([p.ddx_dda for p in self.panels])

    @property
    def ddy_dda(self):
        return np.concatenate([p.ddy_dda for p in self.panels])

    @property
    def k(self):
        return (self.dx_da * self.ddy_dda - self.dy_da * self.ddx_dda) / ((self.dx_da ** 2 + self.dy_da ** 2) ** 1.5)


class Line(Curve):
    standard_start_pt = pt(-1, 0)
    standard_end_pt = pt(1, 0)
    standard_mid_pt = pt(0, 0)
    x_fn = lambda _, a: a
    y_fn = lambda _, a: a
    dx_da_fn = lambda _, a: np.ones_like(a)
    dy_da_fn = lambda _, a: np.zeros_like(a)
    ddx_dda_fn = lambda _, a: np.zeros_like(a)
    ddy_dda_fn = lambda _, a: np.zeros_like(a)

    def __init__(self, start_pt=pt(-1,0), end_pt=pt(1,0)) -> None:
        super().__init__(start_pt, end_pt, (start_pt + end_pt)/2)


class Cap(Curve):
    standard_start_pt = pt(1, 0)
    standard_mid_pt = pt(0, 2)
    standard_end_pt = pt(-1, 0)
    x_fn = lambda _, a: _Psi(a)
    dx_da_fn = lambda _, a:  _psi(a)
    ddx_dda_fn = lambda _, a: _d_psi(a)
    
    y_fn = lambda _, a:  _int_Psi_normalized(a)
    dy_da_fn = lambda _, a:  _Psi_normalized(a)
    ddy_dda_fn = lambda _, a: _psi_normalized(a)

    def __init__(self, start_pt=pt(1, 0), end_pt=pt(-1, 0), mid_pt=pt(0, 2)) -> None:

        super(Cap, self).__init__(start_pt, end_pt, mid_pt)

        # this provides conditions for matching
        self.matching_pt = (start_pt + end_pt) / 2
        assert np.linalg.norm(self.matching_pt - mid_pt) > 1e-15
        out_vec = mid_pt - self.matching_pt
        self.dir = np.arctan2(out_vec[1], out_vec[0])
        self.diameter = np.linalg.norm(end_pt - start_pt)
        assert self.diameter > 1e-15

    def boundary_velocity(self):
        """
        cap is the only thing that will be used at the inflow/outflow.
        so this function returns the velocity condition of outward unit flux
        """

        r = self.diameter / 2

        t = (self.t - self.matching_pt[0] - self.matching_pt[1] * 1j)
        t = t * np.exp(-1j * (self.dir - np.pi / 2))
        x = t.real
        h = (x ** 2 - r ** 2) * 3 / (4 * r ** 3)
        h = np.exp(1j * (self.dir - np.pi / 2)) * h

        return H2U(h)

    def contour_polygon(self): return np.array([self.start_pt])
    

class Corner(Curve):
    standard_start_pt = pt(-1, 1)
    standard_mid_pt = pt(0, 0)
    standard_end_pt = pt(1, 1)
    x_fn = lambda _, a: a
    y_fn = lambda _, a: np.array([convoluted_abs(x) for x in a])
    dx_da_fn = lambda _, a: np.ones_like(a)
    dy_da_fn = lambda _, a: np.array([d_convoluted_abs(x) for x in a])
    ddx_dda_fn = lambda _, a: np.zeros_like(a)
    ddy_dda_fn = lambda _, a: np.array([dd_convoluted_abs(x) for x in a])

    def __init__(self, start_pt=pt(-1,1), end_pt=pt(1,1), mid_pt=pt(0,0)):

        assert (np.linalg.norm(start_pt - mid_pt) > 0)
        assert (np.abs(np.linalg.norm(start_pt - mid_pt) - np.linalg.norm(end_pt - mid_pt)) < 1e-15)
        
        super(Corner, self).__init__(start_pt, end_pt, mid_pt)
