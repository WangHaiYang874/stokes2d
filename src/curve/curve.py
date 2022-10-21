import copy
from utils import *
from .panel import Panel
from curve.linear_transform import affine_transformation


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

        self.build_aff_trans()


    def build_aff_trans(self):
        self.aff_trans = affine_transformation(
            self.standard_start_pt, self.standard_mid_pt, self.standard_end_pt,
            self.start_pt, self.mid_pt, self.end_pt)

    def build(self, max_distance=None, legendre_ratio=None):

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
        return np.zeros_like(self.a)

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

    def clean_copy(self):
        
        ret = copy.deepcopy(self)
        ret.panels = []
        return ret
        
    def reversed(self):
        pass # TODO
        