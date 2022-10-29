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
    panels: list[Panel]

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

    def build(self, required_tol=REQUIRED_TOL,p=16):

        # TODO: change
        
        if not len(self.panels):
            p = Panel(self, (-1, 1),p)
            self.panels.append(p)

        # refine the panels.
        i = 0
        while i < len(self.panels):
            if self.panels[i].good_enough(required_tol):
                i += 1
                continue

            # the panel is not good enough, so we want to refine it.
            p = self.panels.pop(i)
            p1, p2 = p.refined()
            self.panels.insert(i, p2)
            self.panels.insert(i, p1)
            

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
    def dt(self):
        return self.dt_da * self.da
    
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
        
    def transformed(self, shift):

        start_pt = self.start_pt + shift
        end_pt = self.end_pt + shift
        mid_pt = self.mid_pt + shift
        
        return self.__class__(start_pt, end_pt, mid_pt)
    
    def reversed(self):
        pass # TODO
        