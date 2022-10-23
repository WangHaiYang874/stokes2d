from .curve import Curve
from utils import *

class Line(Curve):
    standard_start_pt = pt(-1, 0)
    standard_end_pt = pt(1, 0)
    standard_mid_pt = pt(0, 0)
    x_fn = lambda _, a: a
    y_fn = lambda _, a: np.zeros_like(a)
    dx_da_fn = lambda _, a: np.ones_like(a)
    dy_da_fn = lambda _, a: np.zeros_like(a)
    ddx_dda_fn = lambda _, a: np.zeros_like(a)
    ddy_dda_fn = lambda _, a: np.zeros_like(a)

    def __init__(self, start_pt=pt(-1,0), end_pt=pt(1,0), mid_pt=None) -> None:
        super().__init__(start_pt, end_pt, (start_pt + end_pt) / 2)
        self.build_aff_trans()

    def build_aff_trans(self):
        super().build_aff_trans()
        theta = np.arctan2(self.end_pt[1] - self.start_pt[1], self.end_pt[0] - self.start_pt[0])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        scaling = np.linalg.norm(self.end_pt - self.start_pt)/2
        
        self.aff_trans.A = scaling * rotation_matrix
        self.aff_trans.b = self.mid_pt