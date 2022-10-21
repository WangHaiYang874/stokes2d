from .curve import Curve
from utils import *

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

    def __init__(self, start_pt=pt(-1,0), end_pt=pt(1,0), mid_pt=None) -> None:
        mid_pt = (start_pt + end_pt) / 2
        super().__init__(start_pt, end_pt, (start_pt + end_pt)/2)
