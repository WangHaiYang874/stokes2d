from .smooth_pipe import SmoothPipe
from curve import Line, Cap
from utils import pt


class PipeWithDifferentRadius(SmoothPipe):
    # TODO verify
    def __init__(self, l1, l2, l3, r1, r2, corner_size=1e-1) -> None:
        pts = [pt(0, -r1), pt(l1, -r1), pt(l1+l2, -r2), pt(l1+l2+l3, -r2),
               pt(l1+l2+l3, r2), pt(l1+l2, r2), pt(l1, r1), pt(0, r1)]
        curves = [Line, Line, Line, Cap, Line, Line, Line, Cap]
        super().__init__(pts, curves, corner_size)
