from .smooth_pipe import SmoothPipe
from curve import Line, Cap
from utils import pt


class T_shaped(SmoothPipe):
    # TODO verify

    def __init__(self, l, r, corner_size=1e-1):

        assert r > 0

        pts = [pt(-l, r), pt(-l, -r), pt(-r, -r), pt(-r, -l),
               pt(r, -l), pt(r, -r), pt(l, -r), pt(l, r)]
        curves = [Cap, Line, Line, Cap, Line, Line, Cap, Line]

        super().__init__(pts, curves, corner_size)
