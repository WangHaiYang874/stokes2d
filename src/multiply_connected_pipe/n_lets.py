from .smooth_pipe import SmoothPipe
from utils import *
from curve import Line, Cap

pi = np.pi


class NLets(SmoothPipe):
    def __init__(self, ls, rs, corner_size=0.05):

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
