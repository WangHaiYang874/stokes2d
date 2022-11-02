from curve import Line, Cap, Corner, Boundary
from utils import *
from .multiply_connected_pipe_from_boundary import MultiplyConnectedPipe

class SmoothPipe(MultiplyConnectedPipe):
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

        curves = []

        n = len(lines)
        for i in range(n):
            j = (i + 1) % len(lines)
            if lines[i] == Line:
                curves.append(lines[i](points[i], points[j]))
            else:  # I need to inference the mid point.
                vec = points[i] - points[i-1]
                vec = vec/np.linalg.norm(vec)
                mid_pt = (points[i] + points[j])/2 + 1.5*vec
                curves.append(lines[i](points[i], points[j], mid_pt))
        smooth_corners(curves, corner_size)
        bdr = Boundary(curves)
        self.boundaries = [bdr]

def smooth_corners(curves, corner_size=1e-1):

    i = next_corner(curves)

    while i is not None:

        l1 = curves.pop(i)
        l2 = curves.pop(i)

        p = l1.start_pt
        q = l1.end_pt
        r = l2.end_pt

        corner_size = min(corner_size, np.linalg.norm(
            p - q) / 2, np.linalg.norm(r - q) / 2)
        assert (corner_size > 1e-3)

        start_pt = q + (((p - q) / np.linalg.norm(p - q)) * corner_size)
        end_pt = q + (((r - q) / np.linalg.norm(r - q)) * corner_size)

        curves.insert(i, Line(end_pt, r))
        curves.insert(i, Corner(start_pt, end_pt, q))
        curves.insert(i, Line(p, start_pt))

        i = next_corner(curves)

def next_corner(curves):
    """
    if there are two consecutive Line, they will have a Corner.
    this function return the index of the lines.
    """

    for i in range(len(curves)):
        j = (i + 1) % len(curves)
        if isinstance(curves[i], Line) and isinstance(curves[j], Line):
            return i
    return None
