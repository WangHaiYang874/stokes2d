from .n_lets import NLets
from utils import *


class Cross(NLets):
    def __init__(self, length, radius, corner_size=0.2):

        l1 = pt(-length, 0)
        l2 = pt(0, -length)
        l3 = pt(length, 0)
        l4 = pt(0, length)
        ls = np.array([l1, l2, l3, l4])
        rs = np.array([radius, radius, radius, radius])

        super().__init__(ls, rs, corner_size)
