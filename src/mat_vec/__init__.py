from .mat_vec import MatVec
from .dense_mat import DenseMat
from .fmm import Fmm

def mat_vec_constructor(pipe:"MultiplyConnectedPipe", fmm=None):
    if fmm is None:
        if pipe.n_pts < 5000:
            return DenseMat(pipe)
        else:
            return Fmm(pipe)
    if fmm:
        return Fmm(pipe)
    return DenseMat(pipe)
    