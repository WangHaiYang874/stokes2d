import numpy as np


class affine_transformation:
    '''
    given 2d points:
        p1, p2, p3
        q1, q2, q3
    this returns an affine transformation Ax + b that sends pi to qi.
    '''

    def __init__(self, p1, p2, p3, q1, q2, q3):
        x = np.array([p1, p2, p3]).T
        y = np.array([q1, q2, q3]).T

        x_ = np.vstack([x, [1, 1, 1]])

        try:
            Ab_transpose = np.linalg.solve(x_.T, y.T)
        except np.linalg.LinAlgError:
            Ab_transpose, _, _, _ = np.linalg.lstsq(x_.T, y.T, rcond=0)
            # using is lstsq because the matrix x_ might be singular for curves like a Line.
            # as tested, this would give a numerical error of order 1e-15.
        Ab = Ab_transpose.T
        self.A = Ab[:, :2]
        self.b = Ab[:, 2]

    def __call__(self, x, y, with_affine=False):
        ret = np.matmul(self.A, np.array([x, y]))
        if with_affine:
            return ret + self.b[:, np.newaxis]
        return ret
