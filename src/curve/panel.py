from utils import *


class Panel:
    """
    each Panel is designed by
        - a 16 points gauss quadrature rule,
        - have a max distance of 1.
    """

    def __init__(self, a, da, x, y, domain) -> None:
        self.a = a
        self.da = da
        self.x = x
        self.y = y
        self.domain = domain

        self.dx_da = None
        self.dy_da = None
        self.ddx_dda = None
        self.ddy_dda = None

    def distance(self, pts):
        a = pt(self.x[0], self.y[0])
        b = pt(self.x[-1], self.y[-1])
        return np.distance(pts, a, b)

    @property
    def t(self):
        return self.x + 1j * self.y

    @property
    def dt_da(self):
        return self.dx_da + 1j * self.dy_da

    @property
    def k(self):
        return (self.dx_da * self.ddy_dda - self.dy_da * self.ddx_dda) / \
               ((self.dx_da ** 2 + self.dy_da ** 2) ** 1.5)

    @property
    def max_distance(self):
        return np.max(np.linalg.norm(np.diff(np.array([self.x, self.y])), axis=0))

    @property
    def legendre_coef_ratio(self):
        legendre_coef = np.polynomial.legendre.Legendre.fit(
            self.a, self.t, deg=len(self.a) - 1, domain=self.domain).coef
        return np.sum(np.abs(legendre_coef[-2:])) / np.sum(np.abs(legendre_coef[:2]))

    def good_enough(self, max_distance=None, legendre_ratio=None, domain_threhold=1e-8):

        if self.domain[1] - self.domain[0] < domain_threhold:
            return True

        max_distance = max_distance if max_distance else 1e-2
        legendre_ratio = legendre_ratio if legendre_ratio else 1e-14

        return self.max_distance < max_distance and self.legendre_coef_ratio < legendre_ratio
