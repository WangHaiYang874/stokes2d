from distutils.log import warn
from random import gauss
from utils import *


class Panel:
    """
    # TODO DOCSTRING
    """

    domain: tuple[float, float]
    parent: "Curve"

    n: int
    m: int

    x: np.ndarray
    y: np.ndarray
    dx_da: np.ndarray
    dy_da: np.ndarray
    ddx_dda: np.ndarray
    ddy_dda: np.ndarray

    scale: np.complex128
    center: np.complex128

    @property
    def m(self): return 2*self.n

    @property
    def start_pt(self):
        a = np.array([self.domain[0]])
        ret = np.squeeze(self.aff_trans(
            self.x_fn(a), self.y_fn(a), with_affine=True))
        return ret[0] + 1j*ret[1]

    @property
    def end_pt(self):
        a = np.array([self.domain[1]])
        ret = np.squeeze(self.aff_trans(
            self.x_fn(a), self.y_fn(a), with_affine=True))
        return ret[0] + 1j*ret[1]

    @property
    def scale(self):
        return (self.end_pt - self.start_pt)/2

    @property
    def center(self):
        return (self.start_pt + self.end_pt)/2

    @property
    def t(self): return self.x + 1j * self.y
    @property
    def dt_da(self): return self.dx_da + 1j * self.dy_da
    @property
    def dt(self): return self.dt_da * self.da

    @property
    def k(self):
        return (self.dx_da * self.ddy_dda - self.dy_da * self.ddx_dda) / \
               ((self.dx_da ** 2 + self.dy_da ** 2) ** 1.5)

    @property
    def arclen(self): return np.sum(np.abs(self.dt))
    @property
    def x_fn(self): return self.parent.x_fn
    @property
    def y_fn(self): return self.parent.y_fn
    @property
    def dx_da_fn(self): return self.parent.dx_da_fn
    @property
    def dy_da_fn(self): return self.parent.dy_da_fn
    @property
    def ddx_dda_fn(self): return self.parent.ddx_dda_fn
    @property
    def ddy_dda_fn(self): return self.parent.ddy_dda_fn
    @property
    def aff_trans(self): return self.parent.aff_trans

    def __init__(self, curve, domain, n=16) -> None:
        self.domain = domain
        self.parent = curve
        self.n = n
        self.build()

    def build(self):
        a, da = gauss_quad_rule(domain=self.domain, n=self.n)
        self.a = a
        self.da = da
        x, y = self.aff_trans(self.x_fn(a), self.y_fn(a), with_affine=True)
        dx_da, dy_da = self.aff_trans(self.dx_da_fn(a), self.dy_da_fn(a))
        ddx_dda, ddy_dda = self.aff_trans(
            self.ddx_dda_fn(a), self.ddy_dda_fn(a))
        self.x = x
        self.y = y
        self.dx_da = dx_da
        self.dy_da = dy_da
        self.ddx_dda = ddx_dda
        self.ddy_dda = ddy_dda

    def refined(self):
        a = self.domain[0]
        b = self.domain[1]
        c = (a + b) / 2
        return [Panel(self.parent, (a, c), self.n), Panel(self.parent, (c, b), self.n)]

    def leg_interp(self, lp, pts):
        return np.polynomial.legendre.legval(
            -(self.domain[1] + self.domain[0]) /
            ((self.domain[1] - self.domain[0]))
            + pts*2/((self.domain[1] - self.domain[0])),
            lp.coef)

    def leg_fit(self, y):
        return np.polynomial.legendre.Legendre.fit(
            self.a, y, deg=len(self.a) - 1, domain=self.domain)

    @property
    def leg_interp_error(self):

        m = len(self.a)*2
        test_points = gauss_quad_rule(domain=self.domain, n=m)[0]
        x_eval, y_eval = self.aff_trans(
            self.x_fn(test_points), self.y_fn(test_points), with_affine=True)
        dx_da_eval, dy_da_eval = self.aff_trans(
            self.dx_da_fn(test_points), self.dy_da_fn(test_points))
        ddx_dda_eval, ddy_dda_eval = self.aff_trans(
            self.ddx_dda_fn(test_points), self.ddy_dda_fn(test_points))
        k_eval = (dx_da_eval * ddy_dda_eval - dy_da_eval * ddx_dda_eval) / \
            ((dx_da_eval ** 2 + dy_da_eval ** 2) ** 1.5)
        dt_da_eval = dx_da_eval + 1j * dy_da_eval
        t_eval = x_eval + 1j * y_eval
        g1_eval = t_eval
        g2_eval = np.linalg.norm([dx_da_eval, dy_da_eval], axis=0)
        # g3_eval = k_eval**2

        g4_eval = np.conjugate(dt_da_eval)/dt_da_eval
        # TODO, this term should be testing agains t, not a

        # g5_eval = np.imag(t_eval*np.conjugate(dt_da_eval)/dt_da_eval)
        # g5_eval = np.real(dt_da_eval)
        # g6_eval = x_eval * np.conjugate(dt_da_eval)/dt_da_eval
        # g7_eval = y_eval * np.conjugate(dt_da_eval)/dt_da_eval

        g1_interp = self.leg_interp(self.leg_fit(self.t), test_points)
        g2_interp = self.leg_interp(
            self.leg_fit(np.abs(self.dt_da)), test_points)
        # g3_interp = self.leg_interp(self.leg_fit(self.k)**2, test_points)
        g4_interp = self.leg_interp(self.leg_fit(
            np.conjugate(self.dt_da)/self.dt_da), test_points)
        # g5_interp = self.leg_interp(self.leg_fit(np.imag(self.t*np.conjugate(self.dt_da)/self.dt_da)), test_points)
        # g5_interp = self.leg_interp(self.leg_fit(np.real(self.dt_da)), test_points)
        # g6_interp = self.leg_interp(self.leg_fit(self.x * np.conjugate(self.dt_da)/self.dt_da), test_points)
        # g7_interp = self.leg_interp(self.leg_fit(self.y * np.conjugate(self.dt_da)/self.dt_da), test_points)

        with np.errstate(divide='ignore', invalid='ignore'):
            error1 = np.linalg.norm(g1_interp - g1_eval) / \
                np.linalg.norm(g1_eval)
            error2 = np.linalg.norm(g2_interp - g2_eval) / \
                np.linalg.norm(g2_eval)
            # error3 = np.linalg.norm(g3_interp - g3_eval) / \
                # np.linalg.norm(g3_eval)
            error4 = np.sum(np.abs(g4_interp - g4_eval)) / \
                np.sum(np.abs(g4_eval))
            # error5 = np.sum(np.abs(g5_interp - g5_eval))/np.sum(np.abs(g5_eval))
            # error5 = np.sum(np.abs(g5_interp - g5_eval))/np.sum(np.abs(g5_eval))
            # error67 = np.sum(np.abs(g6_interp - g6_eval)+np.abs(g7_interp - g7_eval))/np.sum(np.abs(g6_eval)+np.abs(g7_eval))

        if self.parent.__class__.__name__ in ["Corner", "Cap"] and (self.domain[0] == -1 or self.domain[1] == 1):
            error3 = 0
        elif self.parent.__class__.__name__ == 'Line':
            error3 = 0
            # error67 = 0

        ret = np.array([error1, error2, error4,  # error5, error67, error3,
                        ])
        ret = ret[~np.isnan(ret)]
        return np.max(ret)

    def good_enough(self, required_tol=REQUIRED_TOL, domain_threhold=DOMAIN_THRESHOLD):

        if self.domain[1] - self.domain[0] < domain_threhold:
            warn(f"domain too small: {self.domain}")
            return True
        
        if np.max(np.abs(self.k * self.scale)) > 0.05:
            return False

        if self.arclen > 1:
            return False

        return self.leg_interp_error <= required_tol

    def normalize(self, t, with_affine=False):
        if with_affine:
            return (t-self.center)/self.scale
        else:
            return t/self.scale

    def _build(self):
        a_fine, da_fine = gauss_quad_rule(self.m, domain=self.domain)

        self.t_fine_norm = self.normalize(
            (lambda x: x[0]+1j*x[1])(self.aff_trans(self.x_fn(a_fine), self.y_fn(a_fine), with_affine=True)), with_affine=True)
        self.dt_fine_norm = self.normalize(
            (lambda x: x[0]+1j*x[1])(self.aff_trans(self.dx_da_fn(a_fine), self.dy_da_fn(a_fine), with_affine=False)), with_affine=False)*da_fine

        a_fine = gauss_quad_rule(self.m)[0]
        a = gauss_quad_rule(self.n)[0]

        self.omega_interp = vand(
            a_fine, self.n) @ np.linalg.solve(vand(a, self.n), np.eye(self.n))
        self.V = vand(self.t_fine_norm, self.m)

    def get_pr(self, targets):
        targ_norm = self.normalize(targets, with_affine=True)
        near = np.abs(targ_norm) < 1.1
        far = ~near

        P = np.zeros((len(targ_norm), self.m + 1), dtype=np.complex128)
        R = np.zeros((len(targ_norm), self.m + 1), dtype=np.complex128)

        P[:, 1] = self.p1(targets)
        P[far, self.m] = self.pm(targets[far])

        for k in range(1, self.m):
            P[near, k+1] = P[near, k] * targ_norm[near] + (1 - (-1)**k)/(k)

        for k in range(self.m-1, 1, -1):
            P[far, k] = (P[far, k+1] - (1-(-1)**k)/k)/targ_norm[far]

        for k in range(1, self.m+1):
            R[:, k] = (k-1)*P[:, k-1] + (-1)**k/(1+targ_norm) - 1/(1-targ_norm)

        P = P[:, 1:]
        R = R[:, 1:]
        R = R/self.scale
        return P, R

    def cauchy_integral(self, targets):
        P, _ = self.get_pr(targets)
        return np.linalg.solve(self.V.T, P.T).T @ self.omega_interp / (2j*np.pi)

    def hadamard_integral(self, targets):
        _, R = self.get_pr(targets)
        return np.linalg.solve(self.V.T, R.T).T @ self.omega_interp / (2j*np.pi)

    def both_integrals(self, targets):

        P, R = self.get_pr(targets)
        C = np.linalg.solve(self.V.T, P.T).T @ self.omega_interp / (2j*np.pi)
        H = np.linalg.solve(self.V.T, R.T).T @ self.omega_interp / (2j*np.pi)

        return C, H

    def p1(self, targets):
        targ_norm = self.normalize(targets, with_affine=True)
        psi = np.pi/4
        return (1j*psi + np.log((1-targ_norm)/((-1-targ_norm)*np.exp(1j*psi))))

    def pm(self, targets_far_away):
        targ_norm = self.normalize(targets_far_away, with_affine=True)
        zj = self.t_fine_norm[:, np.newaxis]
        d_zj = self.dt_fine_norm[:, np.newaxis]
        x = targ_norm[np.newaxis, :]
        return np.sum(np.power(zj, self.m-1)*d_zj/(zj-x), axis=0)


def vand(x, n):
    return np.column_stack([x**i for i in range(n)])
