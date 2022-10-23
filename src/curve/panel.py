from distutils.log import warn
from utils import *

class Panel:
    """
    # TODO DOCSTRING
    """
    
    domain: tuple[float, float]
    parent: "Curve"
    
    x: np.ndarray
    y: np.ndarray
    dx_da: np.ndarray
    dy_da: np.ndarray
    ddx_dda: np.ndarray
    ddy_dda: np.ndarray

    @property
    def t(self):return self.x + 1j * self.y

    @property
    def dt_da(self):return self.dx_da + 1j * self.dy_da

    @property
    def dt(self):return self.dt_da * self.da

    @property
    def k(self):
        return (self.dx_da * self.ddy_dda - self.dy_da * self.ddx_dda) / \
               ((self.dx_da ** 2 + self.dy_da ** 2) ** 1.5)
    @property
    def x_fn(self):return self.parent.x_fn
    @property
    def y_fn(self):return self.parent.y_fn
    @property
    def dx_da_fn(self):return self.parent.dx_da_fn    
    @property
    def dy_da_fn(self):return self.parent.dy_da_fn
    @property
    def ddx_dda_fn(self): return self.parent.ddx_dda_fn
    @property
    def ddy_dda_fn(self):return self.parent.ddy_dda_fn
    @property
    def aff_trans(self): return self.parent.aff_trans

    def __init__(self, curve, domain, p=16) -> None:
        self.domain = domain
        self.parent = curve
        self.p = p
        self.build()
        
    def build(self):
        a, da = gauss_quad_rule(domain=self.domain, n=self.p)
        self.a = a
        self.da = da
        x,y = self.aff_trans(self.x_fn(a), self.y_fn(a), with_affine=True)
        dx_da, dy_da = self.aff_trans(self.dx_da_fn(a), self.dy_da_fn(a))
        ddx_dda, ddy_dda = self.aff_trans(self.ddx_dda_fn(a), self.ddy_dda_fn(a))
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
        return [Panel(self.parent, (a, c)), Panel(self.parent, (c, b))]
    # def distance(self, pts):
    #     a = pt(self.x[0], self.y[0])
    #     b = pt(self.x[-1], self.y[-1])
    #     return np.distance(pts, a, b)

    
    # @property
    # def max_distance(self):
    #     return np.max(np.linalg.norm(np.diff(np.array([self.x, self.y])), axis=0))

    def leg_interp(self,lp,pts):
        return np.polynomial.legendre.legval(
            -(self.domain[1] + self.domain[0])/((self.domain[1] - self.domain[0])) 
            + pts*2/((self.domain[1] - self.domain[0])),
            lp.coef)
        
    def leg_fit(self,y):
        return np.polynomial.legendre.Legendre.fit(
            self.a, y, deg=len(self.a) - 1, domain=self.domain)
        
    @property
    def leg_interp_error(self):
        m = len(self.a)*2
        test_points = np.linspace(self.domain[0], self.domain[1],m)
        x_eval , y_eval = self.aff_trans(self.x_fn(test_points), self.y_fn(test_points), with_affine=True)
        dx_da_eval, dy_da_eval = self.aff_trans(self.dx_da_fn(test_points), self.dy_da_fn(test_points))
        ddx_dda_eval, ddy_dda_eval = self.aff_trans(self.ddx_dda_fn(test_points), self.ddy_dda_fn(test_points))
        k_eval = (dx_da_eval * ddy_dda_eval - dy_da_eval * ddx_dda_eval) / \
               ((dx_da_eval ** 2 + dy_da_eval ** 2) ** 1.5)
        g1_eval = x_eval + 1j * y_eval
        g2_eval = np.linalg.norm([dx_da_eval,dy_da_eval], axis=0)
        g3_eval = k_eval**2
        
        g1_interp = self.leg_interp(self.leg_fit(self.t), test_points)
        g2_interp = self.leg_interp(self.leg_fit(np.abs(self.dt_da)), test_points)
        g3_interp = self.leg_interp(self.leg_fit(self.k)**2, test_points)

        error1 = np.linalg.norm(g1_interp - g1_eval)/np.linalg.norm(g1_eval)
        error2 = np.linalg.norm(g2_interp - g2_eval)/np.linalg.norm(g2_eval)

        if self.parent.__class__.__name__ in ["Corner","Cap"]:
            if (self.domain[0] == -1 or self.domain[1] == 1):
                error3 = 0
        elif self.parent.__class__.__name__ == 'Line':
            error3 = 0
        else:
            error3 = np.linalg.norm(g3_interp - g3_eval)/np.linalg.norm(g3_eval)

        return np.max([error1, error2, error3])
        # return np.sum(np.abs(legendre_coef1[-2:])) / np.sum(np.abs(legendre_coef1[:2]))

    def good_enough(self, required_tol=REQUIRED_TOL, domain_threhold=DOMAIN_THRESHOLD):

        if self.domain[1] - self.domain[0] < domain_threhold:
            warn(f"domain too small: {self.domain}")            
            return True

        return self.leg_interp_error <= required_tol