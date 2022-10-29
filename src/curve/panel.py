from distutils.log import warn
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
        ret = np.squeeze(self.aff_trans(self.x_fn(a), self.y_fn(a),with_affine=True))
        return ret[0] + 1j*ret[1]

    @property
    def end_pt(self):
        a = np.array([self.domain[1]])
        ret = np.squeeze(self.aff_trans(self.x_fn(a), self.y_fn(a), with_affine=True))
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
        return [Panel(self.parent, (a, c),self.n), Panel(self.parent, (c, b),self.n)]

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
        g1_eval = x_eval + 1j * y_eval
        g2_eval = np.linalg.norm([dx_da_eval, dy_da_eval], axis=0)
        g3_eval = k_eval**2
        g4_eval = np.conjugate(dt_da_eval)/dt_da_eval
        # g5_eval = np.real(dt_da_eval)
        # g6_eval = x_eval * np.conjugate(dt_da_eval)/dt_da_eval
        # g7_eval = y_eval * np.conjugate(dt_da_eval)/dt_da_eval
        
        g1_interp = self.leg_interp(self.leg_fit(self.t), test_points)
        g2_interp = self.leg_interp(
            self.leg_fit(np.abs(self.dt_da)), test_points)
        g3_interp = self.leg_interp(self.leg_fit(self.k)**2, test_points)
        g4_interp = self.leg_interp(self.leg_fit(np.conjugate(self.dt_da)/self.dt_da), test_points)
        # g5_interp = self.leg_interp(self.leg_fit(np.real(self.dt_da)), test_points)
        # g6_interp = self.leg_interp(self.leg_fit(self.x * np.conjugate(self.dt_da)/self.dt_da), test_points)
        # g7_interp = self.leg_interp(self.leg_fit(self.y * np.conjugate(self.dt_da)/self.dt_da), test_points)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            error1 = np.linalg.norm(g1_interp - g1_eval)/np.linalg.norm(g1_eval)
            error2 = np.linalg.norm(g2_interp - g2_eval)/np.linalg.norm(g2_eval)
            error3 = np.linalg.norm(g3_interp - g3_eval)/np.linalg.norm(g3_eval)
            error4 = np.sum(np.abs(g4_interp - g4_eval))/np.sum(np.abs(g4_eval))
            # error5 = np.sum(np.abs(g5_interp - g5_eval))/np.sum(np.abs(g5_eval))
            # error67 = np.sum(np.abs(g6_interp - g6_eval)+np.abs(g7_interp - g7_eval))/np.sum(np.abs(g6_eval)+np.abs(g7_eval))

        if self.parent.__class__.__name__ in ["Corner", "Cap"] and (self.domain[0] == -1 or self.domain[1] == 1):
            error3 = 0
        elif self.parent.__class__.__name__ == 'Line':
            error3 = 0
            # error67 = 0

        ret = np.array([error1, error2, error3, error4,
                        # error5, error67
                        ])
        ret = ret[~np.isnan(ret)]
        return np.max(ret)

    def good_enough(self, required_tol=REQUIRED_TOL, domain_threhold=DOMAIN_THRESHOLD):

        if self.domain[1] - self.domain[0] < domain_threhold:
            warn(f"domain too small: {self.domain}")
            return True

        return self.leg_interp_error <= required_tol

    def normalize(self, t, with_affine=False):
        if with_affine:
            return (t-self.center)/self.scale
        else:
            return t/self.scale
        
    def _build(self):
        
        a_refined,da_refined  = gauss_quad_rule(self.m,domain=self.domain)
        t_normalized = self.normalize(self.t,with_affine=True)
        t_refined = self.aff_trans(self.x_fn(a_refined), self.y_fn(a_refined), with_affine=True)
        t_refined = t_refined[0] + 1j*t_refined[1]
        dt_refined = self.aff_trans(self.dx_da_fn(a_refined),self.dy_da_fn(a_refined),with_affine=False) * da_refined
        dt_refined = dt_refined[0] + 1j*dt_refined[1]
        
        t_refined_normalized = self.normalize(t_refined, with_affine=True)
        dt_refined_normalized = self.normalize(dt_refined, with_affine=False)

        self.t_refined_normalized = t_refined_normalized
        self.dt_refined_normalized = dt_refined_normalized

        weight2coeff = np.linalg.solve(np.flip(np.vander(t_normalized,N=self.n),axis=1),np.eye(self.n))
        coef2interp = np.flip(np.vander(t_refined_normalized,N=self.n),axis=1)
        
        self.density_interp = coef2interp @ weight2coeff
        
        self.V = np.flip(np.vander(t_refined_normalized,N=self.m),axis=1)
        
    def _build_for_targets(self, targets):
        targ = self.normalize(targets, with_affine=True)
        near = np.abs(targ) < 1.1
        far = ~near
        
        P = np.zeros((len(targ), self.m + 1), dtype=np.complex128)
        R = np.zeros((len(targ), self.m + 1), dtype=np.complex128)
        
        P[:,1] = self.p1(targets)
        P[far,self.m] = self.pm(targets[far])
        
        for k in range(1,self.m):
            P[near,k+1] = P[near, k] * targ[near] + (1 - (-1)**k)/(k)
        
        for k in range(self.m-1,1,-1):
            P[far,k] = (P[far, k+1] - (1-(-1)**k)/k )/targ[far]
            
        for k in range(1,self.m+1):
            R[:,k] = (k-1)*P[:,k-1] + ((-1)**(k-1)/(-1-targ)) - (1/(1-targ))
        
        P = P[:,1:]
        R = R[:,1:]

        C = np.linalg.solve(self.V.T,P.T)
        H = np.linalg.solve(self.V.T,R.T)
        
        IC = (C.T@self.density_interp) / (2j*np.pi)
        IH = (H.T@self.density_interp) / (2j*np.pi * self.scale)

        K1 = IC + np.conjugate(IC)
        
        K2 = np.conj(
            np.diag(targets.conj())@IH 
            + IH@np.diag(self.t*np.conj(self.dt_da)/self.dt_da)
            - np.diag(targets)@IH@np.diag(np.conj(self.dt_da)/self.dt_da)
            - IH@np.diag(self.t.conj())
        )
        
        return K1, K2

        
    def p1(self, targets):
        targ = self.normalize(targets, with_affine=True)
        psi = np.pi/4
        return (1j*psi + np.log((1-targ)/(-1-targ)/np.exp(1j*psi)))
    
    def pm(self, targets_far_away):
        targ = self.normalize(targets_far_away, with_affine=True)
        zj = self.t_refined_normalized[:,np.newaxis]
        d_zj = self.dt_refined_normalized[:,np.newaxis]
        x = targ[np.newaxis,:]
        return np.sum(np.power(zj,self.m-1)*d_zj/(zj-x), axis=0)
    
    
    # def cauchy_integral(self, density, target):
    #     P = self._build_for_targets(target)[0]
    #     return P @ density
    
    # def hadamard_integral(self, density, target):
    #     R = self._build_for_targets(target)[1]
    #     return R @ density
    
    
    