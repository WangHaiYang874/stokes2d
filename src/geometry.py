import numpy as np
from scipy.special import p_roots as gauss_quad_nodes
from scipy.integrate import fixed_quad as quad


class geometry:
    def __init__(self) -> None:
        self.a = None
        self.da = None
        self.x = None
        self.y = None
        self.dx_da = None
        self.dy_da = None
        self.ddx_dda = None
        self.ddy_dda = None

    def scale(self, scale):
        scale_x, scale_y = scale
        self.x = scale_x*self.x
        self.y = scale_y*self.y
        self.dx_da = scale_x*self.dx_da
        self.dy_da = scale_y*self.dy_da
        self.ddx_dda = scale_x*self.ddx_dda
        self.ddy_dda = scale_y*self.ddy_dda

    def rotate(self, theta):
        t = self.get_t()*np.exp(1j*theta)
        self.x = t.real
        self.y = t.imag

        dt = self.get_dt_da()*np.exp(1j*theta)
        self.dx_da = dt.real
        self.dy_da = dt.imag

        ddt = self.get_ddt_dda()*np.exp(1j*theta)
        self.ddx_dda = ddt.real
        self.ddy_dda = ddt.imag

    def shift(self, shift):
        shift_x, shift_y = shift
        self.x = self.x + shift_x
        self.y = self.y + shift_y

    def get_t(self):
        return self.x + 1j*self.y

    def get_dt_da(self):
        return self.dx_da + 1j*self.dy_da

    def get_ddt_dda(self):
        return self.ddx_dda + 1j*self.ddy_dda

    def get_k(self):
        return (self.dx_da*self.ddy_dda - self.dy_da*self.ddx_dda)/((self.dx_da**2+self.dy_da**2)**(1.5))

    def get_data(self):
        return self.a, self.da, self.get_t(), self.get_dt_da(), self.get_k()


class line(geometry):
    def __init__(self, p1, p2, n=128) -> None:
        a, da = gauss_quad_nodes(n)
        self.a = a
        self.da = da
        self.x = p1[0] + (p2[0]-p1[0])*((a+1)/2)
        self.y = p1[1] + (p2[1]-p1[1])*((a+1)/2)
        self.dx_da = (p2[0]-p1[0])/2 * np.ones(n)
        self.dy_da = (p2[1]-p1[1])/2 * np.ones(n)
        self.ddx_dda = np.zeros(n)
        self.ddy_dda = np.zeros(n)


class cap(geometry):
    '''TODO: make the standard cap a class variable'''

    def __init__(self, n=128, scale=None, rotate=None, shift=None):
        '''
        n: number of points
        scale: (scale_x,scale_y)
        rotate: theta \in [0,2pi]
        shift: (shift_x,shift_y)

        by default, it generates a smooth cap with n points in the counterclockwise direction 
        that goes through points (1,0), (0,1), (-1,0). Looks like a semi-circle. 
        '''
        a, da = gauss_quad_nodes(n)
        self.a = a
        self.da = da
        self.x = -8*_Psi(a)
        b = 1/quad(_Psi, 0, 1, n=128)[0]
        self.y = -b*_int_Psi(a)

        self.dx_da = -8*_psi(a)
        self.dy_da = -b*_Psi(a)

        self.ddx_dda = -8*_d_psi(a)
        self.ddy_dda = -b*_psi(a)

        if scale is not None:
            self.scale(scale)
        if rotate is not None:
            self.rotate(rotate)
        if shift is not None:
            self.shift(shift)


class obstruction(geometry):
    def __init__(self, n=128, scale=None, rotate=None, shift=None):

        a, da = gauss_quad_nodes(n)
        self.a = a
        self.da = da
        self.x = (a+1)/2
        self.y = _bump(a)

        self.dx_da = 1/2*np.ones(n)
        self.dy_da = _d_bump(a)

        self.ddx_dda = np.zeros(n)
        self.ddy_dda = _d_d_bump(a)

        self.shift((-0.5, 0))

        if scale is not None:
            self.scale(scale)
        if rotate is not None:
            self.rotate(rotate)
        if shift is not None:
            self.shift(shift)

    def get_data(self):
        return self.a, self.da, self.get_t(), self.get_dt_da(), self.get_k()


class obstructed_tube(geometry):
    def __init__(self, R=1, L=40, obstruction_h=0.5, obstruction_w=1, cap_height=1.5, N=1024) -> None:

        super().__init__()

        n = N//8
        obstruction_h = obstruction_h*np.e
        gamma1 = cap(rotate=-np.pi/2, scale=(R, cap_height),
                     shift=(L/2, 0), n=n)
        gamma2 = line(p1=(L/2, R), p2=(obstruction_w/2, R), n=n)
        gamma3 = obstruction(
            shift=(0, R),
            rotate=np.pi,
            scale=(obstruction_w, obstruction_h),
            n=n)
        gamma4 = line(p1=(-obstruction_w/2, R), p2=(-L/2, R), n=n)
        gamma5 = cap(rotate=np.pi/2, scale=(R, cap_height),
                     shift=(-L/2, 0), n=n)
        gamma6 = line(p1=(-L/2, -R), p2=(-obstruction_w/2, -R), n=n)
        gamma7 = obstruction(
            shift=(0, -R),
            rotate=0,
            scale=(obstruction_w, obstruction_h), n=n)
        gamma8 = line(p1=(obstruction_w/2, -R), p2=(L/2, -R), n=n)

        Gamma = [gamma1, gamma2, gamma3, gamma4,
                 gamma5, gamma6, gamma7, gamma8]
        self.a = np.concatenate(
            [gamma.a + 2*i for i, gamma in zip(range(len(Gamma)), Gamma)])
        self.da = np.concatenate([gamma.da for gamma in Gamma])
        self.x = np.concatenate([gamma.x for gamma in Gamma])
        self.y = np.concatenate([gamma.y for gamma in Gamma])
        self.dx_da = np.concatenate([gamma.dx_da for gamma in Gamma])
        self.dy_da = np.concatenate([gamma.dy_da for gamma in Gamma])
        self.ddx_dda = np.concatenate([gamma.ddx_dda for gamma in Gamma])
        self.ddy_dda = np.concatenate([gamma.ddy_dda for gamma in Gamma])


class circle(geometry):
    def __init__(self, center, scale, n=128) -> None:
        a = np.linspace(0, 2*np.pi, n+1)[:-1]
        da = 2*np.pi/n
        t = np.exp(a*1j)
        dt_da = 1j*t
        ddt_dda = -t

        self.a = a
        self.da = da
        self.x = t.real
        self.y = t.imag
        self.dx_da = dt_da.real
        self.dy_da = dt_da.imag
        self.ddx_dda = ddt_dda.real
        self.ddy_dda = ddt_dda.imag

        self.scale(scale)
        self.shift(center)


def _psi(a: np.ndarray) -> np.ndarray:

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = ((a**2+1)*np.exp(4*a/(a**2-1))) / \
            ((a**2-1)*(1+np.exp(4*a/(a**2-1))))**2

    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    ret[ret == np.inf] = 0
    return ret


def _Psi(a: np.ndarray) -> np.ndarray:

    with np.errstate(divide='ignore', over='ignore'):
        ret = -np.tanh(-2*a/(1-a**2))/8
    ret[a >= 1] = 1/8
    ret[a <= -1] = -1/8
    nan_mask = np.isnan(ret)
    if np.any(nan_mask):
        ret[nan_mask & a > 0.9] = 1/8
        ret[nan_mask & a < -0.9] = -1/8
    return ret


def _int_Psi(a: np.ndarray) -> np.ndarray:
    '''the number 128 in the expression below is chosen by numerical experiment'''
    return np.array([quad(_Psi, -1, b, n=128)[0] for b in a])


def _d_psi(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        e = np.exp(4*a/(a**2-1))
        ret = (-2 * e * (2 - 3*a + 4*a**2 + 2*a**3 + 2*a**4 + a**5 + e*(-2 - 3 *
               a - 4*a**2 + 2 * a**3 - 2*a**4 + a**5)))/((1 + e)**3 * (-1 + a**2)**4)
    ret[np.isnan(ret)] = 0
    return ret


def _bump(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = np.exp(1/(a**2-1))
    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    return ret


def _d_bump(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = -2 * _bump(a) * a / (a**2-1)**2
    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    return ret


def _d_d_bump(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = 2 * _bump(a) * (3*a**4 - 1) / (a**2-1)**4
    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    return ret
