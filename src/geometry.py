import numpy as np
from scipy.special import p_roots as gauss_quad_nodes
from scipy.integrate import quad
import numbers

class geometry:
    def __init__(self) -> None:
        
        # the parametrization, d_parametrization, etc. 
        self.a = None
        self.da = None
        self.x = None
        self.y = None
        self.dx_da = None
        self.dy_da = None
        self.ddx_dda = None
        self.ddy_dda = None
        
        # the boundary velocity condition. 
        self.u
        self.v

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
    
    def get_h(self):
        return -self.v + 1j*self.u


class line(geometry):
    
    def __init__(self, p1, p2) -> None:
        
        '''
        this gives the line starts from p1 and ends at p2.
        '''
        
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        
        
    def build_geometry(self, max_distance=1e-2):
        
        n = 8
        cond = True
        
        while cond:
            a, da = gauss_quad_nodes(n)
            self.a = a
            self.da = da
            self.x = self.p1[0] + (self.p2[0]-self.p1[0])*((a+1)/2)
            self.y = self.p1[1] + (self.p2[1]-self.p1[1])*((a+1)/2)
            if np.max(np.abs(np.diff(self.get_t()))) < max_distance:
                cond = False
            else:
                n *= 2
        
        self.dx_da = (self.p2[0]-self.p1[0])/2 * np.ones(n)
        self.dy_da = (self.p2[1]-self.p1[1])/2 * np.ones(n)
        self.ddx_dda = np.zeros(n)
        self.ddy_dda = np.zeros(n)

        # because in our geometry, only caps represent the inflow and outflow, which might
        # have non-zero velocity.
        
        self.u = np.zeros(n)
        self.v = np.zeros(n)
        

class cap(geometry):

    def __init__(self, p1=(1,0),p2=(-1,0), max_distance=None) -> None:
        
        '''
        by default, it generates a smooth cap with n points in the counterclockwise direction 
        that goes through points 
            p1 = (1,0)
            p* = (0,1), 
            p2 = (-1,0). 
        Looks like a semi-circle.
        
        Given specified value of p1, p2, this will deduce the p* accordingly. and draw a cap
        that starts at p1 and ends at p2, going in the counterclockwise direction. 
        '''
        
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        
    def build_geometry(self, max_distance=1e-2):
        
        n = 8
        cond = True
        
        while cond:

            a, da = gauss_quad_nodes(n)
            self.a = a
            self.da = da
            
            self.x = -8*_Psi(a)
            
            b = 1/quad(_Psi,0,1,epsabs=1e-14,epsrel=1e-14)[0]
            self.y = -b*_int_Psi(a)

            max_distance_ = np.max(np.abs(np.diff(self.get_t())))
            
            if max_distance_ < max_distance:
                cond = False
            else:
                n *= 2
            
        self.dx_da = -8*_psi(a)
        self.dy_da = -b*_Psi(a)

        self.ddx_dda = -8*_d_psi(a)
        self.ddy_dda = -b*_psi(a)
        
        scale = np.linalg.norm(self.p1 - self.p2)/2
        self.scale((-scale, -scale))
        
        # now it is the time to set the unit flux condition. 
        
        h = (scale**2 - self.x**2)*3/(4*scale**3)
        
        angle = np.angle(self.p2[0] + 1j*self.p2[1] - self.p1[0] - 1j*self.p1[1])
        
        self.h = h*np.exp(1j*angle)
        
        self.u = self.h.imag
        self.v = -self.h.real
        
        self.rotate(angle)        
        self.shift((self.p1 + self.p2)/2)
        
        


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


class doubly_obstructed_tube(geometry):
    def __init__(self) -> None:
        super().__init__()
        gamma1 = line((-21, 1), (-30, 1), n=128*8)
        gamma2 = obstruction(shift=(-20, 1), scale=(2, 1),
                             rotate=np.pi, n=128*2)
        gamma3 = line((19, 1), (-19, 1), n=128*8*4)
        gamma4 = obstruction(shift=(20, 1), scale=(2, 1),
                             rotate=np.pi, n=128*2)
        gamma5 = line((30, 1), (21, 1), n=128*8)
        gamma6 = cap(rotate=-np.pi/2, scale=(1, 1), shift=(30, 0), n=128*4)
        gamma7 = line((21, -1), (30, -1), n=128*8)
        gamma8 = obstruction(shift=(20, -1), scale=(2, 1), n=128*2)
        gamma9 = line((-19, -1), (19, -1), n=128*8*4)
        gamma10 = obstruction(shift=(-20, -1), scale=(2, 1), n=128*2)
        gamma11 = line((-30, -1), (-21, -1), n=128*8)
        gamma12 = cap(rotate=np.pi/2, scale=(1, 1), shift=(-30, 0), n=128*4)

        Gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,
                 gamma7, gamma8, gamma9, gamma10, gamma11, gamma12]
        Gamma.reverse()

        self.a = np.concatenate(
            [gamma.a + 2*i for i, gamma in zip(range(len(Gamma)), Gamma)])
        self.da = np.concatenate([gamma.da for gamma in Gamma])
        self.x = np.concatenate([gamma.x for gamma in Gamma])
        self.y = np.concatenate([gamma.y for gamma in Gamma])
        self.dx_da = np.concatenate([gamma.dx_da for gamma in Gamma])
        self.dy_da = np.concatenate([gamma.dy_da for gamma in Gamma])
        self.ddx_dda = np.concatenate([gamma.ddx_dda for gamma in Gamma])
        self.ddy_dda = np.concatenate([gamma.ddy_dda for gamma in Gamma])


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
    def __init__(self, center=None, scale=None, n=128) -> None:
        a = np.linspace(0, 2*np.pi, n+1)[:-1]
        da = 2*np.pi/n * np.ones(n)
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

        if scale is not None:
            self.scale(scale)
        if center is not None:
            self.shift(center)


class corner(geometry):
    def __init__(self, p, q, r, n=64):
        '''
        the points p, q, r on the R^2 describe a corner that is given by line p-q and then q-r. q is the point of intersection. 
        '''
        self.p = np.array(p)
        self.q = np.array(q)
        self.r = np.array(r)

        assert(np.linalg.norm(p-q) > 0)
        assert(np.linalg.norm(p-q) == np.linalg.norm(q-r))

        self.standard_corner(n)

        # now we begin the transformation of standard_corner to the corner

        standard_p = np.array([-1, 1])
        standard_q = np.array([0, 0])
        standard_r = np.array([1, 1])

        standard_v1 = standard_q - standard_p
        standard_v2 = standard_r - standard_q

        v1 = q-p
        v2 = r-q

        # We need to find a transformation that maps standard_vi to vi.
        standard_v_mat = np.array([[standard_v1[0], standard_v2[0]], [
                                  standard_v1[1], standard_v2[1]]])
        v_mat = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
        A = np.matmul(v_mat, np.linalg.inv(standard_v_mat))
        # the above code is equivalent to: A * standard_v_mat = v_mat

        self.x, self.y = np.matmul(A, np.array([self.x, self.y]))
        self.dx_da, self.dy_da = np.matmul(
            A, np.array([self.dx_da, self.dy_da]))
        self.ddx_dda, self.ddy_dda = np.matmul(
            A, np.array([self.ddx_dda, self.ddy_dda]))

        # Now, we need to shift the corner to the correct position.
        self.shift(q)

    def standard_corner(self, n):
        '''
        the standard corner is a corner with 
        p = (-1,1)
        q = (0,0)
        r = (1,1)

        which is just the graph of y=abs(x) for x in [-1,1]
        '''

        a, da = gauss_quad_nodes(n)
        self.a = a
        self.da = da
        self.x = a.copy()
        self.y = np.array([convoluted_abs(x_) for x_ in self.x])

        self.dx_da = np.ones(self.a.shape)
        self.dy_da = np.array([d_convoluted_abs(x_) for x_ in self.x])
        self.ddx_dda = np.zeros(self.a.shape)
        self.ddy_dda = np.array([dd_convoluted_abs(x_) for x_ in self.x])


def _psi(a: np.ndarray) -> np.ndarray:

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ret = ((a**2+1)*np.exp(4*a/(a**2-1))) / \
            ((a**2-1)*(1+np.exp(4*a/(a**2-1))))**2

    ret[abs(a) >= 1] = 0
    ret[np.isnan(ret)] = 0
    ret[ret == np.inf] = 0
    return ret


def _Psi(a):

    if isinstance(a, np.ndarray):    
        with np.errstate(divide='ignore', over='ignore'):
            ret = -np.tanh(-2*a/(1-a**2))/8
        ret[a >= 1] = 1/8
        ret[a <= -1] = -1/8
        nan_mask = np.isnan(ret)
        if np.any(nan_mask):
            ret[nan_mask & a > 0.9] = 1/8
            ret[nan_mask & a < -0.9] = -1/8
        return ret
    if isinstance(a,numbers.Number):
        if a >= 1:
            return 1/8
        if a <= -1:
            return -1/8
        return -np.tanh(-2*a/(1-a**2))/8
    


def _int_Psi(a):
    '''the number 128 in the expression below is chosen by numerical experiment'''
    if isinstance(a, np.ndarray):
        return np.array([quad(_Psi, -1, b, epsabs=1e-14,epsrel=1e-14)[0] for b in a])
    if isinstance(a, numbers.Number):
        return quad(_Psi, -1, a, epsabs=1e-14,epsrel=1e-14)[0]
    else:
        print('input is not a number or a numpy array')
        assert 0
    


def _d_psi(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        e = np.exp(4*a/(a**2-1))
        ret = (-2 * e * (2 - 3*a + 4*a**2 + 2*a**3 + 2*a**4 + a**5 + e*(-2 - 3 *
               a - 4*a**2 + 2 * a**3 - 2*a**4 + a**5)))/((1 + e)**3 * (-1 + a**2)**4)
    ret[np.isnan(ret)] = 0
    return ret


def _bump(a):
    if isinstance(a, numbers.Number):
        if np.abs(a) >= 1:
            return 0
        return np.exp(1/(a**2-1))

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


bump_def_int = quad(_bump, -1, 1, epsabs=1e-14,epsrel=1e-14)[0]

def _normalized_bump(a):
    return _bump(a)/bump_def_int

def convoluted_abs(x):
    if np.abs(x) >= 1:
        return np.abs(x)

    def b(y): return _normalized_bump(y)*np.abs(x-y)
    return quad(b, -1, 1, epsabs=1e-15, epsrel=1e-15, full_output=1)[0]

def d_convoluted_abs(x):
    if np.abs(x) >= 1:
        return np.sign(x)

    def b(y): return _bump(y)*np.sign(x-y)
    return -quad(b, -1, x, epsabs=1e-15, epsrel=1e-15, full_output=1)[0]\
        + quad(b, x, 1, epsabs=1e-15, epsrel=1e-15, full_output=1)[0]

def dd_convoluted_abs(x):
    if np.abs(x) >= 1:
        return 0
    return 2*_normalized_bump(x)
