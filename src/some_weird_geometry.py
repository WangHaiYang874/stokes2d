import numpy as np
from scipy.special import p_roots as gauss_quad_nodes
from geometry import *
from math_functions import *



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