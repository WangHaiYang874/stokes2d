from utils import *
from .curve import Curve

class Circle(Curve):
    standard_start_pt = pt(1,0)
    standard_end_pt   = pt(0,1)
    standard_mid_pt   = pt(-1,0)
    def x_fn(_,a): return np.cos(a*np.pi)
    def y_fn(_,a): return np.sin(a*np.pi)
    def dx_da_fn(_,a): return -np.pi*np.sin(a*np.pi)
    def dy_da_fn(_,a): return np.pi*np.cos(a*np.pi)
    def ddx_dda_fn(_,a): return -np.pi**2*np.cos(a*np.pi)
    def ddy_dda_fn(_,a): return -np.pi**2*np.sin(a*np.pi)
    
    def __init__(self, radius, center) -> None:
        
        # TODO: this is absolutely bs to make the code runnning....
        start_pt = pt(radius,0) + center
        end_pt = pt(0,radius) + center
        mid_pt = pt(-radius,0) + center
        
        super().__init__(start_pt, end_pt, mid_pt)
        
        self.aff_trans.A = np.array([[radius,0],[0,radius]])
        self.aff_trans.b = center