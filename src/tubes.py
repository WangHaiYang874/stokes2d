'''
from the Curve, we have figured out how to draw stuff:
the lines, obstructions, and caps, smoothed corners, etc. 
those are going to be just a Panel for a tube.

In this file, I'll write the codes to automate the process of 
drawing tubes from the codes of the Curve. There are certain
things I wish this file can do. 

1. a tube should be a sampler, instead of fixing number of points on 
the Panel, we should be able to sample points on the tube to make sure
our stokes2d solver can have given tolerance and distance away from the 
boundary, as the main numerical error appears only near the boundary. In
my previous experiements, 5h rule works out pretty well for controlling the 
numerical error. 

2. I also want to write a codes for several generic tubes with different specifications
such as the radius, the length, the number of bifurcations, how large should the smoothed
Corner be, etc.
'''

import numpy as np
import sys
sys.path.append('.')
from geometry import *
from basic_spec import *

class pipe(geometry):
    def __init__(self) -> None:
        
        super().__init__()
        
        self.curves = None
        self.caps = None
        self.caps_out_normal_direction = None
        self.solver = None
        self.omegas = None
        self.pressure_drops = None
        self.extent = None
        self.grids = None
        self.velocity_fields = None
    
    
    def build_geometry(self, max_distance=5e-3):
        self.curves = [c.build(max_distance) for c in self.curves]
        self.a = np.concatenate([c.a + 2*i for i,c in enumerate(self.curves)])
        self.da = np.concatenate([c.da for c in self.curves])
        self.x = np.concatenate([c.x for c in self.curves])
        self.y = np.concatenate([c.y for c in self.curves])
        self.dx_da = np.concatenate([c.dx_da for c in self.curves])
        self.dy_da = np.concatenate([c.dy_da for c in self.curves])
        self.ddx_dda = np.concatenate([c.ddx_dda for c in self.curves])
        self.ddy_dda = np.concatenate([c.ddy_dda for c in self.curves])

        self.caps_center = []
        self.caps_center_out_normal = []
        self.caps_radius = []
        
        for i,c in enumerate(self.curves):
            if isinstance(c, cap):
                self.caps_center.append(c.center)
                self.caps_center_out_normal.append(c.center_out_normal)
                self.caps_radius.append(np.linalg.norm(c.p1 - c.p2))

    def build_solver(self):
        pass
    
    def build_omegas(self):
        pass
    
    def build_pressure_drops(self):
        pass
    
    def build_velocity_fields(self):
        pass
    
    def build_graph(self):
        pass
    
    def build(self):
        pass
    
    def is_inside(self, z):
        pass

class straight_pipe:
    def __init__(self,p1,p2,r=1) -> None:
        '''
        this creates a simple tube. Why do I create it first? because it serves well as 
        a template for other more sophisticated geometries. 
        '''
        super().__init__()
                
        
        self.angle = np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
        theta = self.angle + np.pi/2
        top_left = p1 + r*np.array([np.cos(theta),np.sin(theta)])
        bottom_left = p1 - r*np.array([np.cos(theta),np.sin(theta)])
        top_right = p2 + r*np.array([np.cos(theta),np.sin(theta)])
        bottom_right = p2 - r*np.array([np.cos(theta),np.sin(theta)])
        
        top_line = line(top_left,top_right)
        right_line = line(top_right,bottom_right)
        bottom_line = line(bottom_right,bottom_left)
        left_line = line(bottom_left,top_left)
        
        self.curves = [top_line,right_line,bottom_line,left_line]

        up = np.max(top_left[1],top_right[1],bottom_left[1],bottom_right[1])
        low = np.min(top_left[1],top_right[1],bottom_left[1],bottom_right[1])
        left = np.min(top_left[0],top_right[0],bottom_left[0],bottom_right[0])
        right = np.max(top_left[0],top_right[0],bottom_left[0],bottom_right[0])
                
    


class closed_geometry(geometry):
    '''
    given points 
        p1, p2, p3, ... , pn
    and the description of the Line
        l1 = p1-p2
        l2 = p2-p3
        ...
        ln = pn-p1
        the descriptions can be:
            a Line
            a Cap
            a obstruction
            etc
    this class should firstly create a cornered Curve with the given specification of points and lines
    and then it should automatically smooth the corners.  
    '''
    
    def __init__(self, points, lines,corner_size=1e-1) -> None:
        
        assert len(points) == len(lines)
        
        if not np.all([i in [line,cap] for i in lines]):            
            raise ValueError('invalid Curve type, only Line and Cap are permitted here. ')
        
        self.curves = []
        self.corner_size = corner_size
        
        n = len(lines)
        for i in range(n):
            j = (i + 1)%len(lines)
            self.curves.append(lines[i](points[i],points[j]))

        self.smooth_corners(self.corner_size)
        
    def smooth_corners(self, corner_size=1e-1):
        
        i = self.next_corner()
        
        while i is not None:
            
            j = (i+1)%len(self.curves)
            
            assert(np.all(self.curves[i].p2 == self.curves[j].p1))
            # they do intersect. 


            p = self.curves[i].p1
            q = self.curves[i].p2
            r = self.curves[j].p2
            
            corner_size = min(corner_size,np.linalg.norm(p-q)/2, np.linalg.norm(r-q)/2)

            assert(corner_size > 0)

            p1 = q + (((p-q) / np.linalg.norm(p-q)) * corner_size)
            r1 = q + (((r-q) / np.linalg.norm(r-q)) * corner_size)

            c = corner(p1, q, r1)
            
            self.curves[i].p2 = p1
            self.curves[j].p1 = r1
            self.curves.insert(j,c)

            i = self.next_corner()
            
    def next_corner(self):
        '''
        if there are two consecutive Line, they will have a Corner.
        this function return the index of the lines. 
        '''
        
        for i in range(len(self.curves)):
            j = (i+1)%len(self.curves)
            if isinstance(self.curves[i], line) and isinstance(self.curves[j], line):
                return i
        return None
    
    def build_geometry(self, max_distance=5e-3):
        [c.build(max_distance) for c in self.curves]
        self.a = np.concatenate([c.a + 2*i for i,c in enumerate(self.curves)])
        self.da = np.concatenate([c.da for c in self.curves])
        self.x = np.concatenate([c.x for c in self.curves])
        self.y = np.concatenate([c.y for c in self.curves])
        self.dx_da = np.concatenate([c.dx_da for c in self.curves])
        self.dy_da = np.concatenate([c.dy_da for c in self.curves])
        self.ddx_dda = np.concatenate([c.ddx_dda for c in self.curves])
        self.ddy_dda = np.concatenate([c.ddy_dda for c in self.curves])

        self.caps = []
        
        for i in range(len(self.curves)):
            if isinstance(self.curves[i], cap):
                self.caps.append(i)
        
class cross(closed_geometry):
    def __init__(self,length, radius, corner_size=0.2):
        
        p1 = np.array([-length,-radius])
        p2 = np.array([-radius, -radius])
        p3 = np.array([-radius, -length])
        p4 = np.array([radius, -length])
        p5 = np.array([radius, -radius])
        p6 = np.array([length, -radius])
        p7 = np.array([length, radius])
        p8 = np.array([radius, radius])
        p9 = np.array([radius, length])
        p10= np.array([-radius, length])
        p11= np.array([-radius, radius])
        p12= np.array([-length, radius])
        
        points = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12]
        curves = [line,line,cap,
                  line,line,cap,
                  line,line,cap,
                  line,line,cap]
        
        super().__init__(points, curves, corner_size)
    
    def build_solver(self, ):
        self.omegas = []
        pass
        
    def get_flows(self):
        self.caps_index = [i for i, j in enumerate(self.curves) if isinstance(j, cap)]
        self.inflow = self.caps[0]
        self.outflows = self.caps[1:]
        
    def get_all_boundary_velocity_conditions(self):
        
        velocities = []
        
        for j in self.outflows:
            velocity = []
            for i,c in enumerate(self.curves):
                if i == self.inflow:
                    velocity.append(c.get_boundary_velocity_condition(c.get_velocity(flux=1)))
                elif i == j:
                    velocity.append(c.get_boundary_velocity_condition(c.get_velocity(flux=-1)))
                else:
                    velocity.append(np.zeros_like(c.a))
            velocities.append(np.concatenate(velocity))
        
        self.velocities = np.array(velocities)
        
    def compute_pressure_drops(self):
        pressure_drops = []
        
        for i, o in enumerate(self.outflows):
            omega = self.omegas[i]
            pressure_drop = []
            
            p1 = cross.curves[self.inflow].p
            p1_cplx = p1[0] + 1j*p1[1]
            p1_pressure = self.solver.compute_pressure(p1_cplx, omega)
            
            for j, o2 in enumerate(self.outflows):
                p2 = cross.curves[o2].p
                p2_cplx = p2[0] + 1j*p2[1]
                p2_pressure = self.solver.compute_pressure(p2_cplx, omega)
                pressure_drop.append(p2_pressure - p1_pressure)
        
            pressure_drops.append(pressure_drop)
        
        self.pressure_drops = pressure_drops
        
        
        
    