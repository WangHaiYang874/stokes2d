'''
from the geometry, we have figured out how to draw stuff: 
the lines, obstructions, and caps, smoothed corners, etc. 
those are going to be just a panel for a tube. 

In this file, I'll write the codes to automate the process of 
drawing tubes from the codes of the geometry. There are certain
things I wish this file can do. 

1. a tube should be a sampler, instead of fixing number of points on 
the panel, we should be able to sample points on the tube to make sure 
our stokes2d solver can have given tolerance and distance away from the 
boundary, as the main numerical error appears only near the boundary. In
my previous experiements, 5h rule works out pretty well for controlling the 
numerical error. 

2. I also want to write a codes for several generic tubes with different specifications
such as the radius, the length, the number of bifurcations, how large should the smoothed
corner be, etc. 
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
        self.caps_center = None
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
    and the description of the line 
        l1 = p1-p2
        l2 = p2-p3
        ...
        ln = pn-p1
        the descriptions can be:
            a line
            a cap
            a obstruction
            etc
    this class should firstly create a cornered geometry with the given specification of points and lines
    and then it should automatically smooth the corners.  
    '''
    
    def __init__(self, points, lines) -> None:
        
        assert len(points) == len(lines)
        
        self.curves = []
        
        for i in range(len(points)):
            
            next = (i+1)%len(points)
            
            if lines[i] == line:
                self.curves.append(line(points[i], points[next]))
            elif lines[i] == cap:
                self.curves.append(cap(points[i], points[next]))
            else:
                raise ValueError('invalid line type')

        self.smooth_corners()
        
    def smooth_corners(self, corner_size=1e-1):
        
        i = self.next_corner()
        
        while i is not None:
            
            j = (i+1)%len(self.curves)
            assert(np.all(self.curves[i].p2 == self.curves[j].p1))
            
            p = self.curves[i].p1
            q = self.curves[i].p2
            r = self.curves[j].p2
            
            p_ = q + corner_size*(p-q)/np.linalg.norm(q-p)
            r_ = q + corner_size*(r-q)/np.linalg.norm(r-q)
            c = corner(p_, q, r_)
            
            self.curves[i].p2 = p_
            self.curves[j].p1 = r_
            self.curves.insert(j,c)

            i = self.next_corner()
            
    def next_corner(self):
        '''
        if there are two consecutive line, they will have a corner. 
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
        
