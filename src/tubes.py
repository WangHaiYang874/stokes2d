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

class straight_line(geometry):
    def __init__(self,p1,p2,r=1) -> None:
        '''
        this creates a simple tube. Why do I create it first? because it serves well as 
        a template for other more sophisticated geometries. 
        '''
        super().__init__()
        self.center_line = line(p1,p2)
        theta = np.arctan2(p2[1]-p1[1],p2[0]-p1[0]) + np.pi/2
        
        ul = p1 + r*np.array([np.cos(theta),np.sin(theta)])
        ll = p1 - r*np.array([np.cos(theta),np.sin(theta)])
        ur = p2 + r*np.array([np.cos(theta),np.sin(theta)])
        lr = p2 - r*np.array([np.cos(theta),np.sin(theta)])
        
        up_line = line(ul,ur)
        right_line = line(ur,lr)
        low_line = line(lr,ll)
        left_line = line(ll,ul)
        
        
        
        
        
        
    


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
    
    def __init__(self, points, lines, corner_size=1e-1) -> None:
        
        assert len(points) == len(lines)
        
        n = len(points)
        self.curves = []
        
        for i in range(n):
            next = (i+1)%n
            
            if lines[i] == line:
                self.curves.append(line(points[i], points[next]))
            elif lines[i] == cap:
                self.curves.append(cap(points[i], points[next]))
            else:
                raise ValueError('invalid line type')


        # smooth the corners
        
        i = self.next_corner()
        while i is not None:
            j = (i+1)%n
            p = self.curves[i].p1
            q = self.curves[i].p2
            r = self.curves[j].p2
            
            p_ = q + corner_size*(q-p)/np.linalg.norm(q-p)
            r_ = q + corner_size*(r-q)/np.linalg.norm(r-q)
            corner = corner(p_, q, r_)
            self.curves[i].p2 = p_
            self.curves[j].p1 = r_
            self.curves.insert(j,corner)
            
        # getting all the caps, as they literally represents all the flows. 
        self.caps_index = [i for i in range(len(self.curves)) if isinstance(self.curves[i], cap)]
        self.caps_points = [(self.curves[i].p1 + self.curves[i].p2)/2 for i in self.caps_index]
        
                
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
    
    
    def build(self,max_distance=1e-2):
        for i in self.curves:
            i.build(max_distance)
        
                
            
        
        
        
        
        
    
    