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
    
    def __init__(self) -> None:
        super().__init__()