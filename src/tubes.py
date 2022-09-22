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
from utility_and_spec import *

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
    
    

    


