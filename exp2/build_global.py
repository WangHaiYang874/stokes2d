import os
curr_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(curr_dir)
os.chdir(project_dir)

import sys
sys.path.insert(0,'./src/')

from utils import *
from multiply_connected_pipe import *
from pipe_system import PipeSystem
from abstract_pipe import *
from multiply_connected_pipe import MultiplyConnectedPipeFromPipeSystem

import numpy as np
import pickle
from time import time

with open('./exp1/pipes_and_shifts.pickle','rb') as f:
    pipes, shifts = pickle.load(f)

bdr_pipe = BoundaryPipe([BoundaryLet(-5,0,0,1,-1),BoundaryLet(31,0,np.pi,1,1)])
real_pipes = [RealPipe(p,shift_x=shift[0],shift_y=shift[1]) for p,shift in zip(pipes,shifts)]
ps = PipeSystem(real_pipes,bdr_pipe)

required_tol = 1e-11
pipe = MultiplyConnectedPipeFromPipeSystem(ps)
pipe.build_geometry(required_tol=required_tol)
pipe.build_A(fmm=True)

print('n_pts: ', len(pipe.t))
pipe.build_omegas(tol=required_tol)

with open(curr_dir + '/global_pipe_built.pickle','wb') as f:
    pickle.dump(pipe, f)
