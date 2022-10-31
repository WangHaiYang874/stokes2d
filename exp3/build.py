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

global_pipe_fmm = MultiplyConnectedPipeFromPipeSystem(ps)
global_pipe_dense_mat = MultiplyConnectedPipeFromPipeSystem(ps)

# assert global_pipe.boundaries[0].orientation == 1
# assert global_pipe.boundaries[1].orientation == -1

t = time()
global_pipe_dense_mat.build(fmm=False)
print('dense_mat_vec time',time()-t)

t = time()
global_pipe_fmm.build(fmm=True)
print("fmmtime", time()-t)

global_pipe_dense_mat.mat_vec.clean()

with open(curr_dir + 'global_pipe_dense_solve.pickle','wb') as f:
    pickle.dump(global_pipe_dense_mat, f)

with open(curr_dir + 'global_pipe_fmm.pickle','wb') as f:
    pickle.dump(global_pipe_fmm, f)
