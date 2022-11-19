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

import os
curr_dir = os.path.dirname(__file__)

with open (curr_dir + '/global_pipe.pickle', 'rb') as f:
    pipe = pickle.load(f)

required_tol = 1e-12

pipe.build_geometry(required_tol=required_tol)
pipe.build_A(fmm=True)

print('n_pts: ', len(pipe.t))
pipe.build_omegas(tol=required_tol)

with open(curr_dir + '/global_pipe_built.pickle','wb') as f:
    pickle.dump(pipe, f)
