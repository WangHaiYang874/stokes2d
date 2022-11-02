import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
import numpy as np
from multiply_connected_pipe import *
import pickle

from time import time

import os
curr_dir = os.path.dirname(__file__)

with open(curr_dir + '/pipes_and_shifts.pickle','rb') as f:
    pipe_and_shifts = pickle.load(f)

pipes = set([p for p,_ in pipe_and_shifts])
required_tol = 1e-11


for i,pipe in enumerate(pipes):
    
    print("----building the pipe", i, "of", len(pipes), '----')
    
    t = time()
    pipe.build_geometry(required_tol=required_tol)
    print("geometry_built, time used: ", time()-t)
    t = time()

    pipe.build_A(fmm=True)

    print("fmm built, time used: ", time()-t)
    t = time()

    pipe.build_omegas(tol=required_tol)

    print("solver built, time used: ", time()-t)
    t = time()

    pipe.build_pressure_drops()
    
    print('finished.')


with open(curr_dir + '/local_pipes.pickle','wb') as f:
    pickle.dump(pipes,f,fix_imports=True,protocol=None)

with open(curr_dir + '/pipes_and_shifts_built.pickle','wb') as f:
    pickle.dump(pipe_and_shifts,f,fix_imports=True,protocol=None)
