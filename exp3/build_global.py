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


with open(curr_dir + '/global_pipe_unbuilt.pickle','rb') as f:
    pipe = pickle.load(f)

required_tol = 1e-11

t = time()
pipe.build_geometry(required_tol=required_tol,n_jobs=4)
for p in pipe.panels: p._build()
print("geometry_built, time used: ", time()-t)
print("number of points: ", len(pipe.t))
t = time()


pipe.build_A(fmm=True)

print("fmm built, time used: ", time()-t)
t = time()

pipe.build_omegas(tol=required_tol)


print("solver built, time used: ", time()-t)
t = time()

print('everything is done. ')

pipe.build_pressure_drops()

with open(curr_dir + '/global_pipe_built.pickle','wb') as f:
    pickle.dump(pipe,f,fix_imports=True,protocol=None)
    