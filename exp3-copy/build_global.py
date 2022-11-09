import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle

from time import time

import os
curr_dir = os.path.dirname(__file__)

with open(curr_dir + '/global_pipe.pickle','rb') as f:
    pipe = pickle.load(f)

t = time()

pipe.build_geometry(required_tol=1e-10,n_jobs=10)
for p in pipe.panels:
    p._build()
pipe.build_A(fmm=True)
pipe.build_omegas(tol=1e-8,max_iter=1, restart=6000)
pipe.build_pressure_drops()
# pipe.build_plotting_data(density=density)

with open(curr_dir + '/global_pipe_built.pickle','wb') as f:
    pickle.dump(pipe,f,fix_imports=True,protocol=None)