import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle

from time import time

import os
curr_dir = os.path.dirname(__file__)

required_tol = 1e-10

with open(curr_dir + '/global_pipe_with_geometry.pickle','rb') as f:
    pipe = pickle.load(f)

pipe.build_omegas(required_tol)

with open('global_pipe_with_omegas.pickle','wb') as f:
    pickle.dump(pipe,f)
