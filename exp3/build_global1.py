from pdb import Restart
import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle

from time import time

import os
curr_dir = os.path.dirname(__file__)

required_tol = 1e-11

with open(curr_dir + '/global_pipe_with_geometry.pickle','rb') as f:
    pipe = pickle.load(f)

print(len(pipe.t),' pts')

t = time()

pipe.build_omegas(required_tol,max_iter=100,restart=4000)

print('totol time cost', time() - t)

with open('global_pipe_with_omegas.pickle','wb') as f:
    pickle.dump(pipe,f)
