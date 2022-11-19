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

with open (curr_dir + '/local_pipes.pickle', 'rb') as f:
    pipes = pickle.load(f)

shifts = [pt(0,0) for _ in pipes]

t = time()
for pipe in pipes:
    pipe.build(required_tol=1e-13,fmm=True)
    print(time()-t)
    t = time()

with open(curr_dir + '/pipes_and_shifts_built.pickle','wb') as f:
    pickle.dump([pipes, shifts],f,fix_imports=True,protocol=None)