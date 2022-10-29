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

l1 = pt(-5,0)
l2 = pt(4,-3)
l3 = pt(4,3)
ls = np.array([l1,l2,l3])
rs = np.array([0.5,0.5,0.5])
pipe1 = NLets(ls,rs)
shift1 = np.array([0,0])

ls2 = np.array([pt(5,0),-l2])
rs2 = np.array([0.5,0.5])
pipe2 = NLets(ls2,rs2)
shift2 = 2*l2

ls3 = np.array([-l3,pt(5,0)])
rs3 = np.array([0.5,0.5])
pipe3 = NLets(ls3,rs3)
shift3 = 2*l3

shift4 = shift2 + pt(10,0)
pipe4  = NLets(np.array([pt(-5,0), pt(4,3)]), np.array([0.5,0.5]))

shift5 = shift3 + pt(10,0)
pipe5  = NLets(np.array([pt(-5,0), pt(4,-3)]), np.array([0.5,0.5]))

pipe6 = NLets(np.array([[-4,-3],[5,0],[-4,3]]), np.array([0.5,0.5,0.5]))
shift6 = np.array([26,0])


pipes = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6]
shifts = [shift1,shift2,shift3,shift4,shift5,shift6]


t = time()
for pipe in pipes:
    pipe.build(required_tol=REQUIRED_TOL)
    print(time()-t)
    t = time()

with open(curr_dir + '/pipes_and_shifts.pickle','wb') as f:
    pickle.dump([pipes, shifts],f,fix_imports=True,protocol=None)