import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle
from joblib import Parallel, delayed

from time import time

import os
curr_dir = os.path.dirname(__file__)

with open(curr_dir + '/pipes_and_shifts.pickle','rb') as f:
    pipe_and_shifts = pickle.load(f)

pipes = [p for p,_ in pipe_and_shifts]

i = 0
while i < len(pipes):
    if pipes[i] in pipes[:i]:
        pipes.pop(i)
    else:
        i+=1
        
print("# pipes: ", len(pipes))
    
        
required_tol = 1e-10

def build_pipe(pipe,i):
    t = time()
    pipe.build_geometry(required_tol=required_tol)
    for p in pipe.panels: p._build()
    pipe.build_A(fmm=True)
    print(i, "geometry_built, time used: ", time()-t)
    print(i, "number of points: ", len(pipe.t))

    t = time()
    pipe.build_omegas(tol=required_tol)
    
    print(i, "solver built, time used: ", time()-t, 'n omegas: ', len(pipe.omegas))
    print()
    
    pipe.build_pressure_drops()
    
    with open(curr_dir + '/local_pipes' + str(i) + '.pickle','wb') as f:
        pickle.dump(pipes,f,fix_imports=True,protocol=None)

Parallel(n_jobs=5)(delayed(build_pipe)(pipe,i) for i,pipe in enumerate(pipes))