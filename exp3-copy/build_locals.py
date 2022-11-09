import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle

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
    
required_tol = 1e-11

for i,pipe in enumerate(pipes):
    
    print("--------pipe: ", i)
    pipe.build(required_tol=required_tol,fmm=True,density=70)
    with open(curr_dir + '/local_pipes' + str(i) + '.pickle','wb') as f:
        pickle.dump(pipe,f,fix_imports=True,protocol=None)
    print()    
    
with open(curr_dir + '/pipes_and_shifts_built.pickle','wb') as f:
    pickle.dump(pipe_and_shifts,f,fix_imports=True,protocol=None)