import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle

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
    
required_tol = 1e-12

for i,pipe in enumerate(pipes):
    
    print("--------pipe: ", i)
    
    pipe.build_geometry(required_tol=required_tol)
    pipe.build_A(fmm=True)
    
    matching_points = np.array([i.matching_pt for i in pipe.lets])
    matching_points_z = matching_points[:,0] + 1j*matching_points[:,1]
    pairs_needing_correction = np.array(pipe.mat_vec.pairs_needing_correction(matching_points_z))
    total_corrections = np.sum(pairs_needing_correction)
    if not total_corrections == 0:
        print('?')
    
    print('n_pts: ', len(pipe.t))
    
    
    pipe.build_omegas(tol=required_tol)
    print()    
    
with open(curr_dir + '/v0.pickle','wb') as f:
    pickle.dump(pipe_and_shifts,f,fix_imports=True,protocol=None)