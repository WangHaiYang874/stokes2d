import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle

from time import time
from scipy.sparse.linalg import gmres

import os
curr_dir = os.path.dirname(__file__)

with open(curr_dir + '/global_pipe_with_geometry.pickle','rb') as f:
    pipe = pickle.load(f)

required_tol = 1e-9
print(len(pipe.t),' pts')

b = np.concatenate([pipe.boundary_value(0).real, pipe.boundary_value(0).imag],dtype=np.float64)
pipe.build_A(fmm=True)
A = pipe.A

omega_sep, _ = gmres(A, b, 
                    atol=0, tol=required_tol,
                    restart=2000, maxiter=100, 
                    callback=Callback(),callback_type='pr_norm')

if _ < 0:
    print('Did not converge')

print('totol time cost', time() - t)

with open('omegasep_global.pickle','wb') as f:
    pickle.dump(omega_sep,f)
