import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle

from time import time
from scipy.sparse.linalg import lgmres

import os
curr_dir = os.path.dirname(__file__)

with open(curr_dir + '/global_pipe_with_geometry.pickle','rb') as f:
    pipe = pickle.load(f)

required_tol = 1e-10
print(len(pipe.t),' pts')

b = np.concatenate([pipe.boundary_value(0).real, pipe.boundary_value(0).imag],dtype=np.float64)
A = pipe.A

class Callback:
    
    def __init__(self):
        self.iter = 0
        self.residuals = []
        self.t = time()
        
    def __call__(self, xk):
        self.iter += 1
        if self.iter % 200 == 0:
            residual = np.linalg.norm(A(xk) - b)
            self.residuals.append(residual)
            print(f"iter {self.iter}", 
                  f"resdiual {residual}", 
                  f"time {int((time()-self.t)/60)} min", 
                  sep='\t')

t = time()

omega_sep, _ = lgmres(A, b, 
                    atol=0, tol=required_tol,
                    inner_m=500, outer_k=100 , maxiter=100000, 
                    callback=Callback())

print('totol time cost', time() - t)

with open('omegasep_global.pickle','wb') as f:
    pickle.dump(pipe,f)
