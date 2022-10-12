# %%
import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
import numpy as np
from pipe import *
import pickle

from joblib import Parallel, delayed
from time import time

start_time = time()

# %%
l1 = pt(-10,0)
l2 = pt(4,-3)
l3 = pt(4,3)
ls = np.array([l1,l2,l3])
rs = np.array([1,1,1])
pipe1 = NLets(ls,rs)
shift1 = np.array([0,0])


def build(pipe):
    pipe.build(density=25,n_jobs=4,max_distance=0.1, legendre_ratio=1e-8, tol=1e-5,h_mult=2)
    pipe.A = None # free up memory
    return pipe

pipes = [pipe1]
shifts = [shift1]

pipes = Parallel(n_jobs=6) (delayed(build) (p) for p in pipes)


# %%
with open('dev_Pipes0.pickle','wb') as f:
    pickle.dump([pipes, shifts],f,fix_imports=True,protocol=None)

print(time() - start_time)
