# %%
import sys
sys.path.insert(0,'./src/')
from curve import *
from utils import *
import numpy as np
from pipe import *
import pickle

from joblib import Parallel, delayed

# %%
l1 = pt(-10,0)
l2 = pt(4,-3)
l3 = pt(4,3)
ls = np.array([l1,l2,l3])
rs = np.array([1,1,1])
pipe1 = NLets(ls,rs)
shift1 = np.array([0,0])

# pipe1.build_geometry()
# pipe1.n_pts

# %%
ls2 = np.array([pt(5,0),-l2])
rs2 = np.array([1,1])
pipe2 = NLets(ls2,rs2)
shift2 = 2*l2

ls3 = np.array([-l3,pt(5,0)])
rs3 = np.array([1,1])
pipe3 = NLets(ls3,rs3)
shift3 = 2*l3

# %%
shift4 = shift2 + pt(10,0)
pipe4  = NLets(np.array([pt(-5,0), pt(4,3)]), np.array([1,1]))

shift5 = shift3 + pt(10,0)
pipe5  = NLets(np.array([pt(-5,0), pt(4,-3)]), np.array([1,1]))

# %%
pipe6 = NLets(np.array([[-4,-3],[10,0],[-4,3]]), np.array([1,1,1]))
shift6 = np.array([26,0])

# %%

def build(pipe):
    pipe.build(density=65, h_mult=4,n_jobs=4)
    pipe.A = None # free up memory
    return pipe

pipes = Parallel(n_jobs=6) (delayed(build) (p) for p in [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6])
    


# %%
with open('dev_Pipes.pickle','wb') as f:
    pickle.dump([
        [pipe1,pipe2,pipe3,pipe4,pipe5,pipe6],
        [shift1,shift2,shift3,shift4,shift5,shift6]],
                f,fix_imports=True,protocol=None)
