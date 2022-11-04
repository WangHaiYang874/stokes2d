import sys
sys.path.insert(0,'../src/')
from curve import *
from utils import *
from multiply_connected_pipe import *
import pickle

with open('global_pipe_with_geometry.pickle','rb') as f:
    pipe = pickle.load(f)

required_tol = 5e-11


pipe.build_omegas(required_tol)

with open('global_pipe_with_omegas.pickle','wb') as f:
    pickle.dump(pipe,f)
