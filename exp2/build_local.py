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


def merging2pipe(pipes):
    curves = []
    for pipe in pipes:
        curves += pipe.curves
    i = 0
    while i < len(curves):
        if not isinstance(curves[i], Cap):
            i += 1
            continue
        for j in range(i+1,len(curves)):
            if not isinstance(curves[j], Cap):
                continue
            if np.linalg.norm(curves[i].matching_pt - curves[j].matching_pt) < 1e-8:
                curves.pop(j)
                curves.pop(i)
                break
        i += 1
    curves = [c.clean_copy() for c in curves]
    return MultiplyConnectedPipeFromCurves(curves)

def transformed(pipe,shift):
    curves = [c.transformed(shift) for c in pipe.curves]
    return MultiplyConnectedPipeFromCurves(curves)

pipe1 = transformed(pipe1,shift1)
pipe2 = transformed(pipe2,shift2)
pipe3 = transformed(pipe3,shift3)
pipe4 = transformed(pipe4,shift4)
pipe5 = transformed(pipe5,shift5)
pipe6 = transformed(pipe6,shift6)

first_pipe = merging2pipe([pipe1,pipe2,pipe3])
second_pipe = merging2pipe([pipe4,pipe5,pipe6])


pipes = [first_pipe,second_pipe]
shifts = [pt(0,0),pt(0,0)]

t = time()
for pipe in pipes:
    pipe.build(required_tol=1e-13,fmm=True)
    print(time()-t)
    t = time()
    pipe.mat_vec.clean()

with open(curr_dir + '/pipes_and_shifts_built.pickle','wb') as f:
    pickle.dump([pipes, shifts],f,fix_imports=True,protocol=None)