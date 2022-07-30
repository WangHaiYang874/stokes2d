import sys
from turtle import clear
import numpy as np
import pathlib 

src_dir = pathlib.Path(__file__).absolute().parent
project_dir = src_dir.parent
build_dir = project_dir.joinpath('build')
gauss_quad_dir = build_dir.joinpath('gauss_quadrature_rules')


# loading gauss quadrature rules. 

quad_rules = {}

'''
this quadrature rules are computed by scipy and stored 
in the build directory by the build script. 

The reason not to use scipy directly is for 
the compatibility with pygame wasm, 
which does not support scipy yet.

Notice that the quadrature rules are only stored with 
number of quadrature nodes = 8, 16, ..., 2**13 = 8192.
'''

try: 
    for i in range (3,14):
        quad_rules[i] = np.load(
            gauss_quad_dir.joinpath(f'n={2**i}.npy'))
except FileNotFoundError:
    print('Could not find gauss quadrature rules. Run the build script to generate them.')
    sys.exit(1)


