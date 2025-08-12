# Utils Module

This module contains common utilities, constants, and helper functions used throughout the Stokes2D solver. It provides mathematical utilities, coordinate transformations, and global configuration parameters.

## Key Constants

### Numerical Parameters
- `ERR = 1e-17` - Quadrature error tolerance for scipy.integrate.quad
- `THRESHOLD = 1e-10` - Tolerance for matching pipe connections
- `FMM_EPS = 5e-16` - Fast Multipole Method accuracy parameter
- `REQUIRED_TOL = 1e-8` - Default tolerance for adaptive refinement
- `DOMAIN_THRESHOLD = 1e-6` - Minimal panel size for 16-point Gaussian quadrature
- `MAX_DISTANCE = 0.08` - Distance threshold for panel quality assessment

### Solver Parameters
- `GMRES_TOL = 1e-10` - GMRES iterative solver tolerance
- `GMRES_MAX_ITER = 4` - Maximum GMRES iterations before restart
- `RESTART = 500` - GMRES restart parameter
- `DENSITY = 50` - Default density for field evaluation grids

## Key Functions

### Quadrature Rules
- `gauss_quad_rule(n=16, domain=(-1,1))` - Generates Gaussian quadrature points and weights

### Geometric Utilities
- `line_intersect(p1, p2, p3, p4)` - Computes intersection of two lines
- `pt(x, y)` - Creates 2D point as numpy array
- `pt2cplx(pt)` - Converts point to complex number
- `cplx2pt(cplx)` - Converts complex number to point

### Coordinate Transformations
- `U2H(U)` - Converts velocity field to complex representation
- `H2U(H)` - Converts complex field to velocity components

These utilities provide the mathematical foundation and consistent parameter settings for the entire Stokes2D solver framework.

## Classes

### `Callback`
Progress monitoring class for iterative solvers. Tracks residual norms and timing information during GMRES iterations, providing periodic progress updates.

## Usage Example

```python
from utils import *

# Create quadrature rule
points, weights = gauss_quad_rule(n=16, domain=(-1, 1))

# Geometric operations
p1, p2 = pt(0, 0), pt(1, 1)
p3, p4 = pt(0, 1), pt(1, 0)
intersection = line_intersect(p1, p2, p3, p4)

# Coordinate conversions
z = pt2cplx(pt(1, 2))  # 1 + 2j
velocity = U2H(np.array([u, v]))  # Convert to complex representation

# Solver monitoring
callback = Callback()
# Use with GMRES: gmres(A, b, callback=callback)
```

## File Structure

- `__init__.py` - Constants, mathematical utilities, coordinate transforms
- `callback.py` - `Callback` class for solver monitoring

This module provides the essential infrastructure for consistent numerical computation throughout the Stokes2D solver.
