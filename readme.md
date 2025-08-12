# Stokes2D: Boundary Integral Solver for 2D Stokes Equations

This repository implements high-accuracy solvers for the 2D Stokes equations using boundary integral methods with stream function formulation, based on [Greengard et al.](https://www.sciencedirect.com/science/article/pii/S0021999196901023?via%3Dihub). The solver supports both simple and multiply-connected domains, with particular focus on pipe networks and complex geometries.

## Key Features

- **Boundary integral formulation** using stream function for 2D Stokes equations
- **Spectral accuracy** through adaptive panel-based discretization
- **Fast Multipole Method (FMM)** acceleration for large problems (O(N log N) complexity)
- **Modular pipe networks** - connect individual solvers to simulate complex pipe systems
- **Multiple geometric primitives** - lines, circles, caps, corners with smooth connections
- **Multiply-connected domains** - handles arbitrary topology with interior boundaries
- **Automatic mesh refinement** with configurable error tolerance

## Mathematical Foundation

The solver uses the biharmonic stream function formulation:
- Stream function ψ satisfies ∇⁴ψ = 0 in the domain
- Velocity field: **u** = (∂ψ/∂y, -∂ψ/∂x)
- Boundary integral equation with logarithmic Green's function kernels
- Handles both velocity and flux boundary conditions

## Applications

1. **Individual pipe solvers** for complex geometries (straight, curved, branched pipes)
2. **Pipe network simulation** through modular connection of individual solvers
3. **General bounded domain** Stokes problems with prescribed boundary conditions
4. **Research applications** in microfluidics, porous media, and low Reynolds number flows

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd stokes2d

# Install dependencies (using uv, conda, or pip)
uv sync  # or conda env create -f environment.yaml
```

### Basic Usage
```python
import sys
sys.path.insert(0, './src/')

from curve import Line, Cap, Circle
from multiply_connected_pipe import MultiplyConnectedPipeFromCurves

# Define pipe geometry
curves = [
    Line(start_pt=pt(0,0), end_pt=pt(2,0)),      # Bottom wall
    Cap(start_pt=pt(2,0), end_pt=pt(2,1)),       # Right cap (outlet)
    Line(start_pt=pt(2,1), end_pt=pt(0,1)),      # Top wall  
    Cap(start_pt=pt(0,1), end_pt=pt(0,0))        # Left cap (inlet)
]

# Create and solve
pipe = MultiplyConnectedPipeFromCurves(curves)
pipe.build()  # Adaptive mesh generation
flows = pipe.solve_flows()  # Solve for flow scenarios

# Evaluate velocity field
x, y = np.meshgrid(np.linspace(0, 2, 100), np.linspace(0, 1, 50))
u, v = pipe.velocity_field(x, y)
```

## Repository Structure

```
stokes2d/
├── src/                    # Main source code
│   ├── abstract_pipe/      # Pipe connection interfaces
│   ├── curve/             # Geometric curve definitions
│   ├── mat_vec/           # Matrix-vector operations & FMM
│   ├── multiply_connected_pipe/  # Core Stokes solvers
│   ├── pipe_system/       # Network topology management
│   └── utils/             # Common utilities and constants
├── docs/                   # Mathematical documentation
├── exp1-5/                # Experimental validation studies
├── graphs/                # Visualization notebooks
├── misc/                  # Development utilities
└── some notebooks/        # Additional testing notebooks
```

## Performance

- **Small problems** (< 5000 points): Dense matrix methods
- **Large problems** (≥ 5000 points): FMM acceleration with O(N log N) scaling
- **Spectral accuracy**: Error decreases exponentially with mesh refinement
- **Typical convergence**: GMRES solves in 2-4 iterations

## Documentation

- **`src/README.md`** - Overview of source code architecture
- **`docs/`** - Mathematical theory and implementation details
- **Individual module READMEs** - Detailed documentation for each component
- **Jupyter notebooks** - Examples and validation studies

## Key Notebooks

### Main Testing Notebooks
- **`test_multiply_connected.ipynb`** - Tests multiply-connected domain solver with multiple interior boundaries
- **`comparing connection and global.ipynb`** - Compares local connection vs global solving approaches
- **`dev.ipynb`** - Main development and testing notebook

### Experimental Validation
- **`exp1/demo.ipynb`** - FMM implementation testing and performance validation
- **`exp2/demo.ipynb`** - Global vs local solving strategy comparison
- **`exp3/connecting_local.ipynb`** - Pipe connection algorithm development
- **`exp5-return2poiseuille_flow/demo.ipynb`** - Flow recovery analysis downstream of obstructions

### Geometric Studies
- **`some notebooks/circle n=*.ipynb`** - Circular domain studies with different discretizations
- **`some notebooks/ellipse n=*.ipynb`** - Elliptical pipe cross-section analysis
- **`graphs/caps.ipynb`** - Inlet/outlet cap geometry visualization

## Summary

This Stokes2D solver provides a comprehensive framework for high-accuracy simulation of 2D viscous flows in complex geometries. Key innovations include:

1. **Modular Architecture**: Individual pipe solvers can be connected to simulate complex networks
2. **Adaptive Accuracy**: Spectral convergence through automatic mesh refinement
3. **Scalable Performance**: Automatic dense/FMM switching for optimal performance
4. **Geometric Flexibility**: Support for arbitrary multiply-connected domains
5. **Validated Implementation**: Extensive testing and comparison with analytical solutions

The solver is particularly well-suited for:
- Microfluidics and lab-on-chip design
- Pipe network analysis and optimization  
- Low Reynolds number flow research
- Educational applications in computational fluid dynamics

# Credit

This is a summer project of Haiyang Wang as an undergraduate math student at NYU Courant, guided and advised by professor Greengard, Phd. Fryklund, and Phd. Samuel Potter.

# todo

## top of my todo. 

- [ ] finish the paper. 
- [x] a solver for a multiply connected domain.
- [x] merge multiply connected domain with simply connected domain. 
- [ ] verify correction term for post-processing

## math-ish

- [ ] create different geometries
  - [ ] a simple straight pipe. 
  - [x] some bands with fixed angles. 
  - [x] Nlets 里面相临的两条线不能是平行的. 

## engineering-ish
- [ ] add the finder for z.
- [ ] add reverse_orientation method to the scheme: probably by simply rebuild the curves. 
- [x] write a build script to be ran on a server. 
- [ ] change some instance of `flatten` to `ravel`. 
- [ ] store every data of the standard pieces. the data needs to include
- [ ] removing scipy, 主要的scipy的dependency在
  - [ ] gmres
  - [ ] quad
  - [ ] LinearOperator

## games
  - [ ] how to use matplotlib with pygame
  - [ ] the coordinate system of pygame is stupid. Can I change that? [see stacks overflow](https://stackoverflow.com/questions/10167329/change-the-position-of-the-origin-in-pygame-coordinate-system#:~:text=Is%20it%20possible%20to%20change%20the%20coordinate%20system,and%20use%20it%20just%20before%20drawing%20any%20object.).

## probably later. 
- [ ] think about particles. it is hard but think about it...
s
## done 
- [X] extracting grad pressure
- [x] extracting pressure difference across the tubes. this could be done with a path integral. More: it can be built into part of the solver. so in the web page we can simply take the pre-solved results easily and take a linear combination. the presolved idea applies to velocity field as well. We can simply store it, when using it we can simply take a linear combination.
- [X] find the appropriate energy function. explained in Fredrik's email, pressure drop is energy per unit volume. So we should only care about that.
- [X] understand how to handle geometry with corners by adding some caps and smooth corners.
  - [X] smooth caps
  - [X] smooth corners
- [X] vectorize the evaluation of the solver
- [x] (maybe later) ffm-ize the evaluation of the solver
- [X] refactoring everything into a easy to use software package
  - [X] refactoring the geometry class
- [X] debugging the solve to get spectral accuracy
- [X] using `gmres` instead of `np.linalg.solve`
- [X] Geometry objects refactoring
  - [X] make them sampler obeying the $5h$-rule is obeyed given a $\epsilon <= 5h$.
  - [X] make them sampler that satisfies the ledregre coefficients. 
  - [X] make them carrying the initial conditions of velocity:
    - [X] for non-cap geometry, it will be simply all zeros
    - [X] for cap geometry, it will need to be assigned a flux and then have a boolean for inlet or outlet. 
- [x] investigating how far away does the flow returns to poiseuille again: The key thing is to attempt to characterize how fast the return to Poiseuille occurs in a length-normalized setting.
  - [X] plot flow everywhere in the domain
  - [X] to visualize the "return to Poiseuille": to plot the error along the axis of symmetry of the pipe. Then, instead of a heat map, you can make a semilogy plot with "length along the tube" on the x-axis and absolute error on the y-axis. That is, plot a 1D slice of your image plots in a semilogy plot to make the error clearer.
  - [x] 😂 do the weird doubly obstructed tube with only one side with obstruction? so it will be asymmetric.
  - [x] Do a parameter study: it is likely that the relative size of your obstruction is what controls the length over which the return to Poiseuille occurs. I recommend creating a parametrized version of your geometry where you can control the size of the obstruction with a single scalar parameter. I doubt there is any good reason to have two bumps on either side of the channel---a simple thing you could do is add a (smoothed?) semicircular "bite" on either side of the pipe of radius r. Then you can try plotting the "deviation from Poiseuille" with respect to the scale parameter r/R.
  - [x] check whether this provides an upper bound on how fast the return to Poiseuille happens. E.g., if you can fit one obstruction inside of another obstruction, will the return happen for the smaller, contained obstruction at least as fast as for the larger obstruction? Probably Fredrik or Leslie can weigh in and give some insight as to whether this is likely to be the case... I don't know, myself.
  - [x] set a tol for return to poiseuille, varying the obstruction height, and observe how fast it is returning.
  - [x] measure the difference of your flow to a Poiseuille flow at distance 7.5 from the obstruction (there you seem to reach the error imposed by the GMRES tolerance) as you increase N? Please plot this error vs the number of points on the boundary in log log scale.
  - [x] Nlets
    - [x] Y-shaped with varying parameter for optimization
    - [x] cross. 
- [x] an algorithm that connects the pipes. 
  - [x] abstract pipe: 
    - [x] lets
        - [x] position: matching_pt
        - [x] dir
        - [x] radius
    - [x] flows (fluxes) (lets[1:])
        - [x] the pressure drops
        - [x] fluxes
    - each lets is a vertex, each flow is an edge.  
  - [x] aggregate those pipes into a pipe system. 
- [x] analytically explain the return to poiseuille behaviour.   
- [x] plotting
  - [x] give a specific criterion to check if a point is in that geometry.
  - [x] the velocity field should be corrected near the boundary. 
    - [x] this essentially ask me to determine if a point is close to a curve, and such criterion should be 
    implemented for each curve to have the best performance: 
  - [x] pressure and vorticity field as well? 
  - [x] have a method for plotting the velocity field using perhaps imshow
  - [x] have a method for plotting the actual boundary of the geometry using a not-so-thin balck line.
  - [x] i can also plot vector field and stream line with number. How surprising.
  - [x] base points for each pipe. and when plotting, every coordinate needs to be shifted. 
- [x] make sure that no two pipe intersects: this requires calculating intersection of two polygons, I am not sure how to do it for now. 
- [x] package the gaussian quadrature rules from scipy into something like a json file?.
- [x] organize my python files. 
- [x] test the matching, see if it really works well. 
