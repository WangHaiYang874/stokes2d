# Stokes2D Source Code

This directory contains the core implementation of the 2D Stokes equation solver using boundary integration methods. The solver supports both simply and multiply-connected domains, with applications to pipe networks and complex geometries.

## Package Structure

### Core Modules

- **`abstract_pipe/`** - Abstract pipe system definitions and connection interfaces
- **`curve/`** - Geometric curve representations and boundary discretization
- **`mat_vec/`** - Matrix-vector operations with FMM acceleration support
- **`multiply_connected_pipe/`** - Main solver implementations for complex geometries
- **`pipe_system/`** - Network topology and pipe system management
- **`utils/`** - Common utilities, constants, and helper functions

## Key Components

### Solver Architecture
The solver follows a hierarchical design:
1. **Curves** define geometric boundaries using parametric representations
2. **MultiplyConnectedPipe** solves Stokes equations on bounded domains
3. **AbstractPipe** provides interfaces for connecting multiple domains
4. **PipeSystem** manages networks of connected pipes

### Numerical Methods
- Boundary integral equation formulation using stream function
- Adaptive panel-based discretization with spectral accuracy
- GMRES iterative solver for linear systems
- Fast Multipole Method (FMM) acceleration for large problems
- Automatic dense/FMM switching based on problem size

### Geometric Capabilities
- Smooth parametric curves (circles, caps, lines, corners)
- Multiply-connected domains with arbitrary topology
- Automatic boundary condition handling for pipe connections
- Support for inlets, outlets, and internal boundaries

## Usage Pattern

```python
# 1. Define geometry using curves
curves = [Circle(radius, center), Cap(start, end, mid)]

# 2. Create pipe solver
pipe = MultiplyConnectedPipeFromCurves(curves)

# 3. Set boundary conditions and solve
pipe.build()
flows = pipe.solve_flows()

# 4. Extract velocity/pressure fields
u, v = pipe.velocity_field(x, y)
p = pipe.pressure_field(x, y)
```

## Mathematical Foundation

The solver implements the boundary integral formulation of the 2D Stokes equations:
- Stream function ψ satisfies ∇⁴ψ = 0 in the domain
- Velocity: u = ∂ψ/∂y, v = -∂ψ/∂x
- Boundary conditions enforce prescribed velocities or flux constraints
- Green's function kernels provide fundamental solutions

## Performance Characteristics

- **Small problems** (< 5000 points): Dense matrix solver
- **Large problems** (≥ 5000 points): FMM-accelerated solver
- Spectral accuracy through adaptive refinement
- O(N log N) complexity for FMM-accelerated cases
