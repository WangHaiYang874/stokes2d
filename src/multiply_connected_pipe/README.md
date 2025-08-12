# Multiply Connected Pipe Module

This module contains the main solver implementations for 2D Stokes equations in multiply-connected domains (domains with holes or multiple boundaries). It provides both the core solver class and various specialized pipe geometries.

## Core Classes

### `MultiplyConnectedPipe`
Main solver class for interior Stokes problems with multiply-connected domains. Handles arbitrary geometries with multiple inlets, outlets, and internal boundaries. Implements the complete boundary integral equation solver using stream function formulation.

### Construction Classes

#### `MultiplyConnectedPipeFromCurves`
Constructs solver from a list of individual curves using automatic topology detection. Uses NetworkX graph algorithms to identify connected components and determine boundary relationships.

#### `MultiplyConnectedPipeFromBoundaries`
Constructs solver from pre-organized boundary objects. Assumes boundaries are already properly oriented and categorized as exterior/interior.

#### `MultiplyConnectedPipeFromPipeSystem`
Constructs solver from an abstract pipe system specification. Integrates with the pipe network framework for modular composition.

## Specialized Pipe Geometries

### Basic Shapes
- **`StraightPipe`** - Simple rectangular channel with uniform width
- **`SmoothPipe`** - Base class for pipes with smooth corner transitions

### Multi-Branch Configurations
- **`Cross`** - Four-way intersection with perpendicular branches
- **`TShaped`** - Three-way T-junction configuration  
- **`NLets`** - General N-way junction with arbitrary branch angles

### Advanced Geometries
- **`PipeWithDifferentRadius`** - Variable cross-section pipes
- **`SmoothPipe`** - Pipes with rounded corners and smooth transitions

## Key Features

### Geometric Flexibility
- Arbitrary multiply-connected domains
- Automatic boundary detection and orientation
- Support for complex pipe network topologies
- Smooth corner handling with configurable radius

### Solver Capabilities
- GMRES iterative solver with preconditioning
- Automatic rank deficiency handling for multiply-connected domains
- Flux conservation across all boundaries
- Multiple flow scenarios with superposition

### Boundary Conditions
- Prescribed velocity on pipe walls (no-slip)
- Flux specification at inlets/outlets
- Automatic pressure drop calculations
- Support for mixed boundary conditions

### Field Evaluation
- Velocity field computation at arbitrary points
- Pressure field and gradient evaluation
- Vorticity calculation
- Stream function values

## Mathematical Foundation

### Stream Function Formulation
The solver uses the biharmonic equation for stream function ψ:
- ∇⁴ψ = 0 in the domain
- Velocity: u = ∂ψ/∂y, v = -∂ψ/∂x
- Vorticity: ω = ∇²ψ

### Boundary Integral Equation
The fundamental solution representation:
- ψ(z) = Re[K₁[ω](z) + K₂[ω̄](z)] + corrections
- ω is the complex boundary density
- K₁, K₂ are integral operators with logarithmic kernels

### Multiply-Connected Handling
- Automatic detection of interior boundaries
- Rank deficiency correction using interior points
- Flux conservation enforcement
- Well-posed formulation through additional constraints

## Usage Example

```python
# Create from curves
curves = [Line(...), Cap(...), Circle(...)]
pipe = MultiplyConnectedPipeFromCurves(curves)

# Build discretization and solver
pipe.build()

# Solve for multiple flow scenarios
flows = pipe.solve_flows()

# Evaluate fields
x, y = np.meshgrid(...)
u, v = pipe.velocity_field(x, y)
p = pipe.pressure_field(x, y)

# Get flow properties
pressure_drops = pipe.pressure_drops
flow_rates = pipe.flow_rates
```

## Specialized Geometries Usage

```python
# Cross-shaped pipe junction
cross = Cross(length=2.0, radius=0.5, corner_size=0.1)

# T-shaped junction
t_pipe = TShaped(l=1.5, r=0.3, corner_size=0.05)

# Custom N-way junction
positions = [pt(1,0), pt(0,1), pt(-1,0)]  # 3 branches
radii = [0.5, 0.3, 0.4]
n_let = NLets(positions, radii, corner_size=0.1)
```

## Performance and Accuracy

### Solver Performance
- Automatic dense/FMM switching based on problem size
- GMRES convergence typically in 2-4 iterations
- Spectral accuracy through adaptive refinement
- Configurable tolerance for different applications

### Geometric Accuracy
- Adaptive panel refinement near corners
- Smooth corner transitions to avoid singularities
- Configurable corner smoothing parameters
- Automatic mesh quality control

## File Structure

- `multiply_connected_pipe.py` - Core solver implementation
- `multiply_connected_pipe_from_*.py` - Construction classes
- `smooth_pipe.py` - Base class for smooth geometries
- `straight_pipe.py` - Simple rectangular channels
- `cross.py` - Four-way intersections
- `t_shaped.py` - T-junction geometries
- `n_lets.py` - General N-way junctions
- `pipe_with_diff_r.py` - Variable radius pipes
- `modified_gmres.py` - Custom GMRES implementation

This module provides the computational heart of the Stokes2D solver, enabling simulation of fluid flow in complex pipe networks and multiply-connected geometries with high accuracy and efficiency.
