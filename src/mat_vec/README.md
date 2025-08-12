# Matrix-Vector Operations Module

This module implements the core matrix-vector operations for solving the boundary integral equations in the 2D Stokes solver. It provides both dense and Fast Multipole Method (FMM) accelerated implementations with automatic selection based on problem size.

## Core Classes

### `MatVec` (Base Class)
Abstract base class for matrix-vector operations in the boundary integral formulation. Handles the fundamental Green's function evaluations and kernel computations for the 2D Stokes equations using stream function formulation.

### `DenseMat`
Dense matrix implementation for small to medium-sized problems (< 5000 boundary points). Precomputes and stores full kernel matrices K1 and K2 for fast matrix-vector products during iterative solving.

### `Fmm`
Fast Multipole Method implementation for large problems (≥ 5000 boundary points). Uses hierarchical algorithms to achieve O(N log N) complexity instead of O(N²) for dense methods.

## Key Features

### Automatic Method Selection
The `mat_vec_constructor()` function automatically chooses between dense and FMM methods based on problem size:
- Dense method: N < 5000 points (faster for small problems)
- FMM method: N ≥ 5000 points (scalable for large problems)

### Kernel Evaluations
Implements the fundamental solution kernels for 2D Stokes equations:
- **K1 kernel**: Primary velocity contribution from boundary density
- **K2 kernel**: Secondary velocity contribution (conjugate density)
- **Singular terms**: Special handling for multiply-connected domains
- **Correction terms**: Ensure proper boundary conditions

### Velocity and Pressure Field Evaluation
- `velocity()` - Computes velocity field at arbitrary points in domain
- `d_phi()` - Evaluates pressure field derivatives
- Handles both near-field (close to boundary) and far-field evaluations
- Automatic singularity handling for evaluation points on boundary

### Multiply-Connected Domain Support
- Handles domains with interior boundaries (holes, islands)
- Automatic rank deficiency correction for solvability
- Flux conservation enforcement across all boundaries
- Interior point selection for well-posed formulation

## Mathematical Foundation

The solver implements the stream function formulation:
- Stream function ψ satisfies ∇⁴ψ = 0 in domain
- Velocity: u = ∂ψ/∂y, v = -∂ψ/∂x
- Boundary integral equation: ψ = K₁[ω] + K₂[ω̄] on boundary
- Density ω represents boundary layer potential

### Kernel Functions
- **K1**: Non-conjugate kernel for boundary integral operator
- **K2**: Conjugate kernel for complete formulation
- **Singular handling**: Limit process for self-interaction terms
- **FMM acceleration**: Hierarchical evaluation of far-field interactions

## Usage Example

```python
# Automatic selection based on problem size
mat_vec = mat_vec_constructor(pipe, fmm=None)  # Auto-select
mat_vec_dense = mat_vec_constructor(pipe, fmm=False)  # Force dense
mat_vec_fmm = mat_vec_constructor(pipe, fmm=True)  # Force FMM

# Matrix-vector product for iterative solver
result = mat_vec(omega_sep)  # omega_sep = [Re(ω), Im(ω)]

# Evaluate fields in domain
u, v = mat_vec.velocity(x, y, omega)  # Velocity field
dpx, dpy = mat_vec.d_phi(x, y, omega)  # Pressure gradients
```

## Performance Characteristics

### Dense Method (`DenseMat`)
- **Complexity**: O(N²) setup, O(N²) per iteration
- **Memory**: O(N²) storage for full matrices
- **Best for**: N < 5000, multiple solves with same geometry

### FMM Method (`Fmm`)
- **Complexity**: O(N log N) setup and per iteration  
- **Memory**: O(N) storage
- **Best for**: N ≥ 5000, large-scale problems
- **Accuracy**: Controlled by `FMM_EPS` parameter

## File Structure

- `mat_vec.py` - Base `MatVec` class with kernel implementations
- `dense_mat.py` - `DenseMat` class for direct matrix methods
- `fmm.py` - `Fmm` class using Fast Multipole Method
- `__init__.py` - Constructor function for automatic method selection

This module provides the computational core of the Stokes solver, enabling efficient solution of boundary integral equations for both simple and complex geometries.
