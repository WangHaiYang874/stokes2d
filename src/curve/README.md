# Curve Module

This module provides geometric curve representations and boundary discretization for 2D Stokes solvers. It implements a hierarchical system from individual parametric curves to complete boundary descriptions with adaptive refinement.

## Core Classes

### `Curve` (Base Class)
Abstract base class for parametric curve representations. All curves are defined on the standard parameter domain [-1, 1] with affine transformations to target geometry. Provides adaptive panel-based discretization with spectral accuracy.

### Specific Curve Types

#### `Line`
Straight line segment between two points. Simplest curve type with linear parametrization. Used for straight pipe walls and geometric connections.

#### `Circle` 
Complete circular boundary with configurable radius, center, and orientation. Supports both clockwise and counterclockwise orientations for different boundary types.

#### `Cap`
Smooth semicircular cap for pipe endpoints. Provides inlet/outlet boundary conditions with prescribed flux. Includes automatic matching point and direction calculation for pipe connections.

#### `Corner`
Smooth corner connection between two line segments. Uses rounded transitions to avoid singular behavior at sharp corners while maintaining prescribed geometry.

### Discretization Classes

#### `Panel`
Individual discretization segment of a curve with Gaussian quadrature points. Provides local geometric information (position, derivatives, curvature) and adaptive refinement capabilities based on error estimates.

#### `Boundary`
Collection of curves forming a complete boundary (open or closed). Handles multi-curve boundaries, ensures geometric consistency, and provides unified access to discretization data across all constituent curves.

## Key Features

### Adaptive Refinement
- Panel-based subdivision with error control
- Spectral accuracy through high-order quadrature
- Automatic refinement near geometric singularities
- Configurable tolerance for solution requirements

### Geometric Transformations
- Affine transformations from standard to physical coordinates
- Automatic scaling, rotation, and translation
- Preservation of geometric properties during transformation

### Boundary Condition Support
- Prescribed velocity on curve segments
- Flux specification for inlet/outlet caps
- Automatic normal vector computation
- Support for mixed boundary conditions

### Mathematical Properties
- Curvature computation at all points
- Arc length parametrization support
- Tangent and normal vector fields
- Integration over curve segments

## Usage Example

```python
# Define individual curves
line1 = Line(start_pt=pt(0,0), end_pt=pt(2,0))
cap = Cap(start_pt=pt(2,0), end_pt=pt(2,1), mid_pt=pt(2.5,0.5))
line2 = Line(start_pt=pt(2,1), end_pt=pt(0,1))

# Create complete boundary
curves = [line1, cap, line2]
boundary = Boundary(curves)

# Build adaptive discretization
for curve in curves:
    curve.build(required_tol=1e-10)

# Access discretization data
x, y = boundary.t.real, boundary.t.imag  # Boundary points
dx, dy = boundary.dt.real, boundary.dt.imag  # Tangent vectors
kappa = boundary.k  # Curvature
```

## File Structure

- `curve.py` - Base `Curve` class with adaptive discretization
- `line.py` - Straight line segments  
- `circle.py` - Circular boundaries
- `cap.py` - Semicircular pipe caps with flux conditions
- `corner.py` - Smooth corner connections
- `panel.py` - Individual discretization panels with quadrature
- `boundary.py` - Multi-curve boundary management
- `linear_transform.py` - Affine transformation utilities

## Mathematical Foundation

Curves use parametric representation with standard domain [-1,1]:
- Position: r(a) = (x(a), y(a))
- Tangent: r'(a) = (dx/da, dy/da)  
- Curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
- Arc length: s = ∫ |r'(a)| da

Adaptive refinement ensures the panel size h satisfies the "5h rule" for boundary integral accuracy, with automatic subdivision until error tolerance is met.
