# Abstract Pipe Module

This module defines the interface and implementations for abstract pipe objects that can be connected to form pipe networks. It provides the foundation for connecting multiple Stokes solvers in a modular way.

## Core Classes

### `AbstractPipe`
Base class for all pipe objects that can be connected in a network. Defines the interface for pipes with inlets/outlets (lets) that can be matched and connected.

### `RealPipe` 
Concrete implementation wrapping a `MultiplyConnectedPipe` solver with spatial positioning. Represents a physical pipe segment that can be translated and rotated in space for network assembly.

### `BoundaryPipe`
Simplified pipe representation using only boundary conditions (flux specifications). Used for external boundary conditions in pipe networks without requiring full geometric solvers.

### `Let` and `BoundaryLet`
Connection point data structures defining inlet/outlet properties:
- Position (x, y coordinates)
- Direction (angle in radians) 
- Diameter
- Flux (for `BoundaryLet`)

## Key Features

### Connection Matching
- `Let.almost_match()` - checks geometric compatibility between connection points
- Automatic verification of position, diameter, and direction alignment
- Configurable tolerance for numerical matching

### Spatial Transformations
- `RealPipe` supports translation via shift parameters
- **Rotation support**: Counter-clockwise rotation by arbitrary angles [0, 2π)
- **Transformation order**: Rotation applied first, then translation
- Maintains geometric consistency during transformations

### Flow Management
- Pressure drop calculations between connected lets
- Flux conservation enforcement across connections
- Integration with underlying Stokes solvers

## Usage Example

```python
# Create pipe with translation only
real_pipe = RealPipe(multiply_connected_pipe, shift_x=1.0, shift_y=0.5)

# Create pipe with rotation (90° counter-clockwise)
rotated_pipe = RealPipe(multiply_connected_pipe, rotation=np.pi/2)

# Create pipe with rotation and translation (rotation applied first)
transformed_pipe = RealPipe(multiply_connected_pipe, 
                           shift_x=2.0, shift_y=1.0, rotation=np.pi/4)

# Check if two lets can connect (rotation affects matching)
if let1.almost_match(let2):
    # Connect pipes in network
    pipe_system.connect(pipe1, let1_idx, pipe2, let2_idx)

# Define boundary condition
boundary_let = BoundaryLet(x=0, y=0, dir=0, dia=1.0, flux=2.5)
boundary_pipe = BoundaryPipe([boundary_let])
```

## File Structure

- `abstract_pipe.py` - Base `AbstractPipe` class definition
- `real_pipe.py` - `RealPipe` implementation with geometric transforms  
- `boundary_pipe.py` - `BoundaryPipe` for external boundary conditions
- `let.py` - Connection point data structures (`Let`, `BoundaryLet`)

This module enables the modular composition of complex pipe networks from individual solved pipe segments, providing the abstraction layer needed for efficient network simulation.
