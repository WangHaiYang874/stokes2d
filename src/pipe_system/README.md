# Pipe System Module

This module implements the network topology management for connecting multiple pipe segments into complex pipe networks. It handles the graph structure, flow routing, and system-level solver for pipe networks.

## Core Classes

### `PipeSystem`
Main class for managing networks of connected pipes. Handles the topology as a graph where vertices represent connection points and edges represent flows between pipes. Provides system-level solving capabilities for the entire network.

### `Vertex`
Represents connection points between two pipe lets (inlets/outlets). Each vertex connects exactly two lets from different pipes and manages the geometric and flow matching constraints.

### `LetIndex`
Index structure for referencing specific lets within the pipe system. Contains pipe index and let index within that pipe, with flags for boundary conditions.

## Key Features

### Network Topology
- Graph-based representation using NetworkX
- Automatic cycle detection for pressure drop calculations
- Vertex management for pipe connections
- Edge mapping to flow variables

### Flow Management
- System-level flow variables and constraints
- Pressure drop calculations across cycles
- Flow conservation at vertices
- Integration with individual pipe solvers

### Solver Integration
- Connects individual `RealPipe` solvers into system
- Handles boundary conditions through `BoundaryPipe`
- Provides system-level linear algebra setup
- Enables efficient network simulation

## Mathematical Foundation

### Network Equations
The pipe system solver sets up a linear system:
- **Flow conservation**: Sum of flows at each vertex = 0
- **Pressure consistency**: Pressure drops around cycles sum to zero
- **Constitutive relations**: Pressure drop = resistance × flow

### System Matrices
- **A matrix**: Flow conservation constraints
- **B matrix**: Cycle pressure drop constraints
- Integration with individual pipe solvers for resistance calculations

## Usage Example

```python
# Create individual pipes
pipe1 = RealPipe(solver1, shift_x=0, shift_y=0)
pipe2 = RealPipe(solver2, shift_x=2, shift_y=0) 
pipe3 = RealPipe(solver3, shift_x=1, shift_y=1)

# Define boundary conditions
boundary_lets = [BoundaryLet(x=0, y=0, dir=0, dia=1.0, flux=1.0)]
boundary_pipe = BoundaryPipe(boundary_lets)

# Create pipe system
system = PipeSystem([pipe1, pipe2, pipe3], boundary_pipe)

# System will automatically:
# - Detect connections between pipes
# - Build network topology
# - Set up system equations
# - Enable network-level solving
```

## Connection Detection

The system automatically detects connections by:
1. Comparing all lets between different pipes
2. Using `Let.almost_match()` for geometric compatibility
3. Creating vertices for matched let pairs
4. Building network graph from vertex connections

## System Assembly

### Graph Construction
- Vertices represent pipe connections
- Edges represent flows between pipes
- Cycles represent closed flow loops
- Boundary vertices handle external conditions

### Linear System Setup
The system constructs matrices for:
- Flow variables (one per edge)
- Conservation constraints (one per vertex)
- Cycle constraints (one per independent cycle)
- Boundary conditions from external specifications

## File Structure

- `pipe_system.py` - Main `PipeSystem` class with network management
- `vertex.py` - `Vertex` class for connection points
- `let_index.py` - `LetIndex` for pipe/let referencing

## Network Solving Workflow

1. **Connection Detection**: Find matching lets between pipes
2. **Graph Construction**: Build network topology
3. **System Assembly**: Create global constraint matrices
4. **Linear Solve**: Solve for flow distribution
5. **Field Evaluation**: Compute velocities/pressures using individual pipe solvers

This module enables the simulation of complex pipe networks by connecting individual Stokes solvers in a modular, efficient framework.
