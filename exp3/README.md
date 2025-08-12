# Experiment 3: Pipe Connection and Interface Matching

This directory focuses on developing and testing algorithms for connecting individual pipe solvers into larger networks through interface matching.

## Files

### Python Scripts
- **`build_global.py`** - Constructs global reference solutions for validation
- **`build_locals.py`** - Builds library of local pipe solvers for network assembly

### Notebooks
- **`constructing geometry.ipynb`** - Geometric construction methods for pipe networks
- **`connecting_local.ipynb`** - Demonstrates local pipe connection algorithms
- **`test.ipynb`** - Tests and validates connection methods

### Data Files
- **`global_pipe_unbuilt.pickle`** - Global solver configuration before discretization
- **`global_pipe_with_geometry.pickle`** - Global solver with complete geometric setup
- **`pipes_and_shifts.pickle`** - Individual pipe configurations for network assembly

### Performance Data
- **`local runtime stat.txt`** - Runtime statistics for local solver performance

## Purpose

This experiment develops the core algorithms for:
1. **Interface detection**: Automatically finding matching connection points between pipes
2. **Geometric compatibility**: Ensuring diameter, position, and orientation matching
3. **Flow continuity**: Enforcing conservation laws at pipe junctions
4. **Pressure consistency**: Maintaining pressure continuity across connections
5. **Network assembly**: Building complete network solvers from individual components

The results from this experiment form the foundation of the `PipeSystem` class and abstract pipe interface.
