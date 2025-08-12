# Experiment 2: Global vs Local Pipe Solving

This directory explores the distinction between global (full domain) and local (individual pipe) solving strategies for pipe networks.

## Files

### Python Scripts
- **`build_global.py`** - Constructs global solver for entire pipe network
- **`build_local.py`** - Builds individual local solvers for each pipe segment

### Notebooks
- **`constructing_geometry.ipynb`** - Demonstrates geometric construction for complex pipe networks
- **`demo.ipynb`** - Compares global vs local solving approaches with visualizations
- **`demo gmres.ipynb`** - Focuses on GMRES solver behavior and convergence analysis

### Visualization Assets
- **`simple-interface-problem.png`** - Illustrates basic pipe connection problem
- **`simple-interface-problem-network.png`** - Shows network topology for interface matching
- **`standard_pipe_flows_demo.png`** - Demonstrates standard flow patterns
- **`superficial_cap_demo.png`** - Shows cap (inlet/outlet) flow behavior

### Data Files
- **`global_pipe.pickle`** - Serialized global network solver
- **`local_pipes.pickle`** - Collection of individual pipe solvers
- **`pipes_and_shifts_built.pickle`** - Pre-built pipe configurations with positioning

## Purpose

This experiment investigates:
1. **Global approach**: Solving entire network as single multiply-connected domain
2. **Local approach**: Solving individual pipes and connecting via interface conditions
3. **Performance comparison**: Memory usage, computational cost, accuracy
4. **Interface handling**: Methods for matching boundary conditions between pipes

The results inform the choice between monolithic and modular solving strategies for different network sizes and topologies.
