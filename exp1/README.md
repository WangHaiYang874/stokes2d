# Experiment 1: FMM Implementation Testing

This directory contains the first experimental setup focusing on Fast Multipole Method (FMM) implementation and performance testing for the Stokes2D solver.

## Files

### Python Scripts
- **`build.py`** - Constructs basic pipe geometries and solver configurations
- **`build_fmm.py`** - Builds pipe solvers specifically configured for FMM acceleration
- **`fmmtest.py`** - Tests and benchmarks FMM performance against dense methods

### Notebooks
- **`demo.ipynb`** - Demonstrates FMM-accelerated Stokes solving with visualizations

### Data Files
- **`pipes_and_shifts.pickle`** - Serialized pipe geometries and positioning data
- **`pipes_and_shifts_fmm.pickle`** - FMM-optimized pipe configurations

## Purpose

This experiment validates the FMM implementation by:
1. Comparing accuracy between dense and FMM solvers
2. Measuring performance scaling with problem size
3. Testing FMM parameter sensitivity
4. Verifying correct behavior for large-scale problems

The results from this experiment informed the automatic dense/FMM switching logic implemented in the main solver.
