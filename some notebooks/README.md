# Some Notebooks Directory

This directory contains a collection of experimental and testing notebooks exploring various geometric configurations and solver behaviors.

## Files

### Geometric Studies
- **`caps demo.ipynb`** - Demonstrates cap geometry behavior and flow patterns
- **`circle n=64.ipynb`**, **`circle n=1024.ipynb`** - Circle discretization with different resolutions (64 and 1024 points)
- **`circle n=64 with pressure.ipynb`** - Circle flow analysis including pressure field computation

### Elliptical Geometries  
- **`ellipse n=64.ipynb`**, **`ellipse n=256.ipynb`** - Elliptical pipe cross-sections with varying discretization
- **`ellipse n=64 copy.ipynb`**, **`ellipse n=64 copy 2.ipynb`** - Variations and modifications of ellipse studies

### Progressive Experiments
- **`experiment0.ipynb`** - Initial experimental setup and basic functionality testing
- **`experiment1.ipynb`** - First major experiment with flow solving
- **`experiment1 copy.ipynb`**, **`experiment1 copy 2.ipynb`**, **`experiment1 copy 3.ipynb`** - Iterations and variations of experiment 1
- **`experiment2.ipynb`** - Second major experiment building on previous results

## Purpose

This collection represents:
1. **Resolution studies**: Understanding discretization requirements for different geometries
2. **Geometric exploration**: Testing solver behavior on various pipe cross-sections
3. **Progressive development**: Iterative improvement of solver capabilities
4. **Validation studies**: Comparing results across different configurations

## Key Insights

The notebooks explore:
- **Convergence with refinement**: How solution accuracy improves with more discretization points
- **Geometric effects**: How pipe cross-section shape affects flow patterns
- **Pressure field accuracy**: Validation of pressure computation in addition to velocity
- **Solver robustness**: Testing on increasingly complex configurations

These studies informed the development of automatic refinement algorithms and discretization strategies used in the final solver implementation.
