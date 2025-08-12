# Experiment 5: Return to Poiseuille Flow Analysis

This directory investigates how quickly disturbed flow returns to the classical Poiseuille (parabolic) profile downstream of geometric perturbations.

## Files

### Notebooks
- **`demo.ipynb`** - Analyzes flow recovery downstream of obstructions and geometric features

### Visualization Assets
- **`rtp_cv.png`** - Convergence plots showing return to Poiseuille behavior
- **`rtppipe.png`** - Pipe geometry visualization with obstruction features

## Purpose

This experiment characterizes:
1. **Recovery length scales**: How far downstream flow takes to return to Poiseuille
2. **Parameter dependence**: Effect of obstruction size, shape, and Reynolds number
3. **Convergence rates**: Exponential vs algebraic decay to Poiseuille profile
4. **Design implications**: Guidelines for pipe spacing in networks

## Mathematical Background

Classical Poiseuille flow in a pipe has the parabolic velocity profile:
- u(y) = (3/2) * U_avg * (1 - (2y/h)²) for pipe half-width h
- This represents the fully developed flow far from any disturbances

The experiment measures deviations from this profile using:
- L² norms of velocity difference
- Centerline velocity tracking
- Wall shear stress recovery
- Pressure gradient normalization

Understanding this recovery behavior is crucial for:
- Predicting when pipe segments can be treated independently
- Determining minimum separation distances in pipe networks
- Validating the modular connection approach used in the solver
