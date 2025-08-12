# Documentation Directory

This directory contains detailed mathematical and implementation documentation for the Stokes2D solver.

## Mathematical Documentation

### Core Theory
- **`math note.md`** - Mathematical foundations and derivations for the boundary integral formulation
- **`kernels.md`** - Green's function kernels and fundamental solutions for 2D Stokes equations
- **`modified kernel.md`** - Modified kernels for handling multiply-connected domains

### Numerical Methods
- **`fmm2d.md`** - Fast Multipole Method implementation details and algorithms
- **`bhffm2d.md`** - Biharmonic FMM specific implementation for stream function formulation
- **`cfmm2d.md`** - Complex FMM algorithms for 2D problems
- **`using fmm2d.md`** - Practical guide for using FMM acceleration

## Implementation Documentation

### Design Documents
- **`class_design.md`** - Software architecture and class hierarchy design
- **`设计连接.md`** - Design document for pipe connection algorithms (in Chinese)

### Application Examples
- **`semi-infinite-pipe.md`** - Mathematical analysis for semi-infinite pipe problems
- **`semi-infinite-pipe demo.md`** - Demonstration and validation of semi-infinite pipe solutions

### Research Documentation
- **`paper outline.md`** - Outline for research paper documenting the solver methods and results

## Purpose

This documentation provides:
1. **Mathematical rigor**: Complete derivations and theoretical justification
2. **Implementation details**: Algorithm specifics and computational considerations
3. **Validation studies**: Verification against analytical solutions
4. **Usage guidance**: How to apply the solver to different problem types

## Target Audience

- **Researchers**: Understanding mathematical foundations and novel contributions
- **Developers**: Implementation details for extending or modifying the solver
- **Users**: Practical guidance for applying the solver to fluid dynamics problems
- **Students**: Educational material for learning boundary integral methods

The documentation supports both the theoretical understanding and practical application of the Stokes2D boundary integral solver.
