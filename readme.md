# stokes 2d

This repository aims to build solvers of 2d Stokes equation using boundary integration method of stream function in [Greengard et al.](https://www.sciencedirect.com/science/article/pii/S0021999196901023?via%3Dihub) for some simple geometries: a straight pipe, a curved pipe, a pipe with multiple income and outcome branches, etc. 

As pipes can be connected to form a network of pipes, our solvers can be connected to form a solver for a network of pipes. This gives us a fast and efficient way to build solver for 2d Stokes equation on complicated network of pipes. 

On the other hand, this could also be used as a direct way of building solvers for 2d Stokes equation on bounded geometries with prescribed boundary velocity. 

# credit
This is a summer project of Haiyang Wang as an undergraduate math student at NYU Courant, guided and advised by professor Greengard, Phd. Fryklund, and Phd. Samuel Potter. 


# todo
## math-ish
- [ ] extracting potentials
- [ ] understand how to handle geometry with corners by adding some caps and smooth corners. 

## engineering-ish
- [x] using `gmres` instead of `np.linalg.solve`
- [x] debugging the solve to get spectral accuracy
- [ ] vectorize the evaluation of the solver
- [ ] refactoring everything into a easy to use software package