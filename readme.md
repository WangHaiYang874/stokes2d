# stokes 2d

This repository aims to build solvers of 2d Stokes equation using boundary integration method of stream function in [Greengard et al.](https://www.sciencedirect.com/science/article/pii/S0021999196901023?via%3Dihub) for some simple geometries: a straight pipe, a curved pipe, a pipe with multiple income and outcome branches, etc. 

As pipes can be connected to form a network of pipes, our solvers can be connected to form a solver for a network of pipes. This gives us a fast and efficient way to build solver for 2d Stokes equation on complicated network of pipes. 

On the other hand, this could also be used as a direct way of building solvers for 2d Stokes equation on bounded geometries with prescribed boundary velocity. 

# credit
This is a summer project of Haiyang Wang as an undergraduate math student at NYU Courant, guided and advised by professor Greengard, Phd. Fryklund, and Phd. Samuel Potter. 


# todo
## math-ish
- [x] extracting pressure
- [ ] understand how to handle geometry with corners by adding some caps and smooth corners. 
  - [x]  smooth caps
  - [ ]  smooth corners 
- [ ] investigating how far away does the flow returns to poiseuille again: The key thing is to attempt to characterize how fast the return to Poiseuille occurs in a length-normalized setting.
  - [ ] plot flow everywhere in the domain
  - [ ] to visualize the "return to Poiseuille": to plot the error along the axis of symmetry of the pipe. Then, instead of a heat map, you can make a semilogy plot with "length along the tube" on the x-axis and absolute error on the y-axis. That is, plot a 1D slice of your image plots in a semilogy plot to make the error clearer.
  - [ ] Do a parameter study: it is likely that the relative size of your obstruction is what controls the length over which the return to Poiseuille occurs. I recommend creating a parametrized version of your geometry where you can control the size of the obstruction with a single scalar parameter. I doubt there is any good reason to have two bumps on either side of the channel---a simple thing you could do is add a (smoothed?) semicircular "bite" on either side of the pipe of radius r. Then you can try plotting the "deviation from Poiseuille" with respect to the scale parameter r/R.
  - [ ] check whether this provides an upper bound on how fast the return to Poiseuille happens. E.g., if you can fit one obstruction inside of another obstruction, will the return happen for the smaller, contained obstruction at least as fast as for the larger obstruction? Probably Fredrik or Leslie can weigh in and give some insight as to whether this is likely to be the case... I don't know, myself.


## engineering-ish
- construct a `Y` shaped tube
  - do the investigation on how far away does the flow returns to poiseuille again. 
- [ ] vectorize the evaluation of the solver
- [ ] refactoring everything into a easy to use software package
  - [ ] refactoring the geometry class
- [x] debugging the solve to get spectral accuracy
- [x] using `gmres` instead of `np.linalg.solve`