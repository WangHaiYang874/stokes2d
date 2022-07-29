# stokes 2d

This repository aims to build solvers of 2d Stokes equation using boundary integration method of stream function in [Greengard et al.](https://www.sciencedirect.com/science/article/pii/S0021999196901023?via%3Dihub) for some simple geometries: a straight pipe, a curved pipe, a pipe with multiple income and outcome branches, etc.

As pipes can be connected to form a network of pipes, our solvers can be connected to form a solver for a network of pipes. This gives us a fast and efficient way to build solver for 2d Stokes equation on complicated network of pipes.

On the other hand, this could also be used as a direct way of building solvers for 2d Stokes equation on bounded geometries with prescribed boundary velocity.

# credit

This is a summer project of Haiyang Wang as an undergraduate math student at NYU Courant, guided and advised by professor Greengard, Phd. Fryklund, and Phd. Samuel Potter.

# todo

## math-ish

- [X] extracting grad pressure
- [ ] extracting pressure difference across the tubes. this could be done with a path integral. More: it can be built into part of the solver. so in the web page we can simply take the pre-solved results easily and take a linear combination. the presolved idea applies to velocity field as well. We can simply store it, when using it we can simply take a linear combination.
- [X] find the appropriate energy function. explained in Fredrik's email, pressure drop is energy per unit volume. So we should only care about that.
- [X] understand how to handle geometry with corners by adding some caps and smooth corners.
  - [X] smooth caps
  - [X] smooth corners
- [ ] create different geometries
  - [ ] Y-shaped with varying parameter for optimization
  - [ ] some bands with fixed angles
  - [ ] pipes with different radius at inflow and outflow
  - [ ] ask microfluids people what shape they are interested in.
- [ ] a solver for a multiply connected domain.
- [ ] think about particles. it is hard but think about it...

## engineering-ish

- [X] vectorize the evaluation of the solver
- [ ] ffm-ize the evaluation of the solver
- [X] refactoring everything into a easy to use software package

  - [X] refactoring the geometry class
- [X] debugging the solve to get spectral accuracy
- [X] using `gmres` instead of `np.linalg.solve`
- [ ] redesign the geometric objects,

  - [ ] make them sampler obeying the $5h$-rule is obeyed given a $\epsilon <= 5h$.
  - [ ] make them carrying the initial conditions of velocity:

    - [ ] for non-cap geometry, it will be simply all zeros
    - [ ] for cap geometry, it will need to be assigned a flux and then have a boolean for inlet or outlet.
  - [ ] for closed geometry, or the standard pieces

    - [ ] specifically include the points/path of integration in it for the evaluation of pressure difference, or simply gives its the rule of integration.
    - [ ] specifically include the points on its corners to make sure that it will be matchable in the grid of the game. this needs to ignore the caps, which are artificial math construct for smoothness.
    - [ ] specifically include the points that velocity needs to be evaluated inside the geometry
    - [ ] give a specific criterion to check if a point is in that geometry: is this really necessary? no
    - [ ] plotting
      - [ ] have a method for plotting the velocity field using perhaps imshow
      - [ ] have a method for plotting the actual boundary of the geometry using a not-so-thin balck line.
- [ ] store every data of the standard pieces. the data needs to include

  - [ ] solver( or solvers for geometry with multiple inlets/outlets)
    - [ ] given fluxed at inlets and outlets, it should be able to compute the pressure difference, the velocity field
  - [ ] infact, the pressure difference and velocity field should be precomputed and taken linear combinations
- [ ] package the gaussian quadrature rules from scipy into something like a json file? Perhaps I need $n=16,32,64,\cdots 4096$. 

## maybe later

- [ ] investigating how far away does the flow returns to poiseuille again: The key thing is to attempt to characterize how fast the return to Poiseuille occurs in a length-normalized setting.
  - [X] plot flow everywhere in the domain
  - [X] to visualize the "return to Poiseuille": to plot the error along the axis of symmetry of the pipe. Then, instead of a heat map, you can make a semilogy plot with "length along the tube" on the x-axis and absolute error on the y-axis. That is, plot a 1D slice of your image plots in a semilogy plot to make the error clearer.
  - [ ] 😂 do the weird doubly obstructed tube with only one side with obstruction? so it will be asymmetric.
  - [ ] Do a parameter study: it is likely that the relative size of your obstruction is what controls the length over which the return to Poiseuille occurs. I recommend creating a parametrized version of your geometry where you can control the size of the obstruction with a single scalar parameter. I doubt there is any good reason to have two bumps on either side of the channel---a simple thing you could do is add a (smoothed?) semicircular "bite" on either side of the pipe of radius r. Then you can try plotting the "deviation from Poiseuille" with respect to the scale parameter r/R.
  - [ ] check whether this provides an upper bound on how fast the return to Poiseuille happens. E.g., if you can fit one obstruction inside of another obstruction, will the return happen for the smaller, contained obstruction at least as fast as for the larger obstruction? Probably Fredrik or Leslie can weigh in and give some insight as to whether this is likely to be the case... I don't know, myself.
  - [ ] set a tol for return to poiseuille, varying the obstruction height, and observe how fast it is returning.
  - [ ] measure the difference of your flow to a Poiseuille flow at distance 7.5 from the obstruction (there you seem to reach the error imposed by the GMRES tolerance) as you increase N? Please plot this error vs the number of points on the boundary in log log scale.
- [ ] web development.
  - [ ] consult someone on what frameworks to use: probably just javascript. Or I can use the pygame-wasm without scipy.
  - [X] need to have gui, scaling and draging geometry shapes.
  - [ ] need to plot the velocity field. this should be done with an fmm. Now notice that I mentioned about pre-compute the velocity field. so probably we don't really need to use fmm. But it will be good to learn fmm.
- [ ] misc
  - [ ] is it possible to produce a general purpose fmm?
