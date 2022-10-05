# stokes 2d

This repository aims to build solvers of 2d Stokes equation using boundary integration method of stream function in [Greengard et al.](https://www.sciencedirect.com/science/article/pii/S0021999196901023?via%3Dihub) for some simple geometries: a straight pipe, a curved pipe, a pipe with multiple income and outcome branches, etc.

As pipes can be connected to form a network of pipes, our solvers can be connected to form a solver for a network of pipes. This gives us a fast and efficient way to build solver for 2d Stokes equation on complicated network of pipes.

On the other hand, this could also be used as a direct way of building solvers for 2d Stokes equation on bounded geometries with prescribed boundary velocity.

# credit

This is a summer project of Haiyang Wang as an undergraduate math student at NYU Courant, guided and advised by professor Greengard, Phd. Fryklund, and Phd. Samuel Potter.

# todo

## math-ish

- [ ] organize my python files. 
- [ ] test the matching, see if it really works well. 
- [ ] create different geometries
  - [ ] a simple straight pipe. 
  - [ ] some bands with fixed angles. 
  - [ ] Nlets 里面相临的两条线不能是平行的. 

## engineering-ish
- [ ] write a build script to be ran on a server. 
- [ ] change some instance of `flatten` to `ravel`. 
- [ ] store every data of the standard pieces. the data needs to include
- [ ] removing scipy

## games
  - [ ] how to use matplotlib with pygame
  - [ ] the coordinate system of pygame is stupid. Can I change that? [see stacks overflow](https://stackoverflow.com/questions/10167329/change-the-position-of-the-origin-in-pygame-coordinate-system#:~:text=Is%20it%20possible%20to%20change%20the%20coordinate%20system,and%20use%20it%20just%20before%20drawing%20any%20object.).

## probably later. 
- [ ] ask microfluids people what tube they are interested in.
- [ ] a solver for a multiply connected domain.
- [ ] think about particles. it is hard but think about it...


## done 
- [X] extracting grad pressure
- [x] extracting pressure difference across the tubes. this could be done with a path integral. More: it can be built into part of the solver. so in the web page we can simply take the pre-solved results easily and take a linear combination. the presolved idea applies to velocity field as well. We can simply store it, when using it we can simply take a linear combination.
- [X] find the appropriate energy function. explained in Fredrik's email, pressure drop is energy per unit volume. So we should only care about that.
- [X] understand how to handle geometry with corners by adding some caps and smooth corners.
  - [X] smooth caps
  - [X] smooth corners
- [X] vectorize the evaluation of the solver
- [x] (maybe later) ffm-ize the evaluation of the solver
- [X] refactoring everything into a easy to use software package
  - [X] refactoring the geometry class
- [X] debugging the solve to get spectral accuracy
- [X] using `gmres` instead of `np.linalg.solve`
- [X] Geometry objects refactoring
  - [X] make them sampler obeying the $5h$-rule is obeyed given a $\epsilon <= 5h$.
  - [X] make them sampler that satisfies the ledregre coefficients. 
  - [X] make them carrying the initial conditions of velocity:
    - [X] for non-cap geometry, it will be simply all zeros
    - [X] for cap geometry, it will need to be assigned a flux and then have a boolean for inlet or outlet. 
- [x] investigating how far away does the flow returns to poiseuille again: The key thing is to attempt to characterize how fast the return to Poiseuille occurs in a length-normalized setting.
  - [X] plot flow everywhere in the domain
  - [X] to visualize the "return to Poiseuille": to plot the error along the axis of symmetry of the pipe. Then, instead of a heat map, you can make a semilogy plot with "length along the tube" on the x-axis and absolute error on the y-axis. That is, plot a 1D slice of your image plots in a semilogy plot to make the error clearer.
  - [x] 😂 do the weird doubly obstructed tube with only one side with obstruction? so it will be asymmetric.
  - [x] Do a parameter study: it is likely that the relative size of your obstruction is what controls the length over which the return to Poiseuille occurs. I recommend creating a parametrized version of your geometry where you can control the size of the obstruction with a single scalar parameter. I doubt there is any good reason to have two bumps on either side of the channel---a simple thing you could do is add a (smoothed?) semicircular "bite" on either side of the pipe of radius r. Then you can try plotting the "deviation from Poiseuille" with respect to the scale parameter r/R.
  - [x] check whether this provides an upper bound on how fast the return to Poiseuille happens. E.g., if you can fit one obstruction inside of another obstruction, will the return happen for the smaller, contained obstruction at least as fast as for the larger obstruction? Probably Fredrik or Leslie can weigh in and give some insight as to whether this is likely to be the case... I don't know, myself.
  - [x] set a tol for return to poiseuille, varying the obstruction height, and observe how fast it is returning.
  - [x] measure the difference of your flow to a Poiseuille flow at distance 7.5 from the obstruction (there you seem to reach the error imposed by the GMRES tolerance) as you increase N? Please plot this error vs the number of points on the boundary in log log scale.
  - [x] Nlets
    - [x] Y-shaped with varying parameter for optimization
    - [x] cross. 
- [x] an algorithm that connects the pipes. 
  - [x] abstract pipe: 
    - [x] lets
        - [x] position: matching_pt
        - [x] dir
        - [x] radius
    - [x] flows (fluxes) (lets[1:])
        - [x] the pressure drops
        - [x] fluxes
    - each lets is a vertex, each flow is an edge.  
  - [x] aggregate those pipes into a pipe system. 
- [x] analytically explain the return to poiseuille behaviour.   
- [x] plotting
  - [x] give a specific criterion to check if a point is in that geometry.
  - [x] the velocity field should be corrected near the boundary. 
    - [x] this essentially ask me to determine if a point is close to a curve, and such criterion should be 
    implemented for each curve to have the best performance: 
  - [x] pressure and vorticity field as well? 
  - [x] have a method for plotting the velocity field using perhaps imshow
  - [x] have a method for plotting the actual boundary of the geometry using a not-so-thin balck line.
  - [x] i can also plot vector field and stream line with number. How surprising.
  - [x] base points for each pipe. and when plotting, every coordinate needs to be shifted. 
- [x] make sure that no two pipe intersects: this requires calculating intersection of two polygons, I am not sure how to do it for now. 
- [x] package the gaussian quadrature rules from scipy into something like a json file?.