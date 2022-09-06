
# pipe network

1. connecting pipes. (matching boundaries and graph) 
2. build the global solvers given boundary conditions. 
3. draw stuff

#### attributes

1. `pipes`, a list of pipes in this network
2. `lets`, a list of inlets and outlets

#### method
1. `get_velocity`
1. `get_pressure`
1. `get_png`
2. `solve`, given velocity at the boundaries, solve for the linear combinations. 
3. `build_solver`:
   1. build the local solvers, this should be done at the pipe level. 
   2. build the global solvers, this should be done by graph theory...


# pipe

This class primarily needs the following functionalities
1. draw stuff.
   2. prebuild the velocity field. 
2. solve. 
   1. for those solvers: the flux difference is already given
   2. build the pressure drop matrix. 
   3. several `linear_independent` stokes equations (#=number of inlets and outlets - 1)
3. connection. 



# curve

1. geometric data
2. capability of providing a refined quad rule for each panel. 
3. velocity boundary conditions.

# graph
todo

#### attribute

1. `V`, vertices of the graph. it should be a list with the following data:
   1. `(x,y)` the coordinate of a single vertice
   2. `out_normal_direction`, this is for legitimate matching. 
   3. `width`, this is for legitimate matching. 
2. `E`, edges of the graph, it should be a list of index of the vertices. 

#### method

1. `join`, this is a classmethod that joins a list of graphs. 
2. `cycles`, find basic cycles of the graph. 
3. 