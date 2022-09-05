
# pipe_network

1. connecting pipes. (matching boundaries and graph) 
2. build the global solvers given boundary conditions. 
3. draw the (velocity-field/pressure/vorticity) at each points inside this network. 
4. draw different part of the network. 


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
1. provide criterion for connecting with other pipes. this would pass to a separate graph object. 
2. solve several `linear_independent` stokes equations (#=number of inlets and outlets - 1)
   1. for those solvers: the flux difference is already given
   2. build the pressure drop matrix. 
3. vector field and etc. 
4. building the geometry with panels that satisfy certain rules. 
   1. legendre coefficients
   2. max_distance
   3. maybe build double geometry, to have a better quadrature rules near the boundary. 
5. building the graph with all the solved data. 
6. building the velocity field, the data of velocity and vorticity. 


# curve

1. build the geometry up to certain creterion


# graph

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