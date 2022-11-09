# the integral equation has rank 1 deficiency, which is probably why
# the gmres is stagnating. So, probably, if I can extract the non-deficient 
# part of the integral equation, and run a gmres on it, it will stop stagnating.
from scipy.sparse.linalg import gmres, LinearOperator