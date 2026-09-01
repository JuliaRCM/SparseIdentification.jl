
const DEFAULT_LAMBDA = 0.05
const DEFAULT_NLOOPS = 10
const DEFAULT_INTEGRATOR_TIMESTEP = 0.01

# Fixed-point iterations for the implicit midpoint step of the flow-map loss. Four is what the
# original implementation used; it is a fixed count and not a convergence test, which is why it
# is a documented parameter rather than a literal buried in the loss.
const DEFAULT_PICARD_ITERATIONS = 4

abstract type SparsificationMethod end
