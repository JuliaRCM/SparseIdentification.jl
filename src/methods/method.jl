"""
    DEFAULT_LAMBDA

The default sparsification threshold used in the SINDy method. 
This value determines which coefficients are considered small and should be set to zero.
"""
const DEFAULT_LAMBDA = 0.05

"""
    DEFAULT_NOISE_LEVEL

The default level of noise added to the time derivatives of the state variables or the state variables themselves in the SINDy method. 
This value helps in making the system more robust to noise.
"""
const DEFAULT_NOISE_LEVEL = 0.05

"""
    DEFAULT_NLOOPS

The default number of sparsification loops in the SINDy method. 
This value defines how many times the sparsification process is repeated to achieve the desired sparsity.
"""
const DEFAULT_NLOOPS = 10

"""
    DEFAULT_t₂_DATA_TIMESTEP

The default time step used for data generation or integration in the SINDy method. 
This value is used when discretizing time for simulations or data processing.
"""
const DEFAULT_t₂_DATA_TIMESTEP = 0.01

"""
    SparsificationMethod

An abstract type representing a general sparsification method. 
Specific sparsification methods, such as SINDy, should subtype this abstract type.
"""
abstract type SparsificationMethod end
