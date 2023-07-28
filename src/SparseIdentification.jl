module SparseIdentification

using Distances
using GeometricIntegrators
using RuntimeGeneratedFunctions
using Symbolics

RuntimeGeneratedFunctions.init(@__MODULE__)


export get_z_vector, get_numCoeffs, get_basis_set

include("util.jl")

export PolynomialBasis, TrigonometricBasis, CompoundBasis

include("basis.jl")

export JuliaLeastSquare, OptimSolver, NNSolver
export solve, sparse_solve

include("solvers.jl")

export sparsify, lorenz, sparsify_hamiltonian_dynamics

include("lorenz.jl")

export TrainingData

include("trainingdata.jl")

export SparsificationMethod, VectorField

include("methods/method.jl")
include("methods/vectorfield.jl")

export SINDy, SINDyVectorField

include("methods/sindy.jl")

export poly_combos, primal_monomial_basis, primal_coeff_basis, primal_operator_basis, primal_power_basis
export polynomial_basis, trigonometric_basis, exponential_basis, logarithmic_basis, mixed_states_basis

include("hamiltonian_basis.jl")

export HamiltonianSINDy, HamiltonianSINDyVectorField

include("methods/hamiltonian_sindy.jl")
include("methods/hamiltonian.jl")

end