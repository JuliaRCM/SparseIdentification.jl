module SparseIdentification

using Distances
using GeometricIntegrators
using RuntimeGeneratedFunctions
using Symbolics

RuntimeGeneratedFunctions.init(@__MODULE__)


export calculate_nparams

include("util.jl")

export PolynomialBasis, TrigonometricBasis, CompoundBasis

include("basis.jl")

export JuliaLeastSquare, OptimSolver
export solve

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

export HamiltonianSINDy, HamiltonianSINDyVectorField

include("methods/hamiltonian.jl")
include("methods/hamiltonian_sindy.jl")

end
