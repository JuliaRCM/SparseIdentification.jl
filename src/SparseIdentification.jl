module SparseIdentification

using LinearAlgebra
using RuntimeGeneratedFunctions
using Symbolics

using GeometricOptimizers: Optimizer, OptimizerState, BFGS, Backtracking

# `solve` belongs to SimpleSolvers and is re-exported by GeometricOptimizers. Importing it and
# adding methods keeps one generic function; defining a fresh `solve` here would either be a
# method-definition error or shadow theirs.
import SimpleSolvers: solve, solve!

RuntimeGeneratedFunctions.init(@__MODULE__)

export calculate_nparams, hamiltonian, hamil_trig

include("util.jl")

export AbstractBasis, PolynomialBasis, TrigonometricBasis, CompoundBasis
export evaluate

include("basis.jl")

export AbstractSolver, JuliaLeastSquare, OptimizerSolver
export solve, minimize

include("solvers.jl")

export lorenz

include("lorenz.jl")

export TrainingData, TrajectoryData
export nsnapshots, statedimension

include("trainingdata.jl")

export SparsificationMethod, VectorField
export sparsify

include("methods/method.jl")
include("methods/vectorfield.jl")

export SINDy, SINDyVectorField

include("methods/sindy.jl")

export HamiltonianSINDy, HamiltonianSINDyVectorField

include("methods/hamiltonian.jl")
include("methods/hamiltonian_sindy.jl")

end
