module SparseIdentification

using RuntimeGeneratedFunctions
using Symbolics
using DifferentialEquations
using ODE

RuntimeGeneratedFunctions.init(@__MODULE__)


export calculate_nparams, hamiltonian, hamil_trig

include("util.jl")

export PolynomialBasis, TrigonometricBasis, CompoundBasis
export evaluate

include("basis.jl")

export poolDataLIST

include("poolDataLIST.jl")

export JuliaLeastSquare, OptimSolver
export solve

include("solvers.jl")

export sparsify, lorenz, sparsify_hamiltonian_dynamics

include("lorenz.jl")

export hamilGradient!, hamiltonianFunction

# include("hamiltonianGenerator.jl")

export hamil_basis_maker, hamiltonian_basis_concat

include("hamiltonian_basis_maker.jl")





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
