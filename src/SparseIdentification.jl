module SparseIdentification

export _prod, calculate_nparams, hamiltonian, hamil_trig

include("util.jl")

export PolynomialBasis, TrigonometricBasis, CompoundBasis
export evaluate

include("basis.jl")

export poolDataLIST

include("poolDataLIST.jl")

export JuliaLeastSquare, OptimSolver
export solve

include("solvers.jl")

export sparse_galerkin!, sparsify_dynamics, lorenz, sparsify_hamiltonian_dynamics

include("sparse_galerkin.jl")
include("sparsify_dynamics.jl")
include("lorenz.jl")

export hamilGradient!, hamiltonianFunction

include("hamiltonianGenerator.jl")

export hamil_basis_maker, hamiltonian_basis_concat

include("hamiltonian_basis_maker.jl")

export hamilGrad_func_builder, hamilGradient_general!

include("hamilGradGen2.jl")

end
