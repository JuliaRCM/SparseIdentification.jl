module SparseIdentification

export PolynomialBasis, TrigonometricBasis, CompoundBasis
export evaluate

include("basis.jl")

export JuliaLeastSquare, OptimSolver
export solve

include("solvers.jl")

export sparse_galerkin, sparsify_dynamics

include("sparse_galerkin.jl")
include("sparsify_dynamics.jl")

end
