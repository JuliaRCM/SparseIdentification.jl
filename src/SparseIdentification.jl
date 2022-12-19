module SparseIdentification

export MonomialBasis, TrigonometricBasis, CompoundBasis
export evaluate
export JuliaLeastSquare, OptimSolver
export solve
export sparse_galerkin, sparsify_dynamics

include("basis.jl")
include("solvers.jl")
include("sparse_galerkin.jl")
include("sparsify_dynamics.jl")

end
