module SparseIdentification

using LinearAlgebra
using RuntimeGeneratedFunctions
using Symbolics

using GeometricBase
using GeometricEquations
using GeometricSolutions

using GeometricOptimizers: Optimizer, OptimizerState, BFGS, Backtracking

# `solve` belongs to SimpleSolvers and is re-exported by GeometricOptimizers. Importing it and
# adding methods keeps one generic function; defining a fresh `solve` here would either be a
# method-definition error or shadow theirs.
import SimpleSolvers: solve, solve!

# The verbs this package answers. GeometricBase declares most of these as bare stubs and exports
# only a handful, so they have to be imported by name before methods can be added, and re-exported
# for callers who reach them through this package.
import GeometricBase: basis, datatype, arrtype, equations, functions, nsamples, parameters,
                      timestep, name, description, reference,
                      isexplicit, isimplicit, issymmetric, issymplectic,
                      isenergypreserving, isstifflyaccurate, order

RuntimeGeneratedFunctions.init(@__MODULE__)

export basis, datatype, arrtype, equations, functions, nsamples, parameters, timestep
export name, description, reference

# `isexplicit`, `isimplicit`, `issymmetric`, `issymplectic`, `isenergypreserving`,
# `isstifflyaccurate` and `order` are imported above but deliberately NOT exported.
# GeometricIntegratorsBase defines its own generics of those names rather than extending
# GeometricBase's stubs, and exports them; a session with both `using SparseIdentification` and
# `using GeometricIntegrators` would then see two different bindings under one name and resolve it
# to neither. Reach them qualified — `SparseIdentification.issymplectic(method)` — as SimpleSolvers
# has callers do with `status` and `isconverged`, and for the same reason.
#
# Four of them carry methods here; `issymmetric`, `isstifflyaccurate` and `order` describe a
# Runge–Kutta tableau and are left at GeometricBase's `missing`.

export calculate_nparams, hamiltonian, hamil_trig, hamiltonian_poly

include("util.jl")

export AbstractBasis, PolynomialBasis, TrigonometricBasis, CompoundBasis
export ExponentialBasis, LogarithmicBasis, RationalBasis
export BasisArguments, StateComponents, Differences
export basis_functions, evaluate, ⊕

include("basis.jl")

export AbstractSolver, JuliaLeastSquare, OptimizerSolver
# `solve` is re-exported rather than redefined: it is SimpleSolvers' generic, and this package
# only adds methods to it. GeometricOptimizers re-exports it the same way.
export solve, minimize

include("solvers.jl")

export IdentificationProblem, TrainingData, TrajectoryData
export statedimension

include("trainingdata.jl")

export SparsificationMethod, VectorField
export identify, sparsify, nterms

include("methods/method.jl")
include("methods/vectorfield.jl")

export SINDy, SINDyResult, SINDyVectorField

include("methods/sindy.jl")

export HamiltonianSINDy, HamiltonianSINDyResult, HamiltonianSINDyVectorField
export HamiltonianFunctions, hamiltonian_functions, hamiltonian_basis
export strip_constants, degreesoffreedom

include("methods/hamiltonian.jl")
include("methods/hamiltonian_sindy.jl")

include("geometricequations.jl")

end
