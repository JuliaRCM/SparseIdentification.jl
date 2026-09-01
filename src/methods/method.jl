
const DEFAULT_LAMBDA = 0.05
const DEFAULT_NLOOPS = 10
const DEFAULT_INTEGRATOR_TIMESTEP = 0.01

# Fixed-point iterations for the implicit midpoint step of the flow-map loss. Four is what the
# original implementation used; it is a fixed count and not a convergence test, which is why it
# is a documented parameter rather than a literal buried in the loss.
const DEFAULT_PICARD_ITERATIONS = 4

"""
    SparsificationMethod

Supertype of the identification methods, e.g. [`SINDy`](@ref) and [`HamiltonianSINDy`](@ref).

A method carries the basis it searches and the parameters of its sparsification. It is applied to
an [`IdentificationProblem`](@ref) with [`identify`](@ref), mirroring the way a
`GeometricIntegrators` method is applied to a problem with `integrate`.

# Traits

`SparsificationMethod` extends `GeometricBase.AbstractMethod`, and the ecosystem's trait functions
are answered where they mean something for a regression method:

  - `issymplectic` and `isenergypreserving` report whether the *identified model* is
    symplectic and energy-preserving by construction. This is the substantive question for a
    structure-preserving method, and the reason `HamiltonianSINDy` exists.
  - `isexplicit` and `isimplicit` report whether the regression is a direct solve or
    goes through an implicit integrator.
  - `name`, `description` and `reference` identify the method and its source.

`issymmetric`, `isstifflyaccurate` and `order` describe a Runge–Kutta tableau and have no meaning
for a regression, so they are left as `missing`. `GeometricBase.isAbstractMethod` therefore returns
`false` for these methods, which is correct: it is a conformance check for *integrator* methods.
"""
abstract type SparsificationMethod <: GeometricBase.AbstractMethod end

"""
    basis(method::SparsificationMethod)

The library of candidate functions the method searches.
"""
GeometricBase.basis(method::SparsificationMethod) = method.basis

"""
    λ(method::SparsificationMethod)

The sparsification threshold: coefficients below it are set to zero after each fit.
"""
sparsity_threshold(method::SparsificationMethod) = method.λ

"""
    nloops(method::SparsificationMethod)

The cap on thresholding passes.
"""
nloops(method::SparsificationMethod) = method.nloops

"""
    identify(problem::IdentificationProblem, method::SparsificationMethod; kwargs...)

Identify the governing equations of `problem` with `method`.

This mirrors `GeometricIntegrators`' `integrate(problem, method)`: the problem carries the data,
the method carries the basis and the sparsification parameters, and the result carries the
identified coefficients.

Which methods accept which problems is not interchangeable, because the two formulations need
different data:

| problem | method | fits |
|:--|:--|:--|
| [`TrainingData`](@ref) | [`SINDy`](@ref) | a vector field, against measured derivatives |
| [`TrajectoryData`](@ref) | [`HamiltonianSINDy`](@ref) | a Hamiltonian, against the flow map |

Applying a method to a problem it does not accept raises an `ArgumentError` naming both, rather
than a `MethodError` from several layers down.

# Examples

```jldoctest
julia> A = [-0.1 2.0; -2.0 -0.1];

julia> x = randn(2, 400);

julia> result = identify(TrainingData(x, A * x), SINDy(CompoundBasis(polyorder = 3); λ = 0.05));

julia> nterms(result)
4
```
"""
function identify end

# Unimplemented (problem, method) combinations report what is missing rather than surfacing a
# `MethodError` from several layers down.
function identify(problem::IdentificationProblem, method::SparsificationMethod; kwargs...)
    throw(ArgumentError("$(typeof(method).name.name) cannot be applied to " *
                        "$(typeof(problem).name.name). See the documentation for which " *
                        "formulation each method needs."))
end
