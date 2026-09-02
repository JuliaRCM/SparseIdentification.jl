
"""
    AbstractBasis

Supertype of the candidate-function libraries a sparse regression selects from.

A basis is defined symbolically by [`basis_functions`](@ref), which returns its candidate functions
as `Symbolics` expressions in the state. Everything else follows from that one definition:
[`evaluate`](@ref) compiles the expressions into a fast numerical evaluator, and the Hamiltonian
methods differentiate them to build `J∇φₖ`. There is deliberately no second, numeric definition
that could drift from the symbolic one.

Concrete bases are [`PolynomialBasis`](@ref), [`TrigonometricBasis`](@ref),
[`ExponentialBasis`](@ref), [`LogarithmicBasis`](@ref), [`RationalBasis`](@ref) and
[`CompoundBasis`](@ref).
"""
abstract type AbstractBasis end

"""
    basis_functions(basis, z)

The candidate functions of `basis` as symbolic expressions in the state vector `z`.

This is the definition of a basis; `evaluate` is generated from it.
"""
function basis_functions end

"""
    nterms(basis, d)

The number of candidate functions the basis contributes for a state of dimension `d`.
"""
nterms(basis::AbstractBasis, d::Int) = length(basis_functions(basis, _symbolic_state(d)))

_symbolic_state(d::Int) = Symbolics.variables(:z, 1:d)

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Arguments: what a univariate candidate function is applied to.
#
# This is the piece that makes the thesis's examples expressible. Applying `exp` to individual
# state components gives `e^{q₁}, e^{q₂}, …`, which is useless for a Toda lattice — that needs
# `e^{-(qₙ₊₁ - qₙ)}`, an exponential of a *difference*. Likewise a point-vortex Hamiltonian needs
# `log|qᵢ - qⱼ|` and an N-body one `1/|qᵢ - qⱼ|`.
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    BasisArguments

Supertype of the argument selections a univariate basis can be applied to:
[`StateComponents`](@ref) and [`Differences`](@ref).
"""
abstract type BasisArguments end

"""
    StateComponents()

Apply the basis functions to each state component separately: `f(z₁), f(z₂), …`.
"""
struct StateComponents <: BasisArguments end

"""
    Differences(indices; consecutive = false)

Apply the basis functions to differences of the state components named by `indices`.

With `consecutive = true` only neighbouring differences `z[i₊₁] - z[i]` are formed, which is what a
lattice with nearest-neighbour interaction needs. Otherwise all pairs `z[i] - z[j]` with `i > j`
are formed, which is what an all-to-all interaction needs.

# Examples

A Toda lattice of four particles interacts through consecutive differences of the *positions*, the
first four of eight phase-space components:

```jldoctest
julia> args = Differences(1:4; consecutive = true);

julia> basis = ExponentialBasis(args; rates = (-1.0,));

julia> nterms(basis, 8)
3
```
"""
struct Differences <: BasisArguments
    indices::Vector{Int}
    consecutive::Bool

    function Differences(indices; consecutive::Bool = false)
        idx = collect(Int, indices)
        length(idx) ≥ 2 ||
            throw(ArgumentError("need at least two indices to form a difference, got $idx"))
        allunique(idx) || throw(ArgumentError("indices must be unique, got $idx"))
        new(idx, consecutive)
    end
end

basis_arguments(::StateComponents, z) = collect(z)

function basis_arguments(a::Differences, z)
    idx = a.indices
    maximum(idx) ≤ length(z) ||
        throw(ArgumentError("Differences references component $(maximum(idx)) but the state has " *
                            "only $(length(z)) components"))

    if a.consecutive
        [z[idx[i + 1]] - z[idx[i]] for i in 1:(length(idx) - 1)]
    else
        [z[idx[i]] - z[idx[j]] for i in eachindex(idx) for j in 1:(i - 1)]
    end
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Bases
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    PolynomialBasis(p)

All monomials of degree exactly `p`, without repetition.

Degree 0 is the constant. For a state of dimension `d` there are `binomial(d + p - 1, p)` terms of
degree `p`.
"""
struct PolynomialBasis <: AbstractBasis
    p::Int
end

function basis_functions(basis::PolynomialBasis, z)
    basis.p == 0 ? [Num(1)] : Num.(hamiltonian_poly(collect(z), basis.p))
end

"""
    TrigonometricBasis(n, args = StateComponents())

The basis functions `sin(k u)` and `cos(k u)` for `1 ≤ k ≤ n`, over the arguments `args`.
"""
struct TrigonometricBasis{A <: BasisArguments} <: AbstractBasis
    n::Int
    args::A

    function TrigonometricBasis(n::Int, args::A = StateComponents()) where {A <:
                                                                            BasisArguments}
        new{A}(n, args)
    end
end

function basis_functions(basis::TrigonometricBasis, z)
    u = basis_arguments(basis.args, z)
    out = Num[]
    for k in 1:(basis.n)
        append!(out, sin.(k .* u))
        append!(out, cos.(k .* u))
    end
    out
end

"""
    ExponentialBasis(args = StateComponents(); rates = (1.0,))

The basis functions `exp(α u)` for each rate `α` and each argument `u`.

The Toda lattice needs `exp(-(qₙ₊₁ - qₙ))`, so both a negative rate and a
[`Differences`](@ref) argument selection:

```jldoctest
julia> basis = ExponentialBasis(Differences(1:3; consecutive = true); rates = (-1.0,));

julia> nterms(basis, 6)
2
```
"""
struct ExponentialBasis{A <: BasisArguments, R} <: AbstractBasis
    args::A
    rates::R

    function ExponentialBasis(args::A = StateComponents();
            rates::R = (1.0,)) where {A <: BasisArguments, R}
        new{A, R}(args, rates)
    end
end

function basis_functions(basis::ExponentialBasis, z)
    u = basis_arguments(basis.args, z)
    Num[exp(α * uᵢ) for α in basis.rates for uᵢ in u]
end

"""
    LogarithmicBasis(args = StateComponents())

The basis functions `log(abs(u))` over the arguments `args`.

A point-vortex Hamiltonian is built from `log|qᵢ - qⱼ|`, so this is normally paired with
[`Differences`](@ref). `abs` is applied inside the logarithm so the basis is defined on both signs
of the argument; it is singular where the argument vanishes, which for a difference means two
coordinates coinciding.
"""
struct LogarithmicBasis{A <: BasisArguments} <: AbstractBasis
    args::A

    LogarithmicBasis(args::A = StateComponents()) where {A <: BasisArguments} = new{A}(args)
end

function basis_functions(basis::LogarithmicBasis, z)
    Num[log(abs(u)) for u in basis_arguments(basis.args, z)]
end

"""
    RationalBasis(args = StateComponents(); powers = (1,))

The basis functions `u^-k` for each `k` in `powers`, over the arguments `args`.

An N-body gravitational Hamiltonian is built from `1/|qᵢ - qⱼ|`, so this is normally paired with
[`Differences`](@ref). Singular where the argument vanishes.
"""
struct RationalBasis{A <: BasisArguments, P} <: AbstractBasis
    args::A
    powers::P

    function RationalBasis(args::A = StateComponents();
            powers::P = (1,)) where {A <: BasisArguments, P}
        new{A, P}(args, powers)
    end
end

function basis_functions(basis::RationalBasis, z)
    u = basis_arguments(basis.args, z)
    Num[uᵢ^(-k) for k in basis.powers for uᵢ in u]
end

"""
    CompoundBasis(bases...)
    CompoundBasis(; polyorder = 5, trigonometric = 0)

A basis assembled from several others, evaluated in the order given.

The keyword form builds the common case: the constant and all monomials up to `polyorder`,
optionally followed by trigonometric terms up to wavenumber `trigonometric`.

Bases are combined with `⊕` as well:

```jldoctest
julia> basis = CompoundBasis(polyorder = 2) ⊕ ExponentialBasis(; rates = (-1.0,));

julia> nterms(basis, 2)
8
```
"""
struct CompoundBasis{BT <: Tuple} <: AbstractBasis
    bases::BT

    CompoundBasis(bases::Tuple) = new{typeof(bases)}(bases)
    CompoundBasis(bases...) = new{typeof(bases)}(bases)
end

function CompoundBasis(; polyorder::Int = 5, trigonometric::Int = 0)
    bases = Tuple([PolynomialBasis(i) for i in 0:polyorder])

    if trigonometric > 0
        bases = (bases..., TrigonometricBasis(trigonometric))
    end

    CompoundBasis(bases)
end

bases(b::CompoundBasis) = b.bases
bases(b::AbstractBasis) = (b,)

function basis_functions(basis::CompoundBasis, z)
    out = Num[]
    for b in bases(basis)
        append!(out, basis_functions(b, z))
    end
    out
end

"""
    b₁ ⊕ b₂

Concatenate two bases into a [`CompoundBasis`](@ref).
"""
⊕(b₁::AbstractBasis, b₂::AbstractBasis) = CompoundBasis((bases(b₁)..., bases(b₂)...))

# No `Base.show` methods: the default struct display names every field, so a basis prints the
# parameters that distinguish it — `PolynomialBasis(3)` rather than `PolynomialBasis()`, and the
# argument selection and rates of the univariate bases. A summary line would have to drop those,
# and a basis that displays the same at every degree is worse than a verbose one.

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Evaluation
#
# The evaluator is compiled from the symbolic definition, once per (basis, state dimension) pair
# and cached. Building it on every call would dominate the cost of a fit; defining a second,
# hand-written numeric path would risk it drifting from the symbolic one.
# ─────────────────────────────────────────────────────────────────────────────────────────────

const EVALUATOR_CACHE = Dict{Tuple{Any, Int}, Any}()

function _evaluator(basis::AbstractBasis, d::Int)
    get!(EVALUATOR_CACHE, (basis, d)) do
        z = _symbolic_state(d)
        φ = basis_functions(basis, z)
        @RuntimeGeneratedFunction(build_function(φ, z)[1])
    end
end

"""
    evaluate(data, basis)

Evaluate every candidate function of `basis` on every snapshot of `data`.

`data` is a matrix whose columns are snapshots, a vector of state vectors, or a single state
vector. The result `Θ` has one row per snapshot and one column per candidate function.

# Examples

```jldoctest
julia> Θ = evaluate([1.0 2.0; 3.0 4.0], CompoundBasis(polyorder = 1));

julia> size(Θ)      # 2 snapshots × (1 constant + 2 linear) terms
(2, 3)
```
"""
function evaluate(data::AbstractMatrix, basis::AbstractBasis)
    _tabulate(_evaluator(basis, size(data, 1)), eachcol(data))
end

function evaluate(data::AbstractVector{<:Number}, basis::AbstractBasis)
    transpose(_evaluator(basis, length(data))(data))
end

function evaluate(data::AbstractVector{<:AbstractVector}, basis::AbstractBasis)
    _tabulate(_evaluator(basis, length(first(data))), data)
end

# `Θ` is allocated once and filled row by row. Concatenating the rows as they are produced copies
# the whole block accumulated so far on every snapshot, so it costs O(nsamples²) memory: on 2000
# snapshots of a ten-term basis that is 174 MB to build a 160 kB matrix.
function _tabulate(f, snapshots)
    isempty(snapshots) &&
        throw(ArgumentError("cannot evaluate a basis on an empty data set"))

    # The first evaluation fixes the element type and the number of candidate functions, which is
    # what the output is sized from.
    φ = f(first(snapshots))
    Θ = similar(φ, length(snapshots), length(φ))

    for (i, snapshot) in enumerate(snapshots)
        Θ[i, :] .= f(snapshot)
    end

    return Θ
end
