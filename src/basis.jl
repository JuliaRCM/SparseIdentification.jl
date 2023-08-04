

abstract type AbstractBasis end

"""
    evaluate(data, basis)

Evaluates a data set on all basis functions in `basis`.
"""
function evaluate end


"""
    PolynomialBasis(p)

Holds polynomials of degree p.
"""
struct PolynomialBasis <: AbstractBasis
    p::Int
end

function _evaluate_polynomial(data, p, inds...)
    # number of degrees of freedom
    ndof = size(data,2)

    # number of snapshots
    ns = size(data,1)

    # initialize output array
    out = zeros(ns,0)

    if p == 0
        tmp = ones(ns,1)
        out = hcat(out, tmp)
    elseif p == length(inds)
        tmp = _prod([data[:,i] for i in inds]...)
        out = hcat(out, tmp)
    else
        start_ind = length(inds) == 0 ? 1 : inds[end]
        for j in start_ind:ndof
            tmp = _evaluate_polynomial(data, p, inds..., j)
            out = hcat(out, tmp)
        end
    end

    return out
end

function (basis::PolynomialBasis)(data::AbstractArray)
    _evaluate_polynomial(data', basis.p)
end


"""
    TrigonometricBasis(n)

Holds the basis functions [sin(kx), cos(kx)] for 1 ≤ k ≤ n.
"""
struct TrigonometricBasis <: AbstractBasis
    n::Int
end

# function (basis::TrigonometricBasis)(data::AbstractVector)
#     # initialize output array
#     out = zeros(0)

#     for k in 1:basis.n
#         tmp = [sin(k*data), cos(k*data)]
#         out = hcat(out, tmp)
#     end

#     return out
# end

function (basis::TrigonometricBasis)(data::AbstractArray)
    # number of snapshots
    ns = size(data,2)

    # initialize output array
    out = zeros(ns,0)

    for k in 1:basis.n
        tmp = [sin(k*data), cos(k*data)]
        out = hcat(out, tmp)
    end

    return out
end


"""
    CompoundBasis(bases)

Holds a basis composed of several different basis functions,
e.g. polynomials of variable data degree and/or trigonometric functions.
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

# function (basis::CompoundBasis)(data::AbstractVector)
#     # initialize output array
#     out = zeros(0)

#     # loop over bases
#     for b in bases(basis)
#         out = hcat(out, b(data))
#     end

#     return out
# end

function (basis::CompoundBasis)(data::AbstractArray)
    # number of snapshots
    ns = size(data,2)

    # initialize output array
    out = zeros(ns,0)

    # loop over bases
    for b in bases(basis)
        out = hcat(out, b(data))
    end

    return out
end
