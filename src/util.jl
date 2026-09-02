# Elementwise product of any number of arrays, used to form a monomial from its chosen factors.
_prod(a, b, c, arrs...) = a .* _prod(b, c, arrs...)
_prod(a, b) = a .* b
_prod(a) = a

"""
    calculate_nparams(d, polyorder, trig_wave_num)

The number of coefficients a Hamiltonian basis carries for `d` degrees of freedom.

The phase space has `2d` dimensions, so the polynomial part counts the monomials of degree up to
`polyorder` in `2d` variables. The trigonometric part, when `trig_wave_num > 0`, adds `sin` and
`cos` at each wave number for each variable.
"""
function calculate_nparams(d, polyorder, trig_wave_num)
    # The monomials of degree up to `polyorder` in `2d` variables number
    # `binomial(2d + polyorder, polyorder)`, counting each monomial once and allowing a variable to
    # repeat within one. The constant is subtracted: it shifts `H` without changing `∇H`, so it is
    # not identifiable from trajectory data.
    nparam = binomial(2d + polyorder, polyorder) - 1

    if trig_wave_num > 0
        # first 2 in the product formula b/c the trig basis are sin and cos i.e. two basis functions
        # 2d: b/c the phase space is two variables p,q each with 2 dims
        nparam += 2 * trig_wave_num * 2d
    end

    return nparam
end

""" 
    hamiltonian_poly(z, order, inds...)

All monomials of degree exactly `order` in the variables `z`, each one once.

A variable may repeat within a monomial, so degree 2 in two variables gives `z₁², z₁z₂, z₂²`.

Shared by `PolynomialBasis` and by the symbolic Hamiltonian construction, so that the two cannot
disagree about which terms exist or in what order.
"""
function hamiltonian_poly(z, order, inds...)
    ham = Num[]

    if order == 0
        # Degree zero is the constant, and it is the one degree with a single term regardless of
        # how many variables there are. Returning it here rather than at the call site is what
        # keeps `PolynomialBasis` from needing a second definition of the same thing.
        push!(ham, Num(1))
    elseif order == length(inds)
        push!(ham, _prod([z[i] for i in inds]...))
    else
        # Starting at `inds[end]` rather than at `inds[end] + 1` is what admits a repeated
        # variable while still generating each monomial only once.
        start_ind = isempty(inds) ? 1 : inds[end]
        for j in start_ind:length(z)
            append!(ham, hamiltonian_poly(z, order, inds..., j))
        end
    end

    return ham
end
