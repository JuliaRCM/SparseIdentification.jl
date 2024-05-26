##########################################################
# Primal Bases
##########################################################

"""
Generates combinations of polynomial bases of a specified order.

# Arguments
- `z`: A vector of symbolic variables.
- `order`: The desired order for the polynomial combination.
- `inds`: Indices for recursion, used internally.

# Returns
- A symbolic vector of polynomial combinations of the specified order.
"""
function poly_combos(z, order, inds...)
    if order == 0
        return Num[1]
    elseif order == length(inds)
        return [_prod([z[i] for i in inds]...)]
    else
        start_ind = length(inds) == 0 ? 1 : inds[end]
        return vcat([poly_combos(z, order, inds..., j) for j in start_ind:length(z)]...)
    end
end

"""
Generates all monomial basis functions up to a specified order.

# Arguments
- `z`: A vector of symbolic variables.
- `order`: The maximum order of the monomials.

# Returns
- A symbolic vector of monomial basis functions.
"""
function primal_monomial_basis(z, order::Int)
    return Vector{Symbolics.Num}(vcat([poly_combos(z, i) for i in 1:order]...))
end

"""
Generates coefficient basis functions up to a specified maximum coefficient.

# Arguments
- `z`: A vector of symbolic variables.
- `max_coeff`: The maximum coefficient value.

# Returns
- A symbolic vector of coefficient basis functions.
"""
function primal_coeff_basis(z, max_coeff::Int)
    return Vector{Symbolics.Num}(vcat([k .* z for k in 1:max_coeff]...))
end

"""
Generates a set of basis functions using the specified operator between states.

# Arguments
- `z`: A vector of symbolic variables.
- `operator`: The operator function to be applied (e.g., +, -, *, /).

# Returns
- A symbolic vector of basis functions created by applying the operator between states.
"""
function primal_operator_basis(z, operator)
    return Vector{Symbolics.Num}([operator(z[i], z[j]) for i in 1:length(z)-1 for j in i+1:length(z)] ∪ [operator(z[j], z[i]) for i in 1:length(z)-1 for j in i+1:length(z)])
end

"""
Generates power basis functions up to a specified maximum power.

# Arguments
- `z`: A vector of symbolic variables.
- `max_power`: The maximum power value.

# Returns
- A symbolic vector of power basis functions.
"""
function primal_power_basis(z, max_power::Int)
    if max_power > 0
        return Vector{Symbolics.Num}(vcat([z.^i for i in 1:max_power]...))
    elseif max_power < 0
        return Vector{Symbolics.Num}(vcat([z.^-i for i in 1:abs(max_power)]...))
    end
end

##########################################################
# Function Bases
##########################################################

"""
Generates a polynomial basis set.

# Arguments
- `z`: A vector of symbolic variables. Defaults to a 2-dimensional vector if not provided.
- `polyorder`: The maximum order of the polynomials.
- `operator`: An optional operator to apply between states.
- `max_coeff`: The maximum coefficient value.

# Returns
- A symbolic vector of polynomial basis functions.
"""
function polynomial_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = primal_monomial_basis(z, polyorder)
    primes = vcat(primes, primal_coeff_basis(z, max_coeff))
    if operator !== nothing
        primes = vcat(primes, primal_operator_basis(z, operator))
    end
    return primes
end

"""
Generates a trigonometric basis set.

# Arguments
- `z`: A vector of symbolic variables. Defaults to a 2-dimensional vector if not provided.
- `polyorder`: The maximum order of the polynomials.
- `operator`: An optional operator to apply between states.
- `max_coeff`: The maximum coefficient value.

# Returns
- A symbolic vector of trigonometric basis functions (sine and cosine).
"""
function trigonometric_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    return vcat(sin.(primes), cos.(primes))
end

"""
Generates an exponential basis set.

# Arguments
- `z`: A vector of symbolic variables. Defaults to a 2-dimensional vector if not provided.
- `polyorder`: The maximum order of the polynomials.
- `operator`: An optional operator to apply between states.
- `max_coeff`: The maximum coefficient value.

# Returns
- A symbolic vector of exponential basis functions.
"""
function exponential_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    return exp.(primes)
end

"""
Generates a logarithmic basis set.

# Arguments
- `z`: A vector of symbolic variables. Defaults to a 2-dimensional vector if not provided.
- `polyorder`: The maximum order of the polynomials.
- `operator`: An optional operator to apply between states.
- `max_coeff`: The maximum coefficient value.

# Returns
- A symbolic vector of logarithmic basis functions.
"""
function logarithmic_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    return log.(abs.(primes))
end

"""
Generates a mixed basis set from multiple provided basis vectors.

# Arguments
- `basis`: One or more vectors of symbolic basis functions.

# Returns
- A symbolic vector of mixed basis functions.
"""
function mixed_states_basis(basis::Vector{Symbolics.Num}...)
    mixed_states = Tuple(basis)
    
    ham = Vector{Symbolics.Num}()
    for i in eachindex(mixed_states)
        for j in i+1:lastindex(mixed_states)
            ham = vcat(ham, [mixed_states[i][k] * mixed_states[j][l] for k in 1:length(mixed_states[i]) for l in 1:length(mixed_states[j])])
        end
    end
    
    return Vector{Symbolics.Num}(ham)
end
