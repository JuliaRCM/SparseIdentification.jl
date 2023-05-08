##########################################################
# Primal Bases
##########################################################

# make combinations of bases of just the order that is given 
# e.g order = 2 will give just the bases whose powers sum to 2
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

# gives all bases monomials up to a certain order
function primal_monomial_basis(z, order::Int)
    return Vector{Symbolics.Num}(vcat([poly_combos(z, i) for i in 1:order]...))
end

# calculates coefficient bases up to a certain order
# mostly for use with trigonometric functions example sin(k*z),
# where k is the coefficient
function primal_coeff_basis(z, max_coeff::Int)
    return Vector{Symbolics.Num}(vcat([k .* z for k in 1:max_coeff]...))
end

# calculates +,-,*,/ between states as a new basis
# the return output is a set to avoid duplicates
function primal_operator_basis(z, operator)
    return Vector{Symbolics.Num}([operator(z[i], z[j]) for i in 1:length(z)-1 for j in i+1:length(z)] ∪ [operator(z[j], z[i]) for i in 1:length(z)-1 for j in i+1:length(z)])
end

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
function polynomial_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = primal_monomial_basis(z, polyorder)
    primes = vcat(primes, primal_coeff_basis(z, max_coeff))
    if operator !== nothing
        primes = vcat(primes, primal_operator_basis(z, operator))
    end
    return primes
end

function trigonometric_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    return vcat(sin.(primes), cos.(primes))
end

function exponential_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    return exp.(primes)
end

function logarithmic_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    return log.(abs.(primes))
end

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