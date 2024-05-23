"""
The `_prod` function takes one or more input arrays and performs an element-wise multiplication on them.
"""
_prod(a, b, c, arrs...) = a .* _prod(b, c, arrs...)
_prod(a, b) = a .* b
_prod(a) = a

"""
Generates a vector out of symbolic arrays `p` and `q` with a certain dimension.

# Arguments
- `dims`: Dimension for the symbolic arrays.

# Returns
- A vector `z` which is a concatenation of vectors `q` and `p`.
"""
function get_z_vector(dims)
    @variables q[1:dims]
    @variables p[1:dims]
    z = vcat(q,p)
    return z
end

"""
Returns the number of required coefficients for the basis.

# Arguments
- `basis::Vector{Symbolics.Num}`: A vector of symbolic numbers.

# Returns
- The number of elements in the basis vector.
"""
function get_numCoeffs(basis::Vector{Symbolics.Num})
    return length(basis)
end

"""
Gets a vector of unique combinations of Hamiltonian basis.

# Arguments
- `basis::Vector{Symbolics.Num}...`: One or more vectors of symbolic numbers.

# Returns
- A vector containing unique combinations of the basis vectors.
"""
function get_basis_set(basis::Vector{Symbolics.Num}...)
    basis = vcat(basis...)
    basis = Vector{Symbolics.Num}(collect(unique(basis)))
    return basis
end

"""
Generates Hamiltonian training data to be operated on by classical SINDy methods.

# Arguments
- `x`: A vector of state data.
- `ẋ`: A vector of time derivative data.

# Returns
- A `TrainingData` object containing the input data and its derivatives.
"""
function newtHam_dataGen(x, ẋ)
    x_mat = zeros(length(x[begin]), length(x))
    ẋ_mat = zeros(length(ẋ[begin]), length(ẋ))
    for i in 1:length(x)
        x_mat[:,i] .= x[i]
        ẋ_mat[:,i] .= ẋ[i]
    end
    return TrainingData(x_mat, ẋ_mat)
end

"""
Generates noisy state data at a specific time step.

# Arguments
- `method::HamiltonianSINDy`: A Hamiltonian SINDy method object containing relevant parameters.
- `x`: A vector of input state data.

# Returns
- Noisy state data at a specified time step.
"""
function gen_noisy_t₂_data(method::HamiltonianSINDy, x)
    tstep = method.t₂_data_timeStep
    tspan = (zero(tstep), tstep)

    function next_timestep(x)
        prob_ref = ODEProblem((dx, t, x, params) -> method.analytical_fθ(dx, x, params, t), tspan, tstep, x)
        sol = integrate(prob_ref, Gauss(4))
        sol.q[end]
    end

    data_ref = [next_timestep(_x) for _x in x]
    data_ref_noisy = [_x .+ method.noise_level .* randn(size(_x)) for _x in data_ref]

    return data_ref_noisy
end