#The _prod function takes one or more input arrays and performs an element-wise multiplication on them.
_prod(a, b, c, arrs...) = a .* _prod(b, c, arrs...)
_prod(a, b) = a .* b
_prod(a) = a

# generates a vector out of symbolic arrays (p,q) with a certain dimension
function get_z_vector(dims)
    @variables q[1:dims]
    @variables p[1:dims]
    z = vcat(q,p)
    return z
end


# returns the number of required coefficients for the basis
function get_numCoeffs(basis::Vector{Symbolics.Num})
    return length(basis)
end


# gets a vector of combinations of hamiltonian basis
function get_basis_set(basis::Vector{Symbolics.Num}...)
    # gets a vector of combinations of basis
    basis = vcat(basis...)
    
    # removes duplicates
    basis = Vector{Symbolics.Num}(collect(unique(basis)))

    return basis
end

function newtHam_dataGen(x, ẋ)
    x_mat = zeros(length(x[begin]), length(x))
    ẋ_mat = zeros(length(ẋ[begin]), length(ẋ))
    for i in 1:length(x)
        x_mat[:,i] .= x[i]
        x_mat[:,i] .= x[i]
        ẋ_mat[:,i] .= ẋ[i]
        ẋ_mat[:,i] .= ẋ[i]
    end
    return TrainingData(x_mat, ẋ_mat)
end