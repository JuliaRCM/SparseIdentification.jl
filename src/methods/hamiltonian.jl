
"""
    ΔH_func_builder(d::Int, z::Vector{Symbolics.Num} = get_z_vector(d), basis::Vector{Symbolics.Num}...)

Returns a function that builds the gradient of the Hamiltonian.

# Arguments
- `d::Int`: The dimension of the state variables (e.g., if q and p each have 3 dimensions, then d = 3).
- `z::Vector{Symbolics.Num}`: The state variable vector (default is generated using `get_z_vector(d)`).
- `basis::Vector{Symbolics.Num}`: The basis functions for the Hamiltonian.

# Returns
- `ΔH_eval::Function`: A function that calculates the Hamiltonian gradient, which is converted to a native Julia function.

# Description
This function performs the following steps:
1. Computes the total number of dimensions `nd` of all state variables.
2. Computes the differentials `Dz` of the state variables `z`.
3. Collects and sums combinations of the basis functions and coefficients to form the Hamiltonian `ham`.
4. Computes the derivative of the Hamiltonian `f` with respect to the state variables.
5. Simplifies the expressions for efficiency.
6. Converts the gradient vector into a Hamiltonian vector field by multiplying with a skew-symmetric matrix.
7. Builds a runtime-generated function `ΔH_eval` to evaluate the Hamiltonian gradient.

The returned function can be used to evaluate the gradient of the Hamiltonian system at given state variables and coefficients.
The returned function `ΔH_eval` takes arguments `dz` (returned values holder), `z`(state values), and `a` (coefficients) and returns the Hamiltonian gradient, stored in `dz`.
"""
function ΔH_func_builder(d::Int, z::Vector{Symbolics.Num} = get_z_vector(d), basis::Vector{Symbolics.Num}...) 
    # nd is the total number of dimensions of all the states, e.g. if q,p each of 3 dims, that is 6 dims in total
    nd = 2d
    Dz = Differential.(z)
    
    # collects and sums combinations of basis and coefficients
    basis = get_basis_set(basis...)
   
    # gets number of terms in the basis
    @variables a[1:get_numCoeffs(basis)]
    
    # collect and sum combinations of basis and coefficients
    ham = sum(collect(a .* basis))
    
    # gives derivative of the Hamiltonian, but not the skew-symmetric true one
    f = [expand_derivatives(dz(ham)) for dz in Dz]

    # simplify the expression potentially to make it faster
    f = simplify(f)
    
    # line below makes the vector into a Hamiltonian vector field by multiplying with the skew-symmetric matrix
    ΔH = vcat(f[d+1:2d], -f[1:d])
    
    # builds a function that calculates Hamiltonian gradient and converts the function to a native Julia function
    ΔH_eval = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(build_function(ΔH, z, a)[2]))
    
    return ΔH_eval
end
