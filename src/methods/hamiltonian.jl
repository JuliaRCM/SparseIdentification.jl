
##########################################################
# Functions for Hamiltonian gradient calculations
##########################################################

# returns a function that can build the gradient of the hamiltonian
function ΔH_func_builder(d::Int, z::Vector{Symbolics.Num} = get_z_vector(d), basis::Vector{Symbolics.Num}...) 
    # nd is the total number of dimensions of all the states, e.g. if q,p each of 3 dims, that is 6 dims in total
    nd = 2d
    Dz = Differential.(z)
    
    # collects and sums combinations of basis and coefficients"
    basis = get_basis_set(basis...)
   
    # gets number of terms in the basis
    @variables a[1:get_numCoeffs(basis)]
    
    # collect and sum combinations of basis and coefficients
    ham = sum(collect(a .* basis))
    
    # gives derivative of the hamiltonian, but not the skew-symmetric true one
    f = [expand_derivatives(dz(ham)) for dz in Dz]

    #simplify the expression potentially to make it faster
    f = simplify(f)
    
    # line below makes the vector into a hamiltonian vector field by multiplying with the skew-symmetric matrix
    ∇H = vcat(f[d+1:2d], -f[1:d])
    
    # builds a function that calculates Hamiltonian gradient and converts the function to a native Julia function
    ∇H_eval = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(build_function(∇H, z, a)[2]))
    
    return ∇H_eval
end