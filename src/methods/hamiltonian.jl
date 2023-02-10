


##########################################################
# Helper functions for Hamiltonian gradient calculations
##########################################################


" makes polynomial combinations of basis "
function hamiltonian_poly(z, order, inds...)
    ham = []

    if order == 0
        Num(1)
    elseif order == length(inds)
        ham = vcat(ham, _prod([z[i] for i in inds]...))
    else
        start_ind = length(inds) == 0 ? 1 : inds[end]
        for j in start_ind:length(z)
            ham = vcat(ham, hamiltonian_poly(z, order, inds..., j))
        end
    end

    return ham
end


" collects and sums only polynomial combinations of basis "
function hamiltonian(z, a, order)
    ham = []

    for i in 1:order
        ham = vcat(ham, hamiltonian_poly(z, i))
    end

    sum(collect(a .* ham))
end


" collects and sums polynomial and trignometric combinations of basis "
function hamil_trig(z, a, order, trig_wave_num)
    ham = []

    # Polynomial basis
    for i in 1:order
        ham = vcat(ham, hamiltonian_poly(z, i))
    end

    # Trignometric basis
    for k = 1:trig_wave_num
        ham = vcat(ham, vcat(sin.(k*z)), vcat(cos.(k*z)))
    end

    ham = sum(collect(a .* ham))

    return ham

end



" returns a function that can build the gradient of the hamiltonian "
function hamilGrad_func_builder(d, polyorder, trig_wave_num)
    # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = calculate_nparams(d, polyorder, trig_wave_num)

    # symbolic variables
    @variables a[1:nparam]
    @variables q[1:d]
    @variables p[1:d]
    z = vcat(q,p)

    # usesine: whether to add trig basis or not
    if trig_wave_num > 0

        # gives derivative of the hamiltonian, but not the skew-symmetric true one
        Dz = Differential.(z)
        ∇H_add_trig = [expand_derivatives(dz(hamil_trig(z, a, polyorder, trig_wave_num))) for dz in Dz]

        # line below makes the vector into a hamiltonian vector field by multiplying with the skew-symmetric matrix
        ∇H_trig = vcat(∇H_add_trig[d+1:2d], -∇H_add_trig[1:d])

        # builds a function that calculates Hamiltonian gradient and converts the function to a native Julia function
        ∇H_eval = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(build_function(∇H_trig, z, a)[2]))

        return ∇H_eval

    else

        # gives derivative of the hamiltonian, but not the skew-symmetric true one
        Dz = Differential.(z)
        f = [expand_derivatives(dz(hamiltonian(z, a, polyorder))) for dz in Dz]

        # line below makes the vector into a hamiltonian by multiplying with the skew-symmetric matrix
        ∇H = vcat(f[d+1:2d], -f[1:d])

        # builds a function that calculates Hamiltonian gradient and converts the function to a native Julia function
        ∇H_eval = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(build_function(∇H, z, a)[2]))
        
        return ∇H_eval

    end

end
