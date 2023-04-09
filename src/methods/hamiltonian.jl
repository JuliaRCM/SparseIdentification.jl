


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


" collects and sums polynomial, trignometric, and states differences combinations of basis "
function hamiltonian(z, a, order, trig_wave_num, diffs_power)
    ham = []

    # Polynomial basis
    for i in 1:order
        ham = vcat(ham, hamiltonian_poly(z, i))
    end

    # Trignometric basis
    for k = 1:trig_wave_num
        ham = vcat(ham, vcat(sin.(k*z)), vcat(cos.(k*z)))
    end

    if diffs_power > 0
        # States difference basis
        diffs = Vector{Num}()
        idx = 1
        for i in 1:length(z)
            for j in 1:length(z)
                if i == j
                    continue  # skip index where difference is between same state
                end
                push!(diffs, (z[i] - z[j]))
                idx += 1
            end
        end

        for k = 1:diffs_power
            ham = vcat(ham, vcat(diffs .^ k))
        end
    end

    ham = sum(collect(a .* ham))
    return ham

end



" returns a function that can build the gradient of the hamiltonian "
function ΔH_func_builder(d, polyorder, trig_wave_num, diffs_power)
    # nd is the total number of dimensions of all the states, e.g. if q,p each of 3 dims, that is 6 dims in total
    nd = 2d
    
    # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = calculate_nparams(nd, polyorder, trig_wave_num, diffs_power)

    # symbolic variables
    @variables a[1:nparam]
    @variables q[1:d]
    @variables p[1:d]
    z = vcat(q,p)
    Dz = Differential.(z)
    
    # make a basis library
    ham = hamiltonian(z, a, polyorder, trig_wave_num, diffs_power)
    
    # gives derivative of the hamiltonian, but not the skew-symmetric true one
    f = [expand_derivatives(dz(ham)) for dz in Dz]

    # line below makes the vector into a hamiltonian vector field by multiplying with the skew-symmetric matrix
    ∇H = vcat(f[d+1:2d], -f[1:d])

    # builds a function that calculates Hamiltonian gradient and converts the function to a native Julia function
    ∇H_eval = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(build_function(∇H, z, a)[2]))

    return ∇H_eval
end
