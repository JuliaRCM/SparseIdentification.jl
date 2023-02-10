" functions to generate hamiltonian function of variable z and 
order 3 of combinations with 2 dims for each variable "

_prod(a, b, c, arrs...) = a .* _prod(b, c, arrs...)
_prod(a, b) = a .* b
_prod(a) = a




##########################################################
" Helper functions for Hamiltonian gradient calculations"
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


"returns the number of required parameters
depending on whether there are trig basis or not"

function calculate_nparams(d, polyorder, usesine, trig_wave_num)
    
    # binomial used to get the combination of polynomials till the highest order without repeat, e.g nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = binomial(2*d + polyorder, polyorder) - 1

    if usesine == false

        return nparam

    elseif usesine == true

        # first 2 in the product formula b/c the trig basis are sin and cos i.e. two basis functions
        # 2d: b/c the phase space is two variables p,q each with 2 dims
        trig_basis_length = 2 * trig_wave_num * 2d

        return (nparam + trig_basis_length)

    end
end