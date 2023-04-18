# builds a function that is able to construct an augmented library of candidate nonlinear functions

function hamiltonian_poly_combos(z, order, inds...)
    ham = []

    if order == 0
        Num(1)
    elseif order == length(inds)
        ham = vcat(ham, _prod([z[i] for i in inds]...))
    else
        start_ind = length(inds) == 0 ? 1 : inds[end]
        for j in start_ind:length(z)
            ham = vcat(ham, hamiltonian_poly_combos(z, order, inds..., j))
        end
    end

    return ham
end

function hamiltonian_basis_concat(z, order)
    ham = []

    for i in 1:order
        ham = vcat(ham, hamiltonian_poly_combos(z, i))
    end

    return hcat(ham)
end

function hamil_basis_maker(data, order)

    #number of variables
    n = size(data, 1)

    #number of combinations
    num_combos = 1 + n + n*(n+1) ÷ 2 + n*(n+1)*(n+2) ÷ 6 - 1

    result = Array{Float64}(undef, 0, num_combos)

    for i in 1:size(data, 2) 
        temp = hamiltonian_basis_concat(data[:, i], order)
        temp = temp'
        result = vcat(result, temp)
    end
    
    return result
end
