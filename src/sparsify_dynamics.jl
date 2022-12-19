
"sequential least squares"
function sparsify_dynamics(Θ, ẋ, λ, nloops=10; solver = JuliaLeastSquare())
    # initial guess: least-squares
    Ξ = solve(Θ, ẋ', solver)

    for _ in 1:nloops
        # find coefficients below λ threshold
        smallinds = abs.(Ξ) .< λ

        # check if there are any small coefficients != 0 left
        all(Ξ[smallinds] .== 0) && break

        # set all small coefficients to zero
        Ξ[smallinds] .= 0

        # Regress dynamics onto remaining terms to find sparse Ξ
        for ind in axes(ẋ,1)
            biginds = .~(smallinds[:,ind])
            Ξ[biginds,ind] .= solve(Θ[:,biginds], ẋ[ind,:], solver)
        end
    end
    
    return Ξ
end
