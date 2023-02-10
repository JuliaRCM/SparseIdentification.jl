
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





function sparsify_hamiltonian_dynamics(a, loss, λ, nloops=10)
    # initial guess
    println("Initial Guess...")
    td = TwiceDifferentiable(loss, a; autodiff = :forward)
    result = Optim.optimize(td, a, Newton())
    a .= result.minimizer

    println(result)

    for n in 1:nloops
        println("Iteration #$n...")

        # find coefficients below λ threshold
        smallinds = abs.(a) .< λ
        biginds = .~smallinds

        # check if there are any small coefficients != 0 left
        all(a[smallinds] .== 0) && break

        # set all small coefficients to zero
        a[smallinds] .= 0

        # Regress dynamics onto remaining terms to find sparse a

        function sparseloss(b::AbstractVector{T}) where {T}
            x = zeros(T, axes(a))
            x[biginds] .= b
            loss(x)
        end

        b = a[biginds]
        td = TwiceDifferentiable(sparseloss, b; autodiff = :forward)
        result = Optim.optimize(td, b, Newton())
        a[biginds] .= result.minimizer

        println(result)
    end
    
    return a
end
