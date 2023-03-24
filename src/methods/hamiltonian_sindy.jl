struct HamiltonianSINDy{T, GHT} <: SparsificationMethod
    analytical_fθ::GHT

    λ::T
    noise_level::T
    integrator_timeStep::T
    nloops::Int

    polyorder::Int
    trigonometric::Int

    function HamiltonianSINDy(analytical_fθ::GHT;
        λ::T = DEFAULT_LAMBDA,
        noise_level::T = DEFAULT_NOISE_LEVEL,
        integrator_timeStep::T = DEFAULT_INTEGRATOR_TIMESTEP,
        nloops = DEFAULT_NLOOPS,
        polyorder::Int = 3,
        trigonometric::Int = 0) where {T, GHT <: Base.Callable}

        new{T, GHT}(analytical_fθ, λ, noise_level, integrator_timeStep, nloops, polyorder, trigonometric)
    end
end

function sparsify(method::HamiltonianSINDy, fθ, x, ẋ, solver)
    # add noise
    ẋnoisy = ẋ .+ method.noise_level .* randn(size(ẋ))

    # dimension of system
    nd = size(x,1)

    # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = calculate_nparams(nd, method.polyorder, method.trigonometric)

    # coeffs initialized to a vector of zeros b/c easier to optimze zeros for our case
    coeffs = zeros(nparam)
    
    # define loss function
    function loss(a::AbstractVector)
        res = zeros(eltype(a), axes(ẋnoisy))
        out = zeros(eltype(a), nd)
        
        for j in axes(res, 2)
            fθ(out, x[:,j], a)
            res[:,j] .= out
        end

        mapreduce(y -> y^2, +, ẋnoisy .- res)
    end
    
    # initial guess
    println("Initial Guess...")
    result = Optim.optimize(loss, coeffs, solver; autodiff = :forward)
    coeffs .= result.minimizer

    println(result)

    for n in 1:method.nloops
        println("Iteration #$n...")

        # find coefficients below λ threshold
        smallinds = abs.(coeffs) .< method.λ
        biginds = .~smallinds

        # check if there are any small coefficients != 0 left
        all(coeffs[smallinds] .== 0) && break

        # set all small coefficients to zero
        coeffs[smallinds] .= 0

        # Regress dynamics onto remaining terms to find sparse coeffs
        function sparseloss(b::AbstractVector)
            c = zeros(eltype(b), axes(coeffs))
            c[biginds] .= b
            loss(c)
        end

        b = coeffs[biginds]
        result = Optim.optimize(sparseloss, b, solver; autodiff = :forward)
        b .= result.minimizer

        println(result)
    end
    
    return coeffs
end




struct HamiltonianSINDyVectorField{DT,CT,GHT} <: VectorField
    # basis::BT
    coefficients::CT
    fθ::GHT

    function HamiltonianSINDyVectorField(coefficients::CT, fθ::GHT) where {DT, CT <: AbstractVector{DT}, GHT <: Base.Callable}
        new{DT,CT,GHT}(coefficients, fθ)
    end
end




function VectorField(method::HamiltonianSINDy, data::TrainingData; solver = Newton())
    # TODO: Check that first dimension x is even

    # dimension of system
    d = size(data.x[begin], 1) ÷ 2

    # returns function that builds hamiltonian gradient through symbolics
    # " the function hamilGradient_general!() needs this "
    fθ = hamilGrad_func_builder(d, method.polyorder, method.trigonometric)

    # Compute Sparse Regression
    coeffs = sparsify_parallel(method, fθ, data.x, data.y, solver)

    HamiltonianSINDyVectorField(coeffs, fθ)
end


" wrapper function for generalized SINDY hamiltonian gradient.
Needs the output of fθ_sparse to work!
It is in a syntax that is suitable to be evaluated by a loss function
for optimization "
function (vectorfield::HamiltonianSINDyVectorField)(dz, z)
    vectorfield.fθ(dz, z, vectorfield.coefficients)
    return dz
end

(vectorfield::HamiltonianSINDyVectorField)(dz, z, p, t) = vectorfield(dz, z)















################################################################################################
################################################################################################
################################################################################################
################################################################################################

function gen_noisy_ref_data(method::HamiltonianSINDy, x)
    # initialize timestep data for analytical solution
    tstep = method.integrator_timeStep
    tspan = (zero(tstep), tstep)

    # # matrix to store solution at next time point
    # data_ref = zero(x)

    function next_timestep(x)
        prob_ref = ODEProblem((dx, t, x, params) -> dx .= method.analytical_fθ(x, params, t), tspan, tstep, x)
        sol = integrate(prob_ref, Gauss(2))
        sol.q[end]
    end

    data_ref = [next_timestep(_x) for _x in x]

    # add noise
    data_ref_noisy = [_x .+ method.noise_level .* randn(size(_x)) for _x in data_ref]

    return data_ref_noisy

    #TODO: Ask Dr. Michael if it is correct to add noise here or to x directly before doing ODE.solve
end


function sparsify_two(method::HamiltonianSINDy, fθ, x, ẋ, solver)

    # generate noisy references data
    data_ref_noisy = gen_noisy_ref_data(method::HamiltonianSINDy, x)

    # dimension of system
    nd = size(x,1)

    # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = calculate_nparams(nd, method.polyorder, method.trigonometric)

    # coeffs initialized to a vector of zeros b/c easier to optimize zeros for our case
    coeffs = zeros(nparam)
    
    # define loss function
    function loss(a::AbstractVector)

        numLoops = 4 # random choice of loop steps

        # initialize matrix to store picard iterations result
        picardX = zeros(eltype(a), axes(x))

        # initialization for the SINDy coefficients result
        res = zeros(eltype(a), axes(ẋ))
        out = zeros(eltype(a), nd)
        
        for j in axes(res, 2)
            fθ(out, x[:,j], a) # gradient at current (x) values
            res[:,j] .= out
            picardX[:,j] .= x[:,j] .+ method.integrator_timeStep .* res[:,j] # for first guess use explicit euler
            
            for loop = 1:numLoops
                fθ(out, (x[:,j] .+ picardX[:,j]) ./ 2, a) # find gradient at {(x̃ₙ + x̃ⁱₙ₊₁)/2} to get Hermite extrapolation
                res[:,j] .= out
                picardX[:,j] .= x[:,j] .+ method.integrator_timeStep .* res[:,j] # mid point rule for integration to next step
            end
        end

        return mapreduce(y -> y^2, +, data_ref_noisy .- picardX)
    end
    
    # initial guess
    println("Initial Guess...")
    result = Optim.optimize(loss, coeffs, solver, Optim.Options(show_trace=true); autodiff = :forward)
    
    coeffs .= result.minimizer

    println(result)

    for n in 1:method.nloops
        println("Iteration #$n...")

        # find coefficients below λ threshold
        smallinds = abs.(coeffs) .< method.λ
        biginds = .~smallinds

        # check if there are any small coefficients != 0 left
        all(coeffs[smallinds] .== 0) && break

        # set all small coefficients to zero
        coeffs[smallinds] .= 0

        # Regress dynamics onto remaining terms to find sparse coeffs
        function sparseloss(b::AbstractVector)
            c = zeros(eltype(b), axes(coeffs))
            c[biginds] .= b
            loss(c)
        end

        b = coeffs[biginds]
        result = Optim.optimize(sparseloss, b, solver, Optim.Options(show_trace=true); autodiff = :forward)
        b .= result.minimizer

        println(result)
    end
    
    return coeffs
end














################################################################################################
################################################################################################
################################################################################################
################################################################################################
function sparsify_parallel(method::HamiltonianSINDy, fθ, x, y, solver)

    # dimension of system
    nd = size(x[begin],1)

    # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = calculate_nparams(nd, method.polyorder, method.trigonometric)

    # coeffs initialized to a vector of zeros b/c easier to optimize zeros for our case
    coeffs = zeros(nparam)


    function loss_kernel(x₀, x₁, fθ, a, Δt)
        numLoops = 4 # random choice of loop steps

        # solution of SINDy Hamiltonian problem
        local x̄ = zeros(eltype(a), axes(x₁))
        local x̃ = zeros(eltype(a), axes(x₁))
        local f = zeros(eltype(a), axes(x₁))

        # gradient at current (x) values
        fθ(f, x₀, a)

        # for first guess use explicit euler
        x̃ .= x₀ .+ Δt .* f
        
        for _ in 1:numLoops
            x̄ .= (x₀ .+ x̃) ./ 2
            # find gradient at {(x̃ₙ + x̃ⁱₙ₊₁)/2} to get Hermite extrapolation
            fθ(f, x̄, a)
            # mid point rule for integration to next step
            x̃ .= x₀ .+ Δt .* f
        end

        sqeuclidean(x₁,x̃)
    end

    
    # define loss function
    function loss(a::AbstractVector)
        mapreduce(z -> loss_kernel(z..., fθ, a, method.integrator_timeStep), +, zip(x, y))
    end

    # function ploss(a::AbstractVector)
    #     pmapreduce(z -> loss_kernel(z..., fθ, a, method.integrator_timeStep), +, zip(x̄, ȳ))
    # end

    # loss(coeffs)
    # @time loss(coeffs)
    
    # ploss(coeffs)
    # @time ploss(coeffs)
    
    # initial guess
    println("Initial Guess...")
    result = Optim.optimize(loss, coeffs, solver, Optim.Options(show_trace=true); autodiff = :forward)
    
    coeffs .= result.minimizer

    println(result)

    for n in 1:method.nloops
        println("Iteration #$n...")

        # find coefficients below λ threshold
        smallinds = abs.(coeffs) .< method.λ
        biginds = .~smallinds

        # check if there are any small coefficients != 0 left
        all(coeffs[smallinds] .== 0) && break

        # set all small coefficients to zero
        coeffs[smallinds] .= 0

        # Regress dynamics onto remaining terms to find sparse coeffs
        function sparseloss(b::AbstractVector)
            c = zeros(eltype(b), axes(coeffs))
            c[biginds] .= b
            loss(c)
        end

        b = coeffs[biginds]
        result = Optim.optimize(sparseloss, b, solver, Optim.Options(show_trace=true); autodiff = :forward)
        b .= result.minimizer

        println(result)
    end
    
    return coeffs
end
