
struct HamiltonianSINDy{T, GHT} <: SparsificationMethod
    analytical_fθ::GHT

    λ::T
    noise_level::T
    nloops::Int

    polyorder::Int
    trigonometric::Int

    function HamiltonianSINDy(analytical_fθ::GHT;
        λ::T = DEFAULT_LAMBDA,
        noise_level::T = DEFAULT_NOISE_LEVEL,
        nloops = DEFAULT_NLOOPS,
        polyorder::Int = 3,
        trigonometric::Int = 0) where {T, GHT <: Base.Callable}

        new{T, GHT}(analytical_fθ, λ, noise_level, nloops, polyorder, trigonometric)
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
    d = size(data.x, 1) ÷ 2

    # returns function that builds hamiltonian gradient through symbolics
    # " the function hamilGradient_general!() needs this "
    fθ = hamilGrad_func_builder(d, method.polyorder, method.trigonometric)

    # Compute Sparse Regression
    coeffs = sparsify_two(method, fθ, data.x, data.ẋ, solver)

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
function sparsify_two(method::HamiltonianSINDy, fθ, x, ẋ, solver)

    # initialize timestep data for analytical solution
    EulerTimeStep = 0.01 # randomly chosen timestep size
    tspan = (0.0, EulerTimeStep)
    trange = range(tspan[begin], step = EulerTimeStep, stop = tspan[end])

    # matrix to store solution at next time point
    data_ref = zero(x)

    for j in axes(data_ref, 2)
        prob_ref = ODEProblem(method.analytical_fθ, x[:,j], tspan)
        sol = ODE.solve(prob_ref, Tsit5(), dt = EulerTimeStep, abstol = 1e-10, reltol = 1e-10, saveat = trange)
        data_ref[:,j] = sol.u[2]
    end

    #TODO: Ask Dr. Michael if it is correct to add noise here or to x directly before doing ODE.solve
    
    # add noise
    data_ref_noisy = data_ref .+ method.noise_level .* randn(size(data_ref))

    # dimension of system
    nd = size(x,1)

    # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = calculate_nparams(nd, method.polyorder, method.trigonometric)

    # coeffs initialized to a vector of zeros b/c easier to optimize zeros for our case
    coeffs = zeros(nparam)
    
    # define loss function
    function loss(a::AbstractVector, λ::Real)

        numLoops = 3 # random choice of loop steps

        # initialize matrix to store picard iterations result
        picardX = zeros(eltype(a), axes(x))

        # initialization for the SINDy coefficients result
        res = zeros(eltype(a), axes(ẋ))
        out = zeros(eltype(a), nd)
        
        for j in axes(res, 2)
            fθ(out, x[:,j], a) # gradient at current (x) values
            res[:,j] .= out
            picardX[:,j] .= x[:,j] .+ EulerTimeStep .* res[:,j] # for first guess use explicit euler
            
            for loop = 1:numLoops
                fθ(out, (x[:,j] .+ picardX[:,j]) ./ 2, a) # find gradient at {(x̃ₙ + x̃ⁱₙ₊₁)/2} to get Hermite extrapolation
                res[:,j] .= out
                picardX[:,j] .= x[:,j] + EulerTimeStep * res[:,j] # mid point rule for integration to next step
            end
        end

        return mapreduce(y -> y^2, +, data_ref_noisy .- picardX) + λ * sum(abs2.(a))
    end
    
    # initial guess
    println("Initial Guess...")

    # example l2 regularization
    λ = 0 #TODO: change this because there is already method.λ
    result = Optim.optimize(a -> loss(a, λ), coeffs, solver, Optim.Options(show_trace=true); autodiff = :forward)
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
            loss(c, λ)
        end

        b = coeffs[biginds]
        result = Optim.optimize(sparseloss, b, solver, Optim.Options(show_trace=true); autodiff = :forward)
        b .= result.minimizer

        println(result)
    end
    
    return coeffs
end