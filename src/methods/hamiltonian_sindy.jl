struct HamiltonianSINDy{T, GHT} <: SparsificationMethod
    basis::Vector{Symbolics.Num} # the augmented basis for sparsification
    analytical_fθ::GHT
    z::Vector{Symbolics.Num} 
    λ::T # Sparsification Parameter
    noise_level::T # Noise amplitude added to the data
    t₂_data_timeStep::T # Time step for the integrator to get noisy data at t₂ for sparsify_picard
    nloops::Int # Sparsification Loops
    
    function HamiltonianSINDy(basis::Vector{Symbolics.Num},
        analytical_fθ::GHT = missing,
        z::Vector{Symbolics.Num} = get_z_vector(2);
        λ::T = DEFAULT_LAMBDA,
        noise_level::Real = DEFAULT_NOISE_LEVEL,
        t₂_data_timeStep::T = DEFAULT_t₂_DATA_TIMESTEP,
        nloops = DEFAULT_NLOOPS) where {T, GHT <: Union{Base.Callable,Missing}}

        new{T, GHT}(basis, analytical_fθ, z, λ, noise_level, t₂_data_timeStep, nloops)
    end
end

function sparsify(method::HamiltonianSINDy, fθ, x, ẋ, solver)
    # add noise
    ẋnoisy = [_ẋ .+ method.noise_level .* randn(size(_ẋ)) for _ẋ in ẋ]

    # coeffs initialized to a vector of zeros b/c easier to optimize zeros for our case
    coeffs = zeros(get_numCoeffs(method.basis))
    
    # define loss function
    function loss_kernel(x₀, x̃, fθ, a)
        # gradient of SINDy Hamiltonian problem
        f = zeros(eltype(a), axes(x₀))
        
        # gradient at current (x) values
        fθ(f, x₀, a)

        # calculate square euclidean distance
        sqeuclidean(f,x̃)
    
    end

    # define loss function
    function loss(a::AbstractVector)
        mapreduce(z -> loss_kernel(z..., fθ, a), +, zip(x, ẋnoisy))
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

        # b is a reference to coeffs[biginds]
        b = coeffs[biginds]
        result = Optim.optimize(sparseloss, b, solver, Optim.Options(show_trace=true); autodiff = :forward)
        b .= result.minimizer

        println(result)
    end

    return coeffs
end


function sparsify_picard(method::HamiltonianSINDy, fθ, x, y, solver)
    # coeffs initialized to a vector of zeros b/c easier to optimize zeros for our case
    coeffs = zeros(get_numCoeffs(method.basis))
    
    # define loss function
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

        # calculate square euclidean distance
        sqeuclidean(x₁,x̃)
    end

    # define loss function
    function loss(a::AbstractVector)
        mapreduce(z -> loss_kernel(z..., fθ, a, method.t₂_data_timeStep), +, zip(x, y))
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


struct HamiltonianSINDyVectorField{DT,CT,GHT} <: VectorField
    # basis::BT
    coefficients::CT
    fθ::GHT

    function HamiltonianSINDyVectorField(coefficients::CT, fθ::GHT) where {DT, CT <: AbstractVector{DT}, GHT <: Base.Callable}
        new{DT,CT,GHT}(coefficients, fθ)
    end
end

function VectorField(method::HamiltonianSINDy, data::TrainingData; solver = BFGS(), algorithm = "sparsify")
    # Check if the first dimension of x is even
    size(data.x[begin], 1) % 2 == 0 || throw(ArgumentError("The first dimension of x must be even."))

    # dimension of system
    d = size(data.x[begin], 1) ÷ 2

    # returns function that builds hamiltonian gradient through symbolics
    fθ = ΔH_func_builder(d, method.z, method.basis)

    # initialize coeffs
    coeffs = zeros(get_numCoeffs(method.basis))
    
    if algorithm == "sparsify"
        coeffs = sparsify(method, fθ, data.x, data.ẋ, solver)
    elseif algorithm == "sparsify_picard"
        coeffs = sparsify_picard(method, fθ, data.x, data.y, solver)
    else throw(ArgumentError("Algorithm must be either sparsify or sparsify_picard"))
    end
    
    HamiltonianSINDyVectorField(coeffs, fθ)
end

" wrapper function for generalized SINDY hamiltonian gradient.
Needs the output of fθ to work! "
function (vectorfield::HamiltonianSINDyVectorField)(dz, z)
    vectorfield.fθ(dz, z, vectorfield.coefficients)
    return dz
end

(vectorfield::HamiltonianSINDyVectorField)(dz, z, p, t) = vectorfield(dz, z)