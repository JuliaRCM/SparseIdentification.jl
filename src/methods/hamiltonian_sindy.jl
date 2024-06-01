
"""
    struct HamiltonianSINDy{T, GHT} <: SparsificationMethod

A structure representing the Hamiltonian Sparse Identification of Nonlinear Dynamics (SINDy) method.

# Fields
- `basis::Vector{Symbolics.Num}`: The augmented basis for sparsification.
- `analytical_fθ::GHT`: The analytical form of the function θ (optional).
- `z::Vector{Symbolics.Num}`: The symbolic state variable vector.
- `λ::T`: Sparsification threshold.
- `noise_level::T`: Noise amplitude added to the data.
- `t₂_data_timeStep::T`: Time step for the integrator to get noisy data at t₂ for `sparsify_picard`.
- `nloops::Int`: Number of sparsification cycles.

# Constructor
Creates a new `HamiltonianSINDy` object with specified parameters.
"""
struct HamiltonianSINDy{T, GHT} <: SparsificationMethod
    basis::Vector{Symbolics.Num}
    analytical_fθ::GHT
    z::Vector{Symbolics.Num}
    λ::T
    noise_level::T
    t₂_data_timeStep::T
    nloops::Int
    
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

"""
    sparsify(method::HamiltonianSINDy, fθ, x, ẋ, solver)

Sparsifies the coefficients of the Hamiltonian system using the SINDy algorithm.

# Arguments
- `method::HamiltonianSINDy`: The Hamiltonian SINDy method object.
- `fθ`: The function representing the Hamiltonian gradient.
- `x`: The state variable data.
- `ẋ`: The time derivatives of the state variable data.
- `solver`: The optimization solver to use.

# Returns
- `coeffs::Vector`: The sparse coefficients of the Hamiltonian system.

# Description
This function performs the following steps:
1. Adds noise to the time derivatives of the state variable data.
2. Initializes the coefficients to a vector of zeros.
3. Defines the loss function for the optimization problem.
4. Optimizes the loss function to find the initial guess of the coefficients.
5. Iteratively refines the coefficients by setting small coefficients to zero and re-optimizing the remaining coefficients.
"""
function sparsify(method::HamiltonianSINDy, fθ, x, ẋ, solver)
    ẋnoisy = [_ẋ .+ method.noise_level .* randn(size(_ẋ)) for _ẋ in ẋ]
    coeffs = zeros(get_numCoeffs(method.basis))
    
    function loss_kernel(x₀, x̃, fθ, a)
        f = zeros(eltype(a), axes(x₀))
        fθ(f, x₀, a)
        sqeuclidean(f,x̃)
    end

    function loss(a::AbstractVector)
        mapreduce(z -> loss_kernel(z..., fθ, a), +, zip(x, ẋnoisy))
    end
    
    println("Initial Guess...")
    result = Optim.optimize(loss, coeffs, solver, Optim.Options(show_trace=true); autodiff = :forward)
    coeffs .= result.minimizer
    println(result)

    for n in 1:method.nloops
        println("SINDy cycle #$n...")
        smallinds = abs.(coeffs) .< method.λ
        biginds = .~smallinds
        # check if there are any small coefficients != 0 left
        all(coeffs[smallinds] .== 0) && break
        coeffs[smallinds] .= 0

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

"""
    sparsify_picard(method::HamiltonianSINDy, fθ, x, y, solver)

Sparsifies the coefficients of the Hamiltonian system using the Picard iteration-based SINDy algorithm.

# Arguments
- `method::HamiltonianSINDy`: The Hamiltonian SINDy method object.
- `fθ`: The function representing the Hamiltonian gradient.
- `x`: The initial state variable data.
- `y`: The state variable data at the next time step.
- `solver`: The optimization solver to use.

# Returns
- `coeffs::Vector`: The sparse coefficients of the Hamiltonian system.

# Description
This function performs the following steps:
1. Initializes the coefficients to a vector of zeros.
2. Defines the loss function for the optimization problem using the Picard iteration.
3. Optimizes the loss function to find the initial guess of the coefficients.
4. Iteratively refines the coefficients by setting small coefficients to zero and re-optimizing the remaining coefficients.
"""
function sparsify_picard(method::HamiltonianSINDy, fθ, x, y, solver)
    coeffs = zeros(get_numCoeffs(method.basis))
    
    function loss_kernel(x₀, x₁, fθ, a, Δt)
        numLoops = 4 # random choice of loop steps
        local x̄ = zeros(eltype(a), axes(x₁))
        local x̃ = zeros(eltype(a), axes(x₁))
        local f = zeros(eltype(a), axes(x₁))
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

    function loss(a::AbstractVector)
        mapreduce(z -> loss_kernel(z..., fθ, a, method.t₂_data_timeStep), +, zip(x, y))
    end
    
    println("Initial Guess...")
    result = Optim.optimize(loss, coeffs, solver, Optim.Options(show_trace=true); autodiff = :forward)
    coeffs .= result.minimizer
    println(result)

    for n in 1:method.nloops
        println("Iteration #$n...")
        smallinds = abs.(coeffs) .< method.λ
        biginds = .~smallinds
        all(coeffs[smallinds] .== 0) && break
        coeffs[smallinds] .= 0

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

"""
    struct HamiltonianSINDyVectorField{DT, CT, GHT} <: VectorField

A structure representing a vector field for the Hamiltonian SINDy method.

# Fields
- `coefficients::CT`: The coefficients of the Hamiltonian system.
- `fθ::GHT`: The function representing the Hamiltonian gradient.

# Constructor
Creates a new `HamiltonianSINDyVectorField` object with specified coefficients and Hamiltonian gradient basis function.
"""
struct HamiltonianSINDyVectorField{DT,CT,GHT} <: VectorField
    coefficients::CT
    fθ::GHT

    function HamiltonianSINDyVectorField(coefficients::CT, fθ::GHT) where {DT, CT <: AbstractVector{DT}, GHT <: Base.Callable}
        new{DT,CT,GHT}(coefficients, fθ)
    end
end

"""
    VectorField(method::HamiltonianSINDy, data::TrainingData; solver = BFGS(), algorithm = "sparsify")

Creates a vector field for the Hamiltonian SINDy method using training data.

# Arguments
- `method::HamiltonianSINDy`: The Hamiltonian SINDy method object.
- `data::TrainingData`: The training data containing state variables and derivatives.
- `solver`: The optimization solver to use (default is BFGS()).
- `algorithm::String`: The algorithm to use for sparsification ("sparsify" or "sparsify_picard").

# Returns
- `HamiltonianSINDyVectorField`: The vector field representing the Hamiltonian SINDy method.

# Description
This function performs the following steps:
1. Checks if the first dimension of the state variables is even.
2. Determines the dimension `d` of the system.
3. Builds the Hamiltonian gradient function `fθ`.
4. Initializes the coefficients to zeros.
5. Uses the specified algorithm to sparsify the coefficients.

 Returns a `HamiltonianSINDyVectorField` object with the sparse coefficients and Hamiltonian gradient function.
"""
function VectorField(method::HamiltonianSINDy, data::TrainingData; solver = BFGS(), algorithm = "sparsify")
    size(data.x[begin], 1) % 2 == 0 || throw(ArgumentError("The first dimension of x must be even."))
    d = size(data.x[begin], 1) ÷ 2
    fθ = ΔH_func_builder(d, method.z, method.basis)
    coeffs = zeros(get_numCoeffs(method.basis))
    
    if algorithm == "sparsify"
        coeffs = sparsify(method, fθ, data.x, data.ẋ, solver)
    elseif algorithm == "sparsify_picard"
        coeffs = sparsify_picard(method, fθ, data.x, data.y, solver)
    else throw(ArgumentError("Algorithm must be either sparsify or sparsify_picard"))
    end
    
    HamiltonianSINDyVectorField(coeffs, fθ)
end

"""
    (vectorfield::HamiltonianSINDyVectorField)(dz, z)

Evaluates the Hamiltonian gradient using the sparse coefficients.

# Arguments
- `dz`: The output vector for the gradient.
- `z`: The input state variable vector.

# Returns
- `dz`: The evaluated gradient.

# Description
This function evaluates the Hamiltonian gradient at the given state variables using the sparse coefficients.
"""
function (vectorfield::HamiltonianSINDyVectorField)(dz, z)
    vectorfield.fθ(dz, z, vectorfield.coefficients)
    return dz
end

(vectorfield::HamiltonianSINDyVectorField)(dz, z, p, t) = vectorfield(dz, z)