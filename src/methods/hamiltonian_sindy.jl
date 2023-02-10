
struct HamiltonianSINDy{T} <: SparsificationMethod
    λ::T
    ϵ::T
    nloops::Int

    polyorder::Int
    trigonometric::Int

    function HamiltonianSINDy(;
        lambda::T = DEFAULT_LAMBDA,
        noise_level::T = DEFAULT_NOISE_LEVEL,
        nloops = DEFAULT_NLOOPS,
        polyorder::Int = 3,
        trigonometric::Int = 0) where {T}

        new{T}(lambda, noise_level, nloops, polyorder, trigonometric)
    end
end


function sparsify(method::HamiltonianSINDy, ∇H, x, ẋ, solver)
    # add noise
    ẋnoisy = ẋ .+ method.ϵ .* randn(size(ẋ))

    # dimension of system
    nd = size(x,1)

    # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = calculate_nparams(nd, method.polyorder, method.trigonometric)

    # (a) initialized to a vector of zeros b/c easier to optimze zeros for our case
    coeffs = zeros(nparam)
    
    # define loss function
    function loss(a::AbstractVector)
        res = zeros(eltype(a), axes(ẋnoisy))
        out = zeros(eltype(a), nd)
        
        for j in axes(res, 2)
            ∇H(out, x[:,j], a)
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

        # Regress dynamics onto remaining terms to find sparse a

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
    ∇H::GHT

    function HamiltonianSINDyVectorField(coefficients::CT, ∇H::GHT) where {DT, CT <: AbstractVector{DT}, GHT <: Base.Callable}
        new{DT,CT,GHT}(coefficients, ∇H)
    end
end




function VectorField(method::HamiltonianSINDy, data::TrainingData; solver = Newton())
    # TODO: Check that first dimension x is even

    # dimension of system
    d = ndims(data.x) ÷ 2

    # returns function that builds hamiltonian gradient through symbolics
    # " the function hamilGradient_general!() needs this "
    ∇H = hamilGrad_func_builder(d, method.polyorder, method.trigonometric)

    # Compute Sparse Regression
    coeffs = sparsify(method, ∇H, data.x, data.ẋ, solver)

    HamiltonianSINDyVectorField(coeffs, ∇H)
end


" wrapper function for generalized SINDY hamiltonian gradient.
Needs the output of ∇H_sparse to work!
It is in a syntax that is suitable to be evaluated by a loss function
for optimization "
function (vectorfield::HamiltonianSINDyVectorField)(dz, z)
    vectorfield.∇H(dz, z, vectorfield.coefficients)
    return dz
end

(vectorfield::HamiltonianSINDyVectorField)(dz, z, p, t) = vectorfield(dz, z)
