
struct HamiltonianSINDy{T, GHT} <: SparsificationMethod
    analytical_∇H::GHT

    λ::T
    noise_level::T
    nloops::Int

    polyorder::Int
    trigonometric::Int

    function HamiltonianSINDy(analytical_∇H::GHT;
        λ::T = DEFAULT_LAMBDA,
        noise_level::T = DEFAULT_NOISE_LEVEL,
        nloops = DEFAULT_NLOOPS,
        polyorder::Int = 3,
        trigonometric::Int = 0) where {T, GHT <: Base.Callable}

        new{T, GHT}(analytical_∇H, λ, noise_level, nloops, polyorder, trigonometric)
    end
end


function sparsify(method::HamiltonianSINDy, ∇H, x, ẋ, solver)
    # add noise
    ẋnoisy = ẋ .+ method.noise_level .* randn(size(ẋ))

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
    d = size(data.x, 1) ÷ 2

    # returns function that builds hamiltonian gradient through symbolics
    # " the function hamilGradient_general!() needs this "
    ∇H = hamilGrad_func_builder(d, method.polyorder, method.trigonometric)

    # Compute Sparse Regression
    coeffs = sparsify_two(method, ∇H, data.x, data.ẋ, solver)

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












################################################################################################
################################################################################################
################################################################################################
################################################################################################
function sparsify_two(method::HamiltonianSINDy, ∇H, x, ẋ, solver)

    # TODO: add noise to data_ref at mapreduce stage and not here
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

        # initialization for euler solution
        hermiteX = zeros(eltype(a), axes(x))
        EulerTimeStep = 0.01
        numLoops = 4 # random choice of loop steps

        # initialization for anaylitical solution
        tspan = (0.0, EulerTimeStep)
        trange = range(tspan[begin], step = EulerTimeStep, stop = tspan[end])
        
        data_ref = zero(x)

        # initialization for the SINDy coefficients result
        res = zeros(eltype(a), axes(ẋnoisy))
        out = zeros(eltype(a), nd)
        
        for j in axes(res, 2)
            ∇H(out, x[:,j], a) # gradient at current (x) values
            res[:,j] .= out
            hermiteX[:,j] .= x[:,j] .+ EulerTimeStep .* res[:,j] # for first guess use explicit euler
            
            for loop = 1:numLoops
                ∇H(out, (x[:,j] .+ hermiteX[:,j]) ./ 2, a) # find gradient at {(x̃ₙ + x̃ⁱₙ₊₁)/2} to get Hermite extrapolation
                res[:,j] .= out
                hermiteX[:,j] .= x[:,j] + EulerTimeStep * res[:,j] # mid point rule for integration to next step
            end

            prob_ref = ODEProblem(method.analytical_∇H, x[:,j], tspan)
            sol = ODE.solve(prob_ref, Tsit5(), dt = EulerTimeStep, abstol = 1e-10, reltol = 1e-10, saveat = trange)

            # sol.u[1] is one of the uniform sampled initial conditions we give, sol.u[2] is the predicted set of values at the EulerTimeStep
            data_ref[:,j] = sol.u[2]
        end

        mapreduce(y -> y^2, +, data_ref .- hermiteX)
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

        # Regress dynamics onto remaining terms to find sparse a

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

# TODO: 
# 1. find SINDY graident function on reduced dims, already decide on size of reduced dims
    # 1.1 find also nparam and coeffs on reduced dims
# 2. use autoencoder to reduce dims i.e less numbers of q,p
# 3. calculate x using SINDY method
# 4. reverse autoencoder to actual variables
# 5. find loss function
function sparsify_three(method::HamiltonianSINDy, ∇H, x, ẋ, solver)

# TODO: remove ẋ, it is unused

   # dimension of system
   nd = size(x,1)

   # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
   nparam = calculate_nparams(nd, method.polyorder, method.trigonometric)

   # coeffs initialized to a vector of zeros b/c easier to optimze zeros for our case
   coeffs = zeros(nparam)
   
   # define loss function
   function loss(a::AbstractVector)

       #TODO: add call to autoencoder:encoder part
       encoder = Dense(4, 4, relu)
       
       # initialization for euler solution
       hermiteX = zeros(eltype(a), axes(x))
       EulerTimeStep = 0.01 # random choice for last step
       numLoops = 4 # random choice of loop steps

       # initialization for anaylitical solution
       tspan = (0.0, EulerTimeStep)
       trange = range(tspan[begin], step = EulerTimeStep, stop = tspan[end])
       
       data_ref = zero(x)

       # initialization for the SINDy coefficients result
       res = zeros(eltype(a), axes(ẋnoisy))
       out = zeros(eltype(a), nd)
       
       for j in axes(res, 2)
           ∇H(out, x[:,j], a) # gradient at current (x) values
           res[:,j] .= out
           hermiteX[:,j] .= x[:,j] .+ EulerTimeStep .* res[:,j] # for first guess use explicit euler
           
           for loop = 1:numLoops
               ∇H(out, (x[:,j] .+ hermiteX[:,j]) ./ 2, a) # find gradient at {(x̃ₙ + x̃ⁱₙ₊₁)/2} to get Hermite extrapolation
               res[:,j] .= out
               hermiteX[:,j] .= x[:,j] + EulerTimeStep * res[:,j] # mid point rule for integration to next step
           end

           prob_ref = ODEProblem(method.analytical_∇H, x[:,j], tspan)
           sol = ODE.solve(prob_ref, Tsit5(), dt = EulerTimeStep, abstol = 1e-10, reltol = 1e-10, saveat = trange)

           # sol.u[1] is one of the uniform sampled initial conditions we give, sol.u[2] is the predicted set of values at the EulerTimeStep
           data_ref[:,j] = sol.u[2]
       end

       # add noise
       data_ref_noisy = data_ref .+ method.noise_level .* randn(size(x))

       mapreduce(y -> y^2, +, data_ref_noisy .- hermiteX)
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

       # Regress dynamics onto remaining terms to find sparse a

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