
using DifferentialEquations
using Distributions
using ODE
using Plots
using Random
using SparseIdentification

gr()


# --------------------
# Setup
# --------------------

#Lorenz's parameters (chaotic)
sigma = 10.0
beta = 8/3
rho = 28.0

# generate basis
#  - search space up to fifth order polynomials
#  - no trigonometric functions
basis = CompoundBasis(polyorder = 5, trigonometric = 0)

# initial data
x₀ = [-8.0, 7.0, 27.0]

# 3D system 
nd = length(x₀)


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

p = (sigma, beta, rho)

num_samp = 15

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output
# s depend on size of nd (total dims), 4 in the case here so we use samp_range x samp_range x samp_range x samp_range
s = collect(Iterators.product(fill(samp_range, nd)...))

# compute vector field from x state values
x = [collect(s[i]) for i in eachindex(s)]

x = hcat(x...)

# compute Derivative

# ẋ is of dims [states x iter]
ẋ = Array{Float64}(undef, size(x,1), size(x,2)) 
for i in axes(x,2)
    ẋ[:, i] = lorenz(x[:, i], p, 0)
end

# choose SINDy method
# method = SINDy(lambda = 0.05, noise_level = 0.0)
method = SINDy(lambda = 0.05, noise_level = 0.0, l_dim = size(x, 1), coeff = 0.52, batch_size = 32)

# add noise
ẋnoisy = ẋ .+ method.noise_level .* randn(size(ẋ))

# collect training data
tdata = TrainingData(Float32.(x), Float32.(ẋnoisy))

println("Computing Vector Field...")

# compute vector field using least squares regression (/) solver
solverType = nothing
# vectorfield = VectorField(method, basis, tdata)

# Using BFGS() solver
solverType = OptimSolver()
# vectorfield = VectorField(method, basis, tdata, solver = solverType)

# Using the Neural-Network solver
solverType = NNSolver()
@time vectorfield, model = VectorField(method, basis, tdata, solver = solverType)

# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

if isa(solverType, NNSolver)
    # With T_end=20.0
    # FIGURE 1: LORENZ for T in[0,20]
    # True model:
    tstep = 0.001
    tspan = (0.001, 20.0)
    trange = range(tspan[begin], step = tstep, stop = tspan[end])

    p = (sigma, beta, rho)
    prob = ODEProblem(lorenz, x₀, tspan, p)
    # stored as dims [states x iters] matrix 
    xA = ODE.solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8) 

    #############################################################
    prob_approx = ODEProblem(vectorfield, model[1].W(Float32.(x₀)), tspan)
    xB = ODE.solve(prob_approx, Tsit5(), abstol=1e-8, reltol=1e-8, saveat = trange, tstops = trange)
    # use decoder to get the solution at each timestep
    xsol = hcat([model[2].W(Float32.(xB[:,i])) for i in axes(xB,2)]...)

    println("Plotting Figure 1...")
    title = plot(title = "Lorenz Attractor", grid = false, showaxis = false, bottom_margin = -50Plots.px)
    p1 = plot(xA, vars=(1, 2, 3), color=:auto, xlabel="state 1", ylabel="state 2", zlabel="state 3", label="True Dynamics", legend=:top, linewidth = 1.5)
    p2 = plot(xsol[1, :], xsol[2, :], xsol[3, :], color=:auto, xlabel="state 1", ylabel="state 2", zlabel="state 3", label="Predicted Dynamics", legend=:top, linewidth = 1.5)
    display(plot(title, p1, p2, layout = @layout([A{0.01h}; [B C]]), size=(1000, 500), show=true, reuse=false))
    savefig("Fig1_ex2_NNlorenz.png")

    println("Plotting Figure 2...")
    p3 = plot(xA, vars=(0,1), linecolor = :black, linewidth = 1.5, label="True State 1")
    plot!(p3, xB.t, xsol[1,:], linecolor = :red, linestyle = :dash, linewidth = 1.5, label="Approximate State 1", ylabel="States 1")
    p4 = plot(xA, vars=(0,2), linecolor = :black, linewidth = 1.5, label="True State 2")
    plot!(p4, xB.t, xsol[2,:], linecolor = :red, linestyle = :dash, linewidth = 1.5, label="Approximate State 1", ylabel="States 2")
    display(plot(p3, p4, layout=(1,2), show = true, reuse = false, size=(900,500), xlabel="Time"))
    savefig("Fig2_ex2_NNlorenz.png")

    println("Integrating and Plotting Figure 3...")

    tstep = 0.001
    tspan = (0.001, 250.0)
    trange = range(tspan[begin], step = tstep, stop = tspan[end])
    # True model:
    prob = ODEProblem(lorenz, x₀, tspan, p)
    xA = ODE.solve(prob, abstol=1e-6, reltol=1e-6)
    # Approximate model:
    prob_approx = ODEProblem(vectorfield, model[1].W(x₀), tspan, saveat = trange, tstops = trange)
    xB = ODE.solve(prob_approx, Tsit5(), abstol=1e-6, reltol=1e-6, saveat = trange, tstops = trange)
    # use decoder to get the solution at each timestep
    xsol = hcat([model[2].W(xB[:,i]) for i in axes(xB,2)]...)
    # Figure 3:
    title = plot(title = "Lorenz Attractor", grid = false, showaxis = false, bottom_margin = -50Plots.px)
    p5 = plot(xA, vars=(1, 2, 3), color=:auto, xlabel="state 1", ylabel="state 2", zlabel="state 3", label="True Dynamics", legend=:top, linewidth = 1.5)
    p6 = plot(xsol[1, :], xsol[2, :], xsol[3, :], color=:auto, xlabel="state 1", ylabel="state 2", zlabel="state 3", label="Predicted Dynamics", legend=:top, linewidth = 1.5)
    display(plot(title, p5, p6, layout = @layout([A{0.01h}; [B C]]), size=(1000, 500), show=true, reuse=false))
    savefig("Fig3_ex2_NNlorenz.png")
    
else
    # With T_end=20.0
    # FIGURE 1: LORENZ for T in[0,20]
    # True model:
    tstep = 0.001
    tspan = (0.001, 20.0)
    trange = range(tspan[begin], step = tstep, stop = tspan[end])

    p = (sigma, beta, rho)
    prob = ODEProblem(lorenz, x₀, tspan, p)
    # stored as dims [states x iters] matrix 
    xA = ODE.solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12) 

    # Approximate model:
    prob_approx = ODEProblem(vectorfield, x₀, tspan)
    xB = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)


    # ----------------------------------------
    # Plot Results
    # ----------------------------------------

    println("Plotting...")

    p1 = plot(xA, vars=(1,2,3), xlabel="x", ylabel="y", zlabel="z", label="true")
    p2 = plot(xB, vars=(1,2,3), xlabel="x", ylabel="y", zlabel="z", label="approx")
    display(plot(p1, p2, layout=(1,2), show = true, reuse = false, size=(900,500)))

    savefig("Fig1_ex2_lorenz.png")

    println("Plotting Figure 2...")

    # Figure 2:
    p3 = plot(xA,vars=(0,1), linecolor = :black, linewidth = 1.5, label="true_state1")
    plot!(p3, xB,vars=(0,1), linecolor = :red, linestyle = :dash, linewidth = 1.5, label="approx_state1")
    p4 = plot(xA,vars=(0,2), linecolor = :black, linewidth = 1.5, label="true_state2")
    plot!(p4, xB,vars=(0,2), linecolor = :red, linestyle = :dash, linewidth = 1.5, label="approx_state2")
    display(plot(p3, p4, layout=(1,2), show = true, reuse = false, size=(900,500), xlabel="Time", ylabel="X"))

    savefig("Fig2_ex2_lorenz.png")


    println("Integrating and Plotting Figure 3...")

    # With T_end=250.0
    # True model:
    tstep = 0.001
    tspan = (0.001, 250.0)
    trange = range(tspan[begin], step = tstep, stop = tspan[end])

    p = (sigma, beta, rho)
    prob = ODEProblem(lorenz, x₀, tspan, p)
    xA = ODE.solve(prob, abstol=1e-6, reltol=1e-6) # stored as dims [3 x iters] matrix 

    # Approximate model:
    prob_approx = ODEProblem(vectorfield, x₀, tspan)
    xB = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)

    # Figure 3:
    dtA = [0; diff(xA.t)]
    dtB = [0; diff(xB.t)]
    p5 = plot(xA[1,:], xA[2,:], xA[3,:], zcolor=dtA, xlabel="x", ylabel="y", zlabel="z", label="true")
    p6 = plot(xB[1,:], xB[2,:], xB[3,:], zcolor=dtB, xlabel="x", ylabel="y", zlabel="z", label="approx")
    display(plot(p5, p6, layout=(1,2), show = true, reuse = false, size=(900,500)))

    savefig("Fig3_ex2_lorenz.png")
end