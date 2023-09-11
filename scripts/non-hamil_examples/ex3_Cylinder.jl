
using DifferentialEquations
using Distributions
using ODE
using Plots
using Random 
using LinearAlgebra
using DataFrames
using CSV
using SparseIdentification

gr()


# --------------------
# Setup
# --------------------

println("Setting Up...")

dt = 0.02

r = 2

# search space up to fifth order polynomials
polyorder = 5

# no trigonometric functions
usesine = false

# lambda parameter
lambda = 0.00002

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Getting Training Data...")

# load data from first run and compute derivative
POD_coeff_alpha = CSV.read(joinpath(@__DIR__, "..",  "data\\POD_coeff_alpha.csv"), DataFrame; delim=",")

POD_coeff_alphaS = CSV.read(joinpath(@__DIR__, "..",  "data\\POD_coeff_alphaS.csv"), DataFrame)

# need to provide column headers in datafiles first to avoid first line of data being counted as column heads
x = [POD_coeff_alpha[1:5000, 1:r] POD_coeff_alphaS[1:5000, 1]]

# convert to matrix
x = Matrix{Float64}(x)

# transpose x to match other test cases
x = x'

M = size(x, 1)

# noise level
eps = 0

ẋ = Array{Float64}(undef, M, size(x, 2) -3 -2) 

for i=3:size(x, 2)-3
    for k=1:M
        ẋ[k, i-2] = (1 / (12 * dt)) * (-x[k, i+2] + 8 .* x[k, i+1] -8 .* x[k, i-1] + x[k, i-2])
    end
end  

# load data from second run and compute derivative
POD_coeff_run1_alpha = CSV.read(joinpath(@__DIR__, "..",  "data\\POD_coeff_run1_alpha.csv"), DataFrame; delim=",")

POD_coeff_run1_alphaS = CSV.read(joinpath(@__DIR__, "..",  "data\\POD_coeff_run1_alphaS.csv"), DataFrame)

x1 = [POD_coeff_run1_alpha[1:3000, 1:r] POD_coeff_run1_alphaS[1:3000, 1]]

# Assuming you have a DataFrame named df
x1 = dropmissing(x1)

# convert to matrix
x1 = Matrix{Float64}(x1)

# transpose x1 to match other test cases
x1 = x1'

M = size(x1, 1)

ẋ_one = Array{Float64}(undef, M, size(x1, 2)-3-2) 
for i=3:size(x1, 2) -3
    for k=1:M
        ẋ_one[k, i-2] = (1 / (12 * dt)) * (-x1[k, i+2] + 8 * x1[k, i+1] -8 * x1[k, i-1] + x1[k, i-2])
    end
end  

# concatenate
x = [x[:, 3:end-3] x1[:, 3:end-3]]

ẋ = [ẋ ẋ_one]

# ------------------------------------------------------------
# Pool Data (evaluate library of candidate basis functions on training data)
# ------------------------------------------------------------

println("Pool Data...")

# Θ = evaluate(x, basis)

# m = size(Θ, 2)

ẋ = ẋ[:, 1:end]


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

println("Sparsify Dynamics...")

# collect training data
tdata = TrainingData(x, ẋ)

# choose SINDy method
method = SINDy(lambda = lambda, noise_level = 0.05, batch_size = 0)

# generate basis
basis = CompoundBasis()

# compute vector field
vectorfield = VectorField(method, basis, tdata)

#Ξ = sparsify_dynamics(Θ, ẋ, lambda)
# Ξ = sparsify_dynamics(Θ, ẋ, lambda; solver = OptimSolver())


######Test to check if Theta and dx from matlab are equal to theta and dx obtained in Julia
# #load data from first run and compute derivative
# check_dx = CSV.read(joinpath(@__DIR__, "data\\check_dx.csv"), DataFrame; delim=",")

# check_Theta = CSV.read(joinpath(@__DIR__, "data\\check_Theta.csv"), DataFrame; delim=",")

# #convert to matrix
# check_dx = Matrix(check_dx)

# check_Theta = Matrix(check_Theta)

# all(dx .== check_dx')
# all(Theta .== check_Theta)
##########################################################################################



# Note: There are constant order terms... this is because fixed points
# are not at (0,0,0) for this data
# To look at data output of pool
# poolDataLIST(["x", "y", "z"], Ξ, M, polyorder, usesine)

# second figure: initial portion
# Figure(1)

# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System One...")

tspan = (0.0, 100.0)

x₀ = x[:, 1]

# p = (Ξ = Ξ, basis = basis)

# prob_approx = ODEProblem(sparse_galerkin!, x₀, tspan, p)
# xD = ODE.solve(prob_approx, Tsit5(), abstol=1e-8, reltol=1e-8)

# Approximate model:
prob_approx = ODEProblem(vectorfield, x₀, tspan)
xD = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10)

# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting Fig One...")

p1 = plot(x[1,1:end-6], x[2,1:end-6], x[3,1:end-6], xlabel="x", ylabel="y", zlabel="z", label="Data", color="black")
p2 = plot(xD[1,:], xD[2,:], xD[3,:], xlabel="x", ylabel="y", zlabel="z", label="Identified", color="red")
l = @layout [a b]
display(plot(p1, p2, layout = l))

#second figure: Lorenz for t=20

# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System Two...")

tspan = (0.0, 95.0)
x₀ = x[:, 5001]

prob_approx = ODEProblem(vectorfield, x₀, tspan)
xD = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10)

# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting Fig Two...")

p1 = plot(x[1,5001:end], x[2,5001:end], x[3,5001:end], xlabel="x", ylabel="y", zlabel="z", label="Data", color="black")
p2 = plot(xD[1,:], xD[2,:], xD[3,:], xlabel="x", ylabel="y", zlabel="z", label="Identified", color="red")
l = @layout [a b]
display(plot(p1, p2, layout = l))