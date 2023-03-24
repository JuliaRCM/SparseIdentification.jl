#This file uses uniform sampling to find the SINDY solution of a hamiltonian system in 1D and 2D

using DifferentialEquations
using Distributions
using ODE
using Plots
using Random
using SparseIdentification
using Zygote
using ForwardDiff
using Optim

gr()


# --------------------
# Setup
# --------------------

println("Setting up...")

# define the number of variables, q,p in this case gives 2 variables
const d = 1

# 1D system with 2 variables [q₁, p₁]
const nd = 2d

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 3 

# maximum wave number of trig basis for function library to explore
# trig_wave_num can be adjusted if higher frequency arguments expected
const trig_wave_num = 3

# 1 dim each of p and q gives 2*d = 2 variables
out = zeros(nd)

# initialize analytical function, keep ϵ bigger than lambda so system is identifiable
ϵ = 0.5
m = 1

# one-dim simple harmonic oscillator (not used anywhere only in case some testing needed)
# H_ana(x, p, t) = ϵ * x[1]^2 + 1/(2*m) * x[2]^2

# Gradient function of the 1D hamiltonian
#grad_H_ana(x) = [x[2]; -2ϵ * x[1]]
grad_H_ana(x) = [x[2]; sin(x[1])]
grad_H_ana(x, p, t) = grad_H_ana(x)

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# number of samples
num_samp = 100

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output

# compute vector field from x state values
# stored as matrix with dims [nd,ntime]
x = zeros(nd, num_samp*num_samp)
ẋ = zero(x)
s = collect(Iterators.product(samp_range,samp_range))

for j in eachindex(s)
    x[:,j] .= s[j]
    ẋ[:,j] .= grad_H_ana(x[:,j])
end

# collect training data
tdata = TrainingData(x, ẋ)


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# choose SINDy method
# (lambda parameter must be close to noise value so that only coeffs with value around the noise are sparsified away)
method = HamiltonianSINDy(lambda = 0.05, noise_level = 0.05, polyorder = polyorder, trigonometric = trig_wave_num)

# compute vector field
vectorfield = VectorField(method, tdata)

println(vectorfield.coefficients)


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

# ----------------------------------------
# Plot error in approximate gradient
# ----------------------------------------

println("Compute approximate gradient...")

ẋid = zero(ẋ)

for j in axes(ẋid, 2)
    @views vectorfield(ẋid[:,j], x[:,j])
end

# calculate difference between answers
ẋerr = sqrt.((ẋid .- ẋ).^2 ./ ẋ.^2)

plot(heatmap(ẋerr), title="Cos(q₁): Relative difference b/w analytical and calculated gradient in a 1D system", titlefontsize=8)
savefig("uniform_sample_ham_1d.png")


# ----------------------------------------
# Plot some solutions
# ----------------------------------------

tstep = 0.01
tspan = (0.0,25.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

for i in 1:5
    idx = rand(1:length(s))

    prob_reference = ODEProblem(grad_H_ana, x[:,idx], tspan)
    data_reference = ODE.solve(prob_reference, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)

    prob_sindy = ODEProblem(vectorfield, x[:,idx], tspan)
    data_sindy = ODE.solve(prob_sindy, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange) 

    p1 = plot(xlabel = "Time", ylabel = "q₁")
    scatter!(p1, data_reference.t, data_reference[1,:], label = "Data q₁")
    scatter!(p1, data_sindy.t, data_sindy[1,:], markershape=:xcross, label = "Identified q₁")
    plot!(size=(1000,1000))

    p2 = plot(xlabel = "Time", ylabel = "p₁")
    scatter!(p2, data_reference.t, data_reference[2,:], label = "Data p₁")
    scatter!(p2, data_sindy.t, data_sindy[2,:], markershape=:xcross, label = "Identified p₁")
    plot!(size=(1000,1000))
    display(plot(p1, p2, title="Analytical vs Calculated gradient in a 2D system"))
    #savefig("uniform_sample_ham_1d_$i.png")
end
