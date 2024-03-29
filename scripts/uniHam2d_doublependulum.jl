#This file uses uniform sampling to find the SINDY solution of a hamiltonian system in 1D and 2D
using DifferentialEquations
using ODE
using Distributions
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

#############################################################
#############################################################
# IMP SETUP NOTE: 2D system with 4 variables [q₁, q₂, p₁, p₂]
const nd = 4d
#############################################################
#############################################################

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 3

######################################################################
######################################################################
# maximum wave number of trig basis for function library to explore
# trig_wave_num can be adjusted if higher frequency arguments expected
const trig_wave_num = 5
######################################################################
######################################################################

# 1 dim each of [q₁, q₂, p₁, p₂] gives 4*d = 4 variables
out = zeros(nd)

# initialize analytical function, keep ϵ bigger than lambda so system is identifiable
m = 1 # m₁=m₂
l₁ = 1
l₂ = 1
g = 9.81

# h₁(x) = (x[3]*x[4]*sin(x[1]-x[2])) / (l₁*l₂*(m + m*(sin(x[1]-x[2]))^2))
# h₂(x) = (m * l₂^2 * x[3]^2 + (m+m) * l₁^2 * x[4]^2 - 2*m*l₁*l₂*x[3]*x[4]*cos(x[1]-x[2])) / 
#         (2*l₁^2 * l₂^2 * (m + m * (sin(x[1]-x[2]))^2)^2)

# grad_H_ana(x) = [(l₂*x[3] - l₁*x[4] * cos(x[1]-x[2])) / (l₁^2 * l₂ * (m + m*(sin(x[1]-x[2]))^2)); 
#                  (-m*l₂*x[3] * cos(x[1]-x[2]) + (m+m)*(l₁*x[4])) / (m * l₁ * l₂^2 * (m + m*(sin(x[1]-x[2]))^2));
#                  -(m+m)*g*l₁*sin(x[1]) - h₁(x) + h₂(x) * sin(2*(x[1]-x[2]));
#                  -m*g*l₂*sin(x[2]) + h₁(x) - h₂(x) * sin(2*(x[1]-x[2]))];

l = 1 # assume both pendulums have same length

# grad_H_ana(x) = [(m*l*l * x[3] * x[4] * sin(x[1] - x[2]) - (m + m) * g * l * sin(x[1]) - m * l * l * x[4]^2 * sin(x[1] - x[2]) * cos(x[1] - x[2])) / (l^2 * (m + m * sin(x[1] - x[2])^2));
#                  (m * l * l * (x[3]^2 * sin(x[1] - x[2]) - g * sin(x[2]) + x[4]^2 * sin(x[1] - x[2]) * cos(x[1] - x[2]))) / (l^2 * (m + m * sin(x[1] - x[2])^2));
#                  ((m + m) * g * l * sin(x[1]) - m * l * l * (x[4]^2 * sin(x[1] - x[2]) - x[3] * x[4] * sin(x[1] - x[2]) * cos(x[1] - x[2]))) / (l^2 * (m + m * sin(x[1] - x[2])^2));
#                  ((m + m) * g * l * sin(x[2]) - m * l * l * (x[3]^2 * sin(x[1] - x[2]) - x[3] * x[4] * sin(x[1] - x[2]) * cos(x[1] - x[2]))) / (l^2 * (m + m * sin(x[1] - x[2])^2))]


grad_H_ana(x) = [(m+m) * l^2 * x[3] + m * l^2 * x[4] * cos(x[1]-x[2]);
                 m * l^2 * x[4] + m *l^2 * x[3] * cos(x[1]-x[2]);
                 - (m * l^2 * x[3] * x[4] * -sin(x[1]-x[2]) - (m + m) * g * l * -sin(x[1]));
                 - (m * l^2 * x[3] * x[4] * -sin(x[1]-x[2]) - m * g * l * -sin(x[2]))]

grad_H_ana(x, p, t) = grad_H_ana(x)

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# number of samples
num_samp = 12

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output

# compute vector field from x state values
# stored as matrix with dims [nd,ntime]
x = zeros(nd, num_samp^nd)
ẋ = zero(x)
s = collect(Iterators.product(samp_range,samp_range, samp_range, samp_range))

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
method = HamiltonianSINDy(grad_H_ana, λ = 0.0005, noise_level = 0.0005, polyorder = polyorder, trigonometric = trig_wave_num)

# compute vector field
vectorfield = VectorField(method, tdata, solver = ConjugateGradient())

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

    p3 = plot(xlabel = "Time", ylabel = "p₁")
    scatter!(p3, data_reference.t, data_reference[3,:], label = "Data p₁")
    scatter!(p3, data_sindy.t, data_sindy[3,:], markershape=:xcross, label = "Identified p₁")

    plot!(size=(1000,1000))
    display(plot(p1, p3, title="Analytical vs Calculated q₁ & p₁ in a 2D system with Euler"))

    p2 = plot(xlabel = "Time", ylabel = "q₂")
    scatter!(p2, data_reference.t, data_reference[2,:], label = "Data q₂")
    scatter!(p2, data_sindy.t, data_sindy[2,:], markershape=:xcross, label = "Identified q₂")

    p4 = plot(xlabel = "Time", ylabel = "p₂")
    scatter!(p4, data_reference.t, data_reference[4,:], label = "Data p₂")
    scatter!(p4, data_sindy.t, data_sindy[4,:], markershape=:xcross, label = "Identified p₂")

    plot!(size=(1000,1000))
    display(plot(p2, p4, title="Analytical vs Calculated q₂ & p₂ in a 2D system with Euler"))

end
