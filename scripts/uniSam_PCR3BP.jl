# The Planar Circular Restricted Three-Body Problem (PCR3BP) is 
# a special case of the n-body problem in which two massive bodies 
# orbit around their center of mass while a third, massless body moves 
# in their gravitational field. The motion of the massless body is restricted 
# to the plane of the two massive bodies and is influenced only by their 
# gravitational forces. These equations describe the motion of the massless body 
# in the plane of the massive bodies. The solutions of these equations give the 
# trajectory of the massless body as it moves under the influence of the gravitational 
# field of the two massive bodies. The PCR3BP is a highly nonlinear problem and does not 
# have an analytical solution, but numerical methods can be used to approximate the solutions of these equations.


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
const trig_wave_num = 0
######################################################################
######################################################################

# 1 dim each of [q₁, q₂, p₁, p₂] gives 4*d = 4 variables
out = zeros(nd)

# initialize all variables to be above sparsification parameter (λ)
G = 6.6743e-11 # gravitational constant
big_m = 100 # masses of the two massive bodies, assumed equal
small_m = 0.01 # mass of assumed masslesss body
dist = 1 # distances between the masless body and the two massive bodies, assumed equal
R = 10 # distance between the two massive bodies

# Gradient function of the 2D hamiltonian
# x_cm = -big_m * R / (big_m + big_m)
# grad_H_ana(x) = [x[3]; 
#                 x[4]; 
#                 -G * (big_m + big_m) * x[1] / (dist^3) - G * small_m * (x[1] - x_cm) / dist^3; 
#                 -G * (big_m + big_m) * x[2] / (dist^3) - G * small_m * x[2] / dist^3]
alpha = 1
wₚ = 1
grad_H_ana(x) = [alpha*x[1]; 
                wₚ*x[4]; 
                -alpha*x[3]; 
                -wₚ*x[2]]

grad_H_ana(x, p, t) = grad_H_ana(x)

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# number of samples
num_samp = 8

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
# (λ parameter must be close to noise value so that only coeffs with value around the noise are sparsified away)
method = HamiltonianSINDy(grad_H_ana, λ = 0.05, noise_level = 0.05, polyorder = polyorder, trigonometric = trig_wave_num)

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
