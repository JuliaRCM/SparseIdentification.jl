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


using GeometricIntegrators
using Distributions
using Plots
using Random
using SparseIdentification
using Optim

gr()


# --------------------
# Setup
# --------------------

println("Setting up...")

# define the number of variables of the phase-space, q,p in this case gives 2 variables
const d = 2


# 2D system with 4 variables [q₁, q₂, p₁, p₂]
const nd = 2d


# 2 dim each, [q₁, q₂, p₁, p₂] gives 2*d = 4 variables
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

function grad_H_ana!(dx, x, p, t)
    dx .= grad_H_ana(x)
end

# Guess some basis functions
z = get_z_vector(Int(nd/2))
polynomial = polynomial_basis(z, polyorder=3)
# trigonometric  = trigonometric_basis(z, max_coeff=1)
basis = get_basis_set(polynomial)

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# number of samples
num_samp = 10

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output

# s depend on size of nd (total dims), 4 in the case here so we use samp_range x samp_range x samp_range x samp_range
s = collect(Iterators.product(fill(samp_range, nd)...))

# compute vector field from x state values
x = [collect(s[i]) for i in eachindex(s)]
dx = zeros(nd)
p = 0
t = 0
ẋ = [grad_H_ana!(copy(dx), _x, p, t) for _x in x]

# collect training data
tdata = TrainingData(x, ẋ)


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# choose SINDy method
# (λ parameter must be close to noise value so that only coeffs with value around the noise are sparsified away)
# method = HamiltonianSINDy(grad_H_ana, λ = 0.05, noise_level = 0.05, polyorder = polyorder, trigonometric = trig_wave_num)
method = HamiltonianSINDy(basis, grad_H_ana!, z, λ = 0.05, noise_level = 0.0, noiseGen_timeStep = 0.0)

# compute vector field
println("Compute approximate gradient...")

# vectorfield = VectorField(method, tdata)
vectorfield = VectorField(method, tdata)
println(vectorfield.coefficients)


println("Plotting...")

# ----------------------------------------
# Plot some solutions
# ----------------------------------------

tstep = 0.01
tspan = (0.0,25.0)

for i in 1:5
    idx = rand(1:length(s))

    prob_reference = ODEProblem((dx, t, x, params) -> grad_H_ana!(dx, x, params, t), tspan, tstep, x[idx])
    data_reference = integrate(prob_reference, Gauss(2))

    prob_sindy = ODEProblem((dx, t, x, params) -> vectorfield(dx, x, params, t), tspan, tstep, x[idx])
    data_sindy = integrate(prob_sindy, Gauss(2))

    p1 = plot(xlabel = "Time", ylabel = "q₁")
    scatter!(p1, data_reference.t, data_reference.q[:,1], label = "Data q₁")
    scatter!(p1, data_sindy.t, data_sindy.q[:,1], markershape=:xcross, label = "Identified q₁")

    p3 = plot(xlabel = "Time", ylabel = "p₁")
    scatter!(p3, data_reference.t, data_reference.q[:,3], label = "Data p₁")
    scatter!(p3, data_sindy.t, data_sindy.q[:,3], markershape=:xcross, label = "Identified p₁")

    plot!(size=(1000,1000))
    display(plot(p1, p3, title="Analytical vs Calculated q₁ & p₁ in a 2D system with Euler"))

    p2 = plot(xlabel = "Time", ylabel = "q₂")
    scatter!(p2, data_reference.t, data_reference.q[:,2], label = "Data q₂")
    scatter!(p2, data_sindy.t, data_sindy.q[:,2], markershape=:xcross, label = "Identified q₂")

    p4 = plot(xlabel = "Time", ylabel = "p₂")
    scatter!(p4, data_reference.t, data_reference.q[:,4], label = "Data p₂")
    scatter!(p4, data_sindy.t, data_sindy.q[:,4], markershape=:xcross, label = "Identified p₂")

    plot!(size=(1000,1000))
    display(plot(p2, p4, title="Analytical vs Calculated q₂ & p₂ in a 2D system with Euler"))

end