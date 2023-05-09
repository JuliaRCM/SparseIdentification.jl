# This file uses uniform sampling to find the SINDy solution of a hamiltonian system in 1D and 2D

using Distributions
using GeometricIntegrators
using Optim
using Plots
using Random
using SparseIdentification

gr()



# --------------------
# Setup
# --------------------

println("Setting up...")

# 2D system with 4 variables [q₁, q₂, p₁, p₂]
const nd = 4

z = get_z_vector(nd/2)
polynomial = polynomial_basis(z, polyorder=3)
trigonometric  = trigonometric_basis(z, max_coeff=1)
prime_diff = primal_operator_basis(z, -)
basis = get_basis_set(polynomial, trigonometric, prime_diff)
# initialize analytical function, keep λ smaller than ϵ so system is identifiable
ϵ = 0.5
m = 1

# two-dim simple harmonic oscillator (not used anywhere only in case some testing needed)
# H_ana(x, p, t) = ϵ * x[1]^2 + ϵ * x[2]^2 + 1/(2*m) * x[3]^2 + 1/(2*m) * x[4]^2
# H_ana(x, p, t) = cos(x[1]) + cos(x[2]) + 1/(2*m) * x[3]^2 + 1/(2*m) * x[4]^2

# Gradient function of the 2D hamiltonian

# Analytical Gradient
function grad_H_ana!(dx, x, param, t) 
    # number of Vortices (n)
    n = div(length(x), 2)
    q = x[1:n]
    p = x[n + 1:end]

    for i in 1:n
        dx[i] = -p[i] * sum(p[j] * (q[i] - q[j]) / (abs(q[i] - q[j])^2) for j in 1:n if j != i)
        dx[n + i] = 0.5 * sum(p[j] * log(abs(q[i] - q[j])) for j in 1:n if j != i) + p[i] * sum(p[j] * log(abs(q[i] - q[j])) for j in 1:n if j != i)
    end
    return dx
end
# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# number of samples
num_samp = 10

# samples in p and q space
samp_range = LinRange(-15, 20, num_samp)

# initialize vector of matrices to store ODE solve output

# s depend on size of nd (total dims), 4 in the case here so we use samp_range x samp_range x samp_range x samp_range
s = collect(Iterators.product(fill(samp_range, nd)...))


# compute vector field from x state values
x = [collect(s[i]) for i in eachindex(s)]
dx = zeros(nd)
p = 0
t = 0
ẋ = [grad_H_ana!(copy(dx), _x, p, t) for _x in x]
if any([any(isnan.(ẋ[i])) for i in 1:length(ẋ)])
    println("isnan")
    x = [any(isnan.(v)) ? randn(length(v)) : v for v in ẋ]
    ẋ = [grad_H_ana!(copy(dx), _x, p, t) for _x in x]
end


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# choose SINDy method
# (λ parameter must be close to noise value so that only coeffs with value around the noise are sparsified away)
# noiseGen_timeStep chosen randomly for now
method = HamiltonianSINDy(basis, grad_H_ana!, z, λ = 0.05, noise_level = 0.05, noiseGen_timeStep = 0.05)

# generate noisy references data at next time step
# y = SparseIdentification.gen_noisy_ref_data(method, x)
y = [x[i] .+ randn(length(x[i])) for i in 1:length(x)]

# collect training data
tdata = TrainingData(x, ẋ, y)

# compute vector field
vectorfield = VectorField(method, tdata, solver = BFGS())

println(vectorfield.coefficients)


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

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