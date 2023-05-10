# This script solves the 2D problem of a point vortex

###############################################################################
##################################THEORY#######################################
###############################################################################



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

# H = 1/2 ∑ᵢ₌₁ᴺ ∑ⱼ₌₁ᴺ ΓᵢΓⱼ log|rᵢ - rⱼ|
                
# Analytical Gradient
function gradient_analytical!(dx, x, param, t) 
    # number of Vortices (n)
    n = div(length(x), 2)
    q = x[1:n]
    p = x[n + 1:end]

    for i in 1:n
        # drᵢ/dt = ∂H/∂Γᵢ = 1/2 ∑ⱼ₌₁ᴺ Γⱼ log|rᵢ - rⱼ| 
        dx[i] = 0.5 * sum(p[j] * log(abs(q[i] - q[j])) for j in 1:n if j != i) #+ p[i] * sum(p[j] * log(abs(q[i] - q[j])) for j in 1:n if j != i)
        
        # dΓᵢ/dt = -∂H/∂rᵢ = -Γᵢ ∑ⱼ₌₁ᴺ Γⱼ (rᵢ - rⱼ)/|rᵢ - rⱼ|^2
        dx[n + i] = - p[i] * sum(p[j] * (q[i] - q[j]) / abs(q[i] - q[j])^2 for j in 1:n if j != i)
    end
    return dx
end

# (q₁,q₂,p₁,p₂) 2 vortex system
num_vortices = 2
num_samples = 1000

z = get_z_vector(num_vortices)
polynomial = polynomial_basis(z, polyorder=2)
z_diff = primal_operator_basis(z, -)
log_diff  = logarithmic_basis(z_diff, polyorder=1)
z_log_power = primal_power_basis(log_diff, 1)
mixed_basis = mixed_states_basis(polynomial, z_log_power)
basis = get_basis_set(mixed_basis)


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# x reference state data 
x = [randn(2*num_vortices) for i in 1:num_samples]

# ẋ reference data 
dx = zeros(2*num_vortices)
param = 0
t = 0
ẋ = [gradient_analytical!(copy(dx), _x, param, t) for _x in x]


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# choose SINDy method
method = HamiltonianSINDy(basis, gradient_analytical!, z, λ = 0.05, noise_level = 0.00)

# generate noisy references data at next time step
y = SparseIdentification.gen_noisy_ref_data(method, x)

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
    idx = rand(1:length(x))

    prob_reference = ODEProblem((dx, t, x, params) -> gradient_analytical!(dx, x, params, t), tspan, tstep, x[idx])
    data_reference = integrate(prob_reference, Gauss(2))

    prob_sindy = ODEProblem((dx, t, x, params) -> vectorfield(dx, x, params, t), tspan, tstep, x[idx])
    data_sindy = integrate(prob_sindy, Gauss(2))

    p1 = plot(xlabel = "Time", ylabel = "q₁")
    plot!(p1, data_reference.t, data_reference.q[:,1], markershape=:star5, label = "Ref q₁")
    plot!(p1, data_sindy.t, data_sindy.q[:,1], markershape=:xcross, label = "Identified q₁")

    p3 = plot(xlabel = "Time", ylabel = "p₁")
    plot!(p3, data_reference.t, data_reference.q[:,3], markershape=:star5, label = "Ref p₁")
    plot!(p3, data_sindy.t, data_sindy.q[:,3], markershape=:xcross, label = "Identified p₁")

    plot!(size=(1000,1000))
    display(plot(p1, p3, title="Numerical vs SINDy q₁ & p₁"))

    p2 = plot(xlabel = "Time", ylabel = "q₂")
    plot!(p2, data_reference.t, data_reference.q[:,2], markershape=:star5, label = "Ref q₂")
    plot!(p2, data_sindy.t, data_sindy.q[:,2], markershape=:xcross, label = "Identified q₂")

    p4 = plot(xlabel = "Time", ylabel = "p₂")
    plot!(p4, data_reference.t, data_reference.q[:,4], markershape=:star5, label = "Ref p₂")
    plot!(p4, data_sindy.t, data_sindy.q[:,4], markershape=:xcross, label = "Identified p₂")

    plot!(size=(1000,1000))
    display(plot(p2, p4, title="Numerical vs SINDy q₂ & p₂"))

end