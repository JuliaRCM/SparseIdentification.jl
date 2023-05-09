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

# ∂H/∂rᵢ = -Γᵢ ∑ⱼ₌₁ᴺ Γⱼ (rᵢ - rⱼ)/|rᵢ - rⱼ|^2
# ∂H/∂Γᵢ = 1/2 ∑ⱼ₌₁ᴺ Γⱼ log|rᵢ - rⱼ| + Γᵢ ∑ⱼ₌₁ᴺ ≠ᵢ Γⱼ log|rᵢ - rⱼ|

# Analytical Gradient
function gradient_analytical!(dx, x, param, t) 
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

# (q₁,q₂,q₃,q₄,p₁,p₂,p₃,p₄) 4 vortex system
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
test = get_numCoeffs(basis)

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# initialize vector of matrices to store ODE solve output
# Define range for positive values
range = 0:0.01:3

# Sample random values from the range
samples = rand(range, 2*num_vortices, num_samples)

# Transpose the samples and store in s
s = [samples[:,i]' for i in 1:num_samples]

# s depend on size of nd (total dims), 2*num_vortices in this case
# s = collect(Iterators.product(fill(samples, 2*num_vortices)...))


# compute vector field from x state values
x = [collect(s[i]) for i in eachindex(s)]
x = [vec(m) for m in x]

dx = zeros(2*num_vortices)
param = 0
t = 0
ẋ = [gradient_analytical!(copy(dx), _x, param, t) for _x in x]
if any([any(isnan.(ẋ[i])) for i in 1:length(ẋ)])
    println("isnan")
    ẋ = [any(isnan.(v)) ? randn(length(v)) : v for v in ẋ]
end





# # x reference state data 
# function generate_q(n::Int, num_vortices::Int, eps::Float64)
#     q = [randn(num_vortices) for i in 1:n]
#     for q_temp in q
#         for i in 1:num_vortices, j in (i+1):num_vortices
#             diff = q_temp[i] - q_temp[j]
#             log_diff = log(abs(diff))
#             while log_diff < eps
#                 q_temp[i] += randn(1)[1]
#                 diff = q_temp[i] - q_temp[j]
#                 log_diff = log(abs(diff))
#             end
#         end
#     end
#     return q
# end

# # q = generate_q(num_samples, num_vortices, 1e-9)
# # p = [randn(num_vortices) for i in 1:num_samples]

# # x = [vcat(q[i], p[i]) for i in 1:num_samples]
# x = [randn(2*num_vortices) for i in 1:num_samples]
# # ẋ reference data 
# dx = zeros(2*num_vortices)
# param = 0
# t = 0
# ẋ = [gradient_analytical!(copy(dx), _x, param, t) for _x in x]


# for i in 1:length(ẋ)
#     for j in 1:length(ẋ[i])
#         if abs(ẋ[i][j]) < 1e-4
#             println("Element at index ($i, $j) is below eps.")
#         end
#     end
# end







# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# choose SINDy method
method = HamiltonianSINDy(basis, gradient_analytical!, z, λ = 0.01, noise_level = 0.00)

# generate noisy references data at next time step
# y = SparseIdentification.gen_noisy_ref_data(method, x)
y = [x[i] .+ randn(length(x[i])) for i in 1:length(x)]


# collect training data
tdata = TrainingData(x, ẋ, y)

# compute vector field
vectorfield = VectorField(method, tdata, solver = BFGS()) 

println(vectorfield.coefficients)

test = vectorfield.coefficients
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

    # plot positions
    p1 = plot(xlabel = "Time", ylabel = "position")
    plot!(p1, data_reference.t, data_reference.q[:,1], markershape=:star5, label = "Vortex one Ref_pos")
    plot!(p1, data_sindy.t, data_sindy.q[:,1], markershape=:xcross, label = "Vortex one Iden_pos")

    p3 = plot(xlabel = "Time", ylabel = "position")
    plot!(p3, data_reference.t, data_reference.q[:,3], markershape=:star5, label = "Vortex two Ref_pos")
    plot!(p3, data_sindy.t, data_sindy.q[:,3], markershape=:xcross, label = "Vortex two Iden_pos")

    plot!(xlabel = "Time", ylabel = "x_pos", size=(1000,1000))
    display(plot(p1, p3, title="Analytical vs Calculated x Positions"))

    p2 = plot(xlabel = "Time", ylabel = "position")
    plot!(p2, data_reference.t, data_reference.q[:,2], markershape=:star5, label = "Vortex three Ref_pos")
    plot!(p2, data_sindy.t, data_sindy.q[:,2], markershape=:xcross, label = "Vortex three Iden_pos")

    p4 = plot(xlabel = "Time", ylabel = "position")
    plot!(p4, data_reference.t, data_reference.q[:,4], markershape=:star5, label = "Vortex four Ref_pos")
    plot!(p4, data_sindy.t, data_sindy.q[:,4], markershape=:xcross, label = "Vortex four Iden_pos")

    plot!(xlabel = "Time", ylabel = "y_pos", size=(1000,1000))
    display(plot(p2, p4, title="Analytical vs Calculated y Positions"))

end