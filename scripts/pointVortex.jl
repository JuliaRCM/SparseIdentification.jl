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
# drᵢ/dt = ∂H/∂Γᵢ = 1/2 ∑ⱼ₌₁ᴺ Γⱼ log|rᵢ - rⱼ|                 #+ Γᵢ ∑ⱼ₌₁ᴺ ≠ᵢ Γⱼ log|rᵢ - rⱼ|
# dΓᵢ/dt = -∂H/∂rᵢ = -Γᵢ ∑ⱼ₌₁ᴺ Γⱼ (rᵢ - rⱼ)/|rᵢ - rⱼ|^2

# Analytical Gradient
function gradient_analytical!(dx, x, param, t) 
    # number of Vortices (n)
    n = div(length(x), 2)
    q = x[1:n]
    p = x[n + 1:end]

    for i in 1:n
        dx[i] = 0.5 * sum(p[j] * log(abs(q[i] - q[j])) for j in 1:n if j != i) #+ p[i] * sum(p[j] * log(abs(q[i] - q[j])) for j in 1:n if j != i)
        dx[n + i] = - p[i] * sum(p[j] * (q[i] - q[j]) / abs(q[i] - q[j])^2 for j in 1:n if j != i)
    end
    return dx
end


# # H = - Σ pᵢpⱼ/(4π) * log|qᵢ-qⱼ| for i != j (http://www.cds.caltech.edu/~marsden/wiki/uploads/cds140b-08/home/Joris_LectWk4.pdf)
# function gradient_analytical!(dx, x, param, t) 
#     # number of Vortices (n)
#     n = div(length(x), 2)
#     q = x[1:n]
#     p = x[n + 1:end]

#     for i in 1:n
#         # dqᵢ/dt = ∂H/∂pᵢ = - Σpⱼ/(4π) * log|qᵢ-qⱼ| for i != j
#         dx[i] = - sum(p[j]/(4π) * log(abs(q[i] - q[j])) for j in 1:n if j != i)

#         # dpᵢ/dt = -∂H/∂qᵢ = Σpᵢpⱼ/(4π)  * 1/|qᵢ-qⱼ| * sign(qᵢ-qⱼ) for i != j
#         dx[n + i] = sum(p[i]*p[j]/(4π) * 1/(q[i] - q[j]) for j in 1:n if j != i)
#     end
#     return dx
# end

# (q₁,q₂,q₃,q₄,p₁,p₂,p₃,p₄) 4 vortex system
num_vortices = 3
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
test = vectorfield.coefficients

# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

tstep = 0.01
tspan = (0.0,25.0)



# for i in 1:5
#     idx = rand(1:length(x))

#     prob_reference = ODEProblem((dx, t, x, params) -> gradient_analytical!(dx, x, params, t), tspan, tstep, x[idx])
#     data_reference = integrate(prob_reference, Gauss(2))

#     prob_sindy = ODEProblem((dx, t, x, params) -> vectorfield(dx, x, params, t), tspan, tstep, x[idx])
#     data_sindy = integrate(prob_sindy, Gauss(2))

#     # plot positions
#     p1 = plot(xlabel = "Time", ylabel = "position")
#     plot!(p1, data_reference.t, data_reference.q[:,1], markershape=:star5, label = "Vortex one Ref_pos")
#     plot!(p1, data_sindy.t, data_sindy.q[:,1], markershape=:xcross, label = "Vortex one Iden_pos")

#     p3 = plot(xlabel = "Time", ylabel = "position")
#     plot!(p3, data_reference.t, data_reference.q[:,3], markershape=:star5, label = "Vortex two Ref_pos")
#     plot!(p3, data_sindy.t, data_sindy.q[:,3], markershape=:xcross, label = "Vortex two Iden_pos")

#     plot!(xlabel = "Time", ylabel = "x_pos", size=(1000,1000))
#     display(plot(p1, p3, title="Analytical vs Calculated x Positions"))

#     p2 = plot(xlabel = "Time", ylabel = "position")
#     plot!(p2, data_reference.t, data_reference.q[:,2], markershape=:star5, label = "Vortex three Ref_pos")
#     plot!(p2, data_sindy.t, data_sindy.q[:,2], markershape=:xcross, label = "Vortex three Iden_pos")

#     p4 = plot(xlabel = "Time", ylabel = "position")
#     plot!(p4, data_reference.t, data_reference.q[:,4], markershape=:star5, label = "Vortex four Ref_pos")
#     plot!(p4, data_sindy.t, data_sindy.q[:,4], markershape=:xcross, label = "Vortex four Iden_pos")

#     plot!(xlabel = "Time", ylabel = "y_pos", size=(1000,1000))
#     display(plot(p2, p4, title="Analytical vs Calculated y Positions"))

# end


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

    # plot momenta
    p5 = plot(xlabel = "Time", ylabel = "momentum")
    plot!(p5, data_reference.t, data_reference.q[:,5], markershape=:star5, label = "Vortex one Ref_mom")
    plot!(p5, data_sindy.t, data_sindy.q[:,5], markershape=:xcross, label = "Vortex one Iden_mom")

    # p7 = plot(xlabel = "Time", ylabel = "momentum")
    # plot!(p7, data_reference.t, data_reference.q[:,7], markershape=:star5, label = "Vortex two Ref_mom")
    # plot!(p7, data_sindy.t, data_sindy.q[:,7], markershape=:xcross, label = "Vortex two Iden_mom")


    # plot!(xlabel = "Time", ylabel = "x_mom", size=(1000,1000))
    # display(plot(p5, p7, title="Analytical vs Calculated x Momenta"))

    p6 = plot(xlabel = "Time", ylabel = "momentum")
    plot!(p6, data_reference.t, data_reference.q[:,6], markershape=:star5, label = "Vortex three Ref_mom")
    plot!(p6, data_sindy.t, data_sindy.q[:,6], markershape=:xcross, label = "Vortex three Iden_mom")

    # p8 = plot(xlabel = "Time", ylabel = "momentum")
    # plot!(p8, data_reference.t, data_reference.q[:,8], markershape=:star5, label = "Vortex four Ref_mom")
    # plot!(p8, data_sindy.t, data_sindy.q[:,8], markershape=:xcross, label = "Vortex four Iden_mom")

    plot!(xlabel = "Time", ylabel = "y_mom", size=(1000,1000))
    display(plot(p5, p6, title="Analytical vs Calculated y Momenta"))

end