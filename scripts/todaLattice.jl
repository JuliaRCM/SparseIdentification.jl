# This script solves the Toda lattice problem

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

# Analytical Gradient
function gradient_analytical!(dx, x, p, t) 
    # number of particles (n)
    n = div(length(x), 2)
    q = x[1:n]
    p = x[n + 1:end]

    dx[1:n] = p
    dx[n+1] = exp(-(q[1] - q[end])) - exp(-(q[2] - q[1]))
    for i = 2:n-1
        dx[i + n] = exp(-(q[i] - q[i-1])) - exp(-(q[i+1] - q[i]))
    end
    dx[end] = exp(-(q[end] - q[end-1])) - exp(-(q[1] - q[end]))

    return dx
end

# (q₁,q₂,q₃,q₄,p₁,p₂,p₃,p₄) 4 particle system
num_particles = 4
num_samples = 1000

z = get_z_vector(num_particles)
polynomial = polynomial_basis(z, polyorder=2)
z_diff = primal_operator_basis(z, -)
exponential_diff  = exponential_basis(z_diff, polyorder=1)
z_power = primal_power_basis(exponential_diff, 2)
basis = get_basis_set(polynomial, z_power)


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# x reference state data 
x = [randn(2*num_particles) for i in 1:num_samples]

# ẋ reference data 
dx = zeros(2*num_particles)
p = 0
t = 0
ẋ = [gradient_analytical!(copy(dx), _x, p, t) for _x in x]


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# choose SINDy method
method = HamiltonianSINDy(basis, gradient_analytical!, z, λ = 0.1, noise_level = 0.00)

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

    # plot positions
    p1 = plot(xlabel = "Time", ylabel = "position")
    plot!(p1, data_reference.t, data_reference.q[:,1], markershape=:star5, label = "particle one Ref_pos")
    plot!(p1, data_sindy.t, data_sindy.q[:,1], markershape=:xcross, label = "particle one Iden_pos")

    p3 = plot(xlabel = "Time", ylabel = "position")
    plot!(p3, data_reference.t, data_reference.q[:,3], markershape=:star5, label = "particle two Ref_pos")
    plot!(p3, data_sindy.t, data_sindy.q[:,3], markershape=:xcross, label = "particle two Iden_pos")

    plot!(xlabel = "Time", ylabel = "x_pos", size=(1000,1000))
    display(plot(p1, p3, title="Analytical vs Calculated x Positions"))

    p2 = plot(xlabel = "Time", ylabel = "position")
    plot!(p2, data_reference.t, data_reference.q[:,2], markershape=:star5, label = "particle three Ref_pos")
    plot!(p2, data_sindy.t, data_sindy.q[:,2], markershape=:xcross, label = "particle three Iden_pos")

    p4 = plot(xlabel = "Time", ylabel = "position")
    plot!(p4, data_reference.t, data_reference.q[:,4], markershape=:star5, label = "particle four Ref_pos")
    plot!(p4, data_sindy.t, data_sindy.q[:,4], markershape=:xcross, label = "particle four Iden_pos")

    plot!(xlabel = "Time", ylabel = "y_pos", size=(1000,1000))
    display(plot(p2, p4, title="Analytical vs Calculated y Positions"))

    # plot momenta
    p5 = plot(xlabel = "Time", ylabel = "momentum")
    plot!(p5, data_reference.t, data_reference.q[:,5], markershape=:star5, label = "particle one Ref_mom")
    plot!(p5, data_sindy.t, data_sindy.q[:,5], markershape=:xcross, label = "particle one Iden_mom")

    p7 = plot(xlabel = "Time", ylabel = "momentum")
    plot!(p7, data_reference.t, data_reference.q[:,7], markershape=:star5, label = "particle two Ref_mom")
    plot!(p7, data_sindy.t, data_sindy.q[:,7], markershape=:xcross, label = "particle two Iden_mom")


    plot!(xlabel = "Time", ylabel = "x_mom", size=(1000,1000))
    display(plot(p5, p7, title="Analytical vs Calculated x Momenta"))

    p6 = plot(xlabel = "Time", ylabel = "momentum")
    plot!(p6, data_reference.t, data_reference.q[:,6], markershape=:star5, label = "particle three Ref_mom")
    plot!(p6, data_sindy.t, data_sindy.q[:,6], markershape=:xcross, label = "particle three Iden_mom")

    p8 = plot(xlabel = "Time", ylabel = "momentum")
    plot!(p8, data_reference.t, data_reference.q[:,8], markershape=:star5, label = "particle four Ref_mom")
    plot!(p8, data_sindy.t, data_sindy.q[:,8], markershape=:xcross, label = "particle four Iden_mom")

    plot!(xlabel = "Time", ylabel = "y_mom", size=(1000,1000))
    display(plot(p6, p8, title="Analytical vs Calculated y Momenta"))

end