# This script solves the 2D n-body problem for a number of planets

###############################################################################
##################################THEORY#######################################
###############################################################################



using Distributions
using GeometricIntegrators
using Plots
using Random
using SparseIdentification
using Optim

include("solarsystem.jl") 

gr()


# --------------------
# Setup
# --------------------

println("Setting up...")

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 2

# mass of each particle
m = 1

# Analytical Gradient
function gradient_analytical!(dx, x, t) 
    # number of particles (n)
    n = div(length(x), 2)
    q = x[1:div(end, 2)]
    p = x[div(end, 2) + 1:end]

    dx = zeros(2*n)

    dx[1:n] .= p
    dx[n+1] = exp(-(q[1] - q[end])) - exp(-(q[2] - q[1]))
    for i = 2:n-1
        dx[i + n] = exp(-(q[i] - q[i-1])) - exp(-(q[i+1] - q[i]))
    end
    dx[end] = exp(-(q[end] - q[end-1])) - exp(-(q[1] - q[end]))

    return dx
end


# (q₁,q₂,q₃,q₄,p₁,p₂,p₃,p₄) 4 particle system

# Initial conditions
num_particles = 4
q₀ = rand(num_particles)
p₀ = rand(num_particles)
x₀ = [q₀; p₀]

tstep = 0.01
tspan = (0.0, 1e3)
trange = range(tspan[begin], step = tstep, stop = tspan[end])


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob_reference = ODEProblem((dx, t, x, params) -> gradient_analytical!(dx, x, t), tspan, tstep, x₀)
data_reference = integrate(prob_reference, Gauss(2))
x_ref = data_reference.q

# choose SINDy method
method = HamiltonianSINDy(λ = 0.08, noise_level = 0.00, polyorder = polyorder)

# make ẋ reference data 
t = 0.0
dx = zeros(num_particles)
ẋ_ref = [gradient_analytical!(copy(dx),_x, t) for _x in x_ref]


# Flatten x and ẋ for TrainingData struct
# position data
x = Float64[]
x = [vcat(x, vec(_x)) for _x in x_ref]

# momentum data
ẋ = Float64[]
ẋ = [vcat(ẋ, vec(_ẋ)) for _ẋ in ẋ_ref]


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# collect training data
tdata = TrainingData(x, ẋ)

# compute vector field
vectorfield = VectorField(method, tdata, solver = BFGS()) 

println(vectorfield.coefficients)


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

tstep = 0.01
tspan = (0.0, 5e3)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

prob_reference = GeometricIntegrators.ODEProblem((dx, t, x, params) -> gradient_analytical!(dx, x, t), tspan, tstep, x₀)
data_reference = integrate(prob_reference, Gauss(2))

prob_sindy = GeometricIntegrators.ODEProblem((dx, t, x, params) -> vectorfield(dx, x, params, t), tspan, tstep, x[1])
data_sindy = integrate(prob_sindy, Gauss(2))

# plot positions
p1 = plot(xlabel = "Time", ylabel = "position")
plot!(p1, data_reference.t, data_reference.q[:,1], label = "particle one Ref_pos")
plot!(p1, data_sindy.t, data_sindy.q[:,1], markershape=:xcross, label = "particle one Iden_pos")

p3 = plot(xlabel = "Time", ylabel = "position")
plot!(p3, data_reference.t, data_reference.q[:,3], label = "particle two Ref_pos")
plot!(p3, data_sindy.t, data_sindy.q[:,3], markershape=:xcross, label = "particle two Iden_pos")

plot!(xlabel = "Time", ylabel = "x_pos", size=(1000,1000))
display(plot(p1, p3, title="Analytical vs Calculated x Positions"))

p2 = plot(xlabel = "Time", ylabel = "position")
plot!(p2, data_reference.t, data_reference.q[:,2], label = "particle three Ref_pos")
plot!(p2, data_sindy.t, data_sindy.q[:,2], markershape=:xcross, label = "particle three Iden_pos")

p4 = plot(xlabel = "Time", ylabel = "position")
plot!(p4, data_reference.t, data_reference.q[:,4], label = "particle four Ref_pos")
plot!(p4, data_sindy.t, data_sindy.q[:,4], markershape=:xcross, label = "particle four Iden_pos")

plot!(xlabel = "Time", ylabel = "y_pos", size=(1000,1000))
display(plot(p2, p4, title="Analytical vs Calculated y Positions"))

# plot momenta
p5 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p5, data_reference.t, data_reference.q[:,5], label = "particle one Ref_mom")
plot!(p5, data_sindy.t, data_sindy.q[:,5], markershape=:xcross, label = "particle one Iden_mom")

p7 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p7, data_reference.t, data_reference.q[:,7], label = "particle two Ref_mom")
plot!(p7, data_sindy.t, data_sindy.q[:,7], markershape=:xcross, label = "particle two Iden_mom")


plot!(xlabel = "Time", ylabel = "x_mom", size=(1000,1000))
display(plot(p5, p7, title="Analytical vs Calculated x Momenta"))

p6 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p6, data_reference.t, data_reference.q[:,6], label = "particle three Ref_mom")
plot!(p6, data_sindy.t, data_sindy.q[:,6], markershape=:xcross, label = "particle three Iden_mom")

p8 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p8, data_reference.t, data_reference.q[:,8], label = "particle four Ref_mom")
plot!(p8, data_sindy.t, data_sindy.q[:,8], markershape=:xcross, label = "particle four Iden_mom")

plot!(xlabel = "Time", ylabel = "y_mom", size=(1000,1000))
display(plot(p6, p8, title="Analytical vs Calculated y Momenta"))