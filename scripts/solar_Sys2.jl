# This script solves the 2D n-body problem for a number of planets

###############################################################################
##################################THEORY#######################################
###############################################################################

# Reference for equations: 
# Easton, Robert W. "Introduction to Hamiltonian dynamical systems and the N-body problem (KR Meyer and GR Hall)." SIAM Review 35, no. 4 (1993): 659-659.

# Q_i and p_i are the i-th coordinate and momentum, 
# respectively, and H is the Hamiltonian of the system.
# where m_i and p_i are the mass and momentum of the 
# i-th particle, G is the gravitational constant, r_ij 
# is the distance between the i-th and j-th particles, 
# and the summations are over all pairs of particles
# H = Σ (1/2m_i) * p_i^2 - G * Σ(m_i*m_j / ||r_ij||) {1<=i<j<= N}

# The distance between the i-th and j-th particles:
# r_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2)

# Time evolution of the positions and momenta of the particles is given by the Hamilton equations
# dq_i/dt = ∂H/∂p_i = (1/m_i) * p_i
# dp_i/dt = -∂H/∂q_i = -G * Σ(m_i * m_j * (x_i - x_j) / r_ij^3) {the sum goes from j=1 to j=N and i≠j}


using DifferentialEquations
using Distributions
using ODE
using Plots
using Random
using SparseIdentification
using Optim
using GeometricIntegrators

include("solarsystem.jl") 

gr()


# --------------------
# Setup
# --------------------

println("Setting up...")

# (q₁,q₂,q₃,q₄,p₁,p₂,p₃,p₄) system, sun earth system, each with 2 dims positions and 2 dims momenta
const nd = 8

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 2

# max or min power of state difference basis for function library to explore
const diffs_power = -2

# Get states information of 3 planets
earth = solar_system[:earth]
sun = solar_system[:sun]

# mass of each planet
m₁ = earth.m #* 1e24 # [kg]
m₂ = sun.m #* 1e24  # [kg]

m = [m₁, m₂]

# gravitational constant
# G = 6.6743e-11 # m³kg-¹s-²
G = 9.983431049193709e8 # km³(10^24 kg)⁻¹days⁻²

function grad_pos_ana!(dq,q,p,m,t) 
        dq .= [p[1]./m[1]; p[2]./m[1];
              p[3]./m[2]; p[4]./m[2];]
end

function grad_mom_ana!(dp,q,p,m,t) 
        dp .= [-G .* (m[1] .* m[2] .* (q[1] - q[3]) ./ (abs(q[1] - q[3]).^3)); 
            -G .* (m[1] .* m[2] .* (q[2] - q[4]) ./ (abs(q[2] - q[4]).^3)); 
            -G .* (m[2] .* m[1] .* (q[3] - q[1]) ./ (abs(q[3] - q[1]).^3));
            -G .* (m[2] .* m[1] .* (q[4] - q[2]) ./ (abs(q[4] - q[2]).^3));] 
end

# Initial conditions
q₀ = [earth.x[1:2]; sun.x[1:2];] #.* 1e3 #km to m      sun.x[1:2];
p₀ = [earth.v[1:2] .* m[1]; sun.v[1:2] .* m[2];] #.* 1e3 ./ (3600.0 .* 24.0) #km/d to m/s    sun.v[1:2] .* m[3];


tstep = 5000
tspan = (0.0, 1e7)
trange = range(tspan[begin], step = tstep, stop = tspan[end])


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob_reference = DynamicalODEProblem(grad_pos_ana!, grad_mom_ana!, q₀, p₀, tspan, m)
data_reference = ODE.solve(prob_reference, KahanLi6(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)
x_ref = data_reference.u


# choose SINDy method
method_params = HamiltonianSINDy(λ = 5e-7, noise_level = 0.00, polyorder = polyorder, diffs_power = diffs_power)

# add noise to data
y_noisy = [_x .+ method_params.noise_level .* randn(size(_x)) for _x in x_ref]

# wrapper function to make ẋ reference data 
function grad_ana(x,m)
    # dummy values
    dq = zeros(4) 
    dp = zeros(4) 
    t = 0
    p = zeros(4) 
    q = zeros(4) 

    ẋ_ref_pos = grad_pos_ana!(dq, q, x[5:8], m, t)
    ẋ_ref_mom = grad_mom_ana!(dp, x[1:4], p, m, t)

    return [ẋ_ref_pos; ẋ_ref_mom]
end
ẋ_ref = [grad_ana(_x, m) for _x in x_ref]

# Flatten data for TrainingData struct
x = Float64[]
x = [vcat(x, vec(_x)) for _x in x_ref]

ẋ = Float64[]
ẋ = [vcat(ẋ, vec(_ẋ)) for _ẋ in ẋ_ref]

y = Float64[]
y = [vcat(y, vec(_y)) for _y in y_noisy]


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# collect training data
tdata = TrainingData(x, ẋ, y)

# compute vector field
vectorfield = VectorField(method_params, tdata, solver = BFGS()) 

println(vectorfield.coefficients)


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

tstep = 100000
tspan = (0.0, 1e6)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

prob_reference = DynamicalODEProblem(grad_pos_ana!, grad_mom_ana!, q₀, p₀, tspan, m)
data_reference = ODE.solve(prob_reference, KahanLi6(), abstol=1e-7, reltol=1e-7, saveat = trange, tstops = trange)

prob_sindy = GeometricIntegrators.ODEProblem((dx, t, x, params) -> vectorfield(dx, x, params, t), tspan, tstep, x[1])
data_sindy = integrate(prob_sindy, QinZhang())


# Sun and Earth plots
# plot positions
p1 = plot(xlabel = "Time", ylabel = "position")
plot!(p1, data_reference.t, data_reference[1,:], label = "EarthRef xPos")
plot!(p1, data_sindy.t, data_sindy.q[:,1], label = "EarthId xPos")

p3 = plot(xlabel = "Time", ylabel = "position")
plot!(p3, data_reference.t, data_reference[3,:], label = "SunRef xPos")
plot!(p3, data_sindy.t, data_sindy.q[:,3], label = "SunId xPos")

plot!(xlabel = "Time", ylabel = "x_pos", size=(1000,1000))
display(plot(p1, p3, title="Analytical vs Calculated x Positions"))

p2 = plot(xlabel = "Time", ylabel = "position")
plot!(p2, data_reference.t, data_reference[2,:], label = "EarthRef yPos")
plot!(p2, data_sindy.t, data_sindy.q[:,2], label = "EarthId yPos")

p4 = plot(xlabel = "Time", ylabel = "position")
plot!(p4, data_reference.t, data_reference[4,:], label = "SunRef yPos")
plot!(p4, data_sindy.t, data_sindy.q[:,4], label = "SunId yPos")

plot!(xlabel = "Time", ylabel = "y_pos", size=(1000,1000))
display(plot(p2, p4, title="Analytical vs Calculated y Positions"))

# plot momenta
p5 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p5, data_reference.t, data_reference[5,:], label = "EarthRef xMom")
plot!(p5, data_sindy.t, data_sindy.q[:,5], label = "EarthId xMom")

p7 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p7, data_reference.t, data_reference[7,:], label = "SunRef xMom")
plot!(p7, data_sindy.t, data_sindy.q[:,7], label = "SunId xMom")


plot!(xlabel = "Time", ylabel = "x_mom", size=(1000,1000))
display(plot(p5, p7, title="Analytical vs Calculated x Momenta"))

p6 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p6, data_reference.t, data_reference[6,:], label = "EarthRef yMom")
plot!(p6, data_sindy.t, data_sindy.q[:,6], label = "EarthId yMom")

p8 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p8, data_reference.t, data_reference[8,:], label = "SunRef yMom")
plot!(p8, data_sindy.t, data_sindy.q[:,8], label = "SunId yMom")

plot!(xlabel = "Time", ylabel = "y_mom", size=(1000,1000))
display(plot(p6, p8, title="Analytical vs Calculated y Momenta"))