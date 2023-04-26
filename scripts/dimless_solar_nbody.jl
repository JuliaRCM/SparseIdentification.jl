# This script solves the 2D n-body hamiltonian problem for sun and earth

# The problem is made dimensionless in this example to test if that improves the results

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

# (q₁,q₂,q₃,q₄,p₁,p₂,p₃,p₄) system, sun earth system, each with 2 dims positions and 2 dims momenta
const nd = 8

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 2

# max or min power of state difference basis for function library to explore
const diffs_power = -2

# Get states information of earth and sun
earth = solar_system[:earth]
sun = solar_system[:sun]

# gravitational constant
# G = 6.6743e-11 # m³kg-¹s-²
G = 9.983431049193709e8  # km³(10^24 kg)⁻¹days⁻²

# characteristic length given by initial separation between earth and sun
relative_L = sum((earth.x[1:2] .- sun.x[1:2]).^2)

# dimless positions
q₁ = earth.x[1:2] ./ relative_L
q₂ = sun.x[1:2] ./ relative_L

# scaled Time
τ = sqrt(relative_L^3/(G*(earth.m + sun.m)))

# scaled mass
μ₁ = earth.m/(earth.m + sun.m)
μ₂ = sun.m/(earth.m + sun.m)

# named tuple
μ = (μ₁ = μ₁, μ₂ = μ₂, τ = τ)

# characteristic velocity: initial relative velocity
relative_V = sum((earth.v[1:2] .- sun.v[1:2]).^2)

# dimless velocity
v₁ = μ₁ .* earth.v[1:2] ./ relative_V
v₂ = μ₂ .* sun.v[1:2] ./ relative_V

# Analytical Gradient
function rel_gradient_analytical!(dx,x,μ,t) 
    q = x[1:4]
    p = x[5:8]
    μ₁ = μ.μ₁
    μ₂ = μ.μ₂
    τ = μ.τ

    dx .= [p[1]./μ₁./τ; p[2]./μ₁./τ; 0; 0;
          - (q[1] - q[3]) ./ (abs(q[1] - q[3]).^3) ./ τ; 
          - (q[2] - q[4]) ./ (abs(q[2] - q[4]).^3) ./ τ; 
          0; 0;]
end

# Initial conditions
q₀ = [q₁; q₂]
p₀ = [μ₁ .* v₁; μ₂ .* v₂]
x₀ = [q₀; p₀]

t_step = 5000.0
t_span = (0.0, 1e7)
trange = range(t_span[begin], step = t_step, stop = t_span[end])


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob_reference = ODEProblem((dx, t, x, params) -> rel_gradient_analytical!(dx, x, params, t), t_span, t_step, x₀, parameters = μ)
data_reference = integrate(prob_reference, Gauss(2))
x_ref = data_reference.q

# choose SINDy method
method = HamiltonianSINDy(λ = 5e-7, noise_level = 0.00, polyorder = polyorder, diffs_power = diffs_power)

# make ẋ reference data 
t = 0.0
dx = zeros(nd)
ẋ_ref = [rel_gradient_analytical!(copy(dx),_x, μ, t) for _x in x_ref]


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

t_step = 1e5
t_span = (0.0, 1e6)
trange = range(t_span[begin], step = t_step, stop = t_span[end])

prob_reference = ODEProblem((dx, t, x, params) -> rel_gradient_analytical!(dx, x, params, t), t_span, t_step, x₀, parameters = μ)
data_reference = integrate(prob_reference, Gauss(2))

prob_sindy = ODEProblem((dx, t, x, params) -> vectorfield(dx, x, params, t), t_span, t_step, x[1])
data_sindy = integrate(prob_sindy, Gauss(2))


# Sun and Earth plots
# plot positions
p1 = plot(xlabel = "Time", ylabel = "position")
plot!(p1, data_reference.t, data_reference.q[:,1], label = "EarthRef xPos")
plot!(p1, data_sindy.t, data_sindy.q[:,1], label = "EarthId xPos")

p3 = plot(xlabel = "Time", ylabel = "position")
plot!(p3, data_reference.t, data_reference.q[:,3], label = "SunRef xPos")
plot!(p3, data_sindy.t, data_sindy.q[:,3], label = "SunId xPos")

plot!(xlabel = "Time", ylabel = "x_pos", size=(1000,1000))
display(plot(p1, p3, title="Analytical vs Calculated x Positions"))

p2 = plot(xlabel = "Time", ylabel = "position")
plot!(p2, data_reference.t, data_reference.q[:,2], label = "EarthRef yPos")
plot!(p2, data_sindy.t, data_sindy.q[:,2], label = "EarthId yPos")

p4 = plot(xlabel = "Time", ylabel = "position")
plot!(p4, data_reference.t, data_reference.q[:,4], label = "SunRef yPos")
plot!(p4, data_sindy.t, data_sindy.q[:,4], label = "SunId yPos")

plot!(xlabel = "Time", ylabel = "y_pos", size=(1000,1000))
display(plot(p2, p4, title="Analytical vs Calculated y Positions"))

# plot momenta
p5 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p5, data_reference.t, data_reference.q[:,5], label = "EarthRef xMom")
plot!(p5, data_sindy.t, data_sindy.q[:,5], label = "EarthId xMom")

p7 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p7, data_reference.t, data_reference.q[:,7], label = "SunRef xMom")
plot!(p7, data_sindy.t, data_sindy.q[:,7], label = "SunId xMom")


plot!(xlabel = "Time", ylabel = "x_mom", size=(1000,1000))
display(plot(p5, p7, title="Analytical vs Calculated x Momenta"))

p6 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p6, data_reference.t, data_reference.q[:,6], label = "EarthRef yMom")
plot!(p6, data_sindy.t, data_sindy.q[:,6], label = "EarthId yMom")

p8 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p8, data_reference.t, data_reference.q[:,8], label = "SunRef yMom")
plot!(p8, data_sindy.t, data_sindy.q[:,8], label = "SunId yMom")

plot!(xlabel = "Time", ylabel = "y_mom", size=(1000,1000))
display(plot(p6, p8, title="Analytical vs Calculated y Momenta"))