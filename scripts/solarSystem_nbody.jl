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


using Distributions
using GeometricIntegrators
using Plots
using Random
using SparseIdentification
using Optim
using Symbolics

include("solarsystem.jl") 

gr()


# --------------------
# Setup
# --------------------

println("Setting up...")

# (q₁,q₂,q₃,q₄,p₁,p₂,p₃,p₄) system, sun earth system, each with 2 dims positions and 2 dims momenta
const nd = 8

z = get_z_vector(Int(nd/2))

function calculate_differences(z)
    n = length(z)
    differences = []

    for i in 1:n÷2
        for j in i:n÷2
            if i != j
                # push!(differences, z[i] - z[j])
                push!(differences, (z[i] - z[j])^2)
            end
        end
    end

    return differences
end

mom_power = primal_power_basis(z[Int(nd/2)+1:end], 2)
basis = primal_operator_basis(calculate_differences(z), *)
basis = primal_operator_basis(basis, +)
basis = basis.^(1/2)
basis = get_basis_set(basis, mom_power)


# Get states information of earth and sun
earth = solar_system[:earth]
sun = solar_system[:sun]

# mass of each planet
m₁ = earth.m #* 1e24 # [kg]
m₂ = sun.m #* 1e24  # [kg]

# named tuple
m = (m₁ = m₁, m₂ = m₂)

# gravitational constant
# G = 6.6743e-11 # m³kg-¹s-²
G = 9.983431049193709e8  # km³(10^24 kg)⁻¹days⁻²

# Analytical Gradient
function gradient_analytical!(dx,x,m,t) 
    q = x[1:4]
    p = x[5:8]
    m₁ = m.m₁
    m₂ = m.m₂

    dx .= [p[1]./m₁; p[2]./m₁; 0; 0;
          -G .* (m₁ .* m₂ .* (q[1] - q[3]) ./ (abs(q[1] - q[3]).^3)); 
          -G .* (m₁ .* m₂ .* (q[2] - q[4]) ./ (abs(q[2] - q[4]).^3)); 
          0; 0;]
end

# Initial conditions
q₀ = [earth.x[1:2]; sun.x[1:2];] #.* 1e3 #km to m
p₀ = [earth.v[1:2] .* m[1]; sun.v[1:2] .* m[2];] #.* 1e3 ./ (3600.0 .* 24.0) #km/d to m/s
x₀ = [q₀; p₀]

tstep = 250
tspan = (0.0, 1e6)
trange = range(tspan[begin], step = tstep, stop = tspan[end])


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob_reference = ODEProblem((dx, t, x, params) -> gradient_analytical!(dx, x, params, t), tspan, tstep, x₀, parameters = m)
data_reference = integrate(prob_reference, Gauss(2))
x_ref = data_reference.q

# choose SINDy method
method = HamiltonianSINDy(basis, gradient_analytical!, z, λ = 5e-6, noise_level = 0.00)

# make ẋ reference data 
t = 0.0
dx = zeros(nd)
ẋ_ref = [gradient_analytical!(copy(dx),_x, m, t) for _x in x_ref]


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
@time vector_field = VectorField(method, tdata, solver = BFGS()) 

println(vector_field.coefficients)


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

tstep = 1e3
tspan = (0.0, 1e6)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

prob_reference = ODEProblem((dx, t, x, params) -> gradient_analytical!(dx, x, params, t), tspan, tstep, x₀, parameters = m)
data_reference = integrate(prob_reference, Gauss(2))

prob_sindy = ODEProblem((dx, t, x, params) -> vector_field(dx, x, params, t), tspan, tstep, x₀, parameters = m)
data_sindy = integrate(prob_sindy, Gauss(2))


# Sun and Earth plots
# plot positions
p1 = plot()
plot!(p1, data_reference.t, data_reference.q[:,1], label = "EarthRef xPos")
plot!(p1, data_sindy.t, data_sindy.q[:,1], label = "EarthId xPos", xlabel = "Time", ylabel = "X Position")

p3 = plot()
plot!(p3, data_reference.t, data_reference.q[:,3], label = "SunRef xPos")
plot!(p3, data_sindy.t, data_sindy.q[:,3], label = "SunId xPos", xlabel = "Time", ylabel = "X Position")

title = plot(title = "True vs Predicted X Positions", grid = false, showaxis = false, bottom_margin = -50Plots.px)
display(plot(title, p1, p3, layout = @layout([A{0.1h}; [B C]]), size=(850, 600), show=true, reuse=false, linewidth = 1.5))
savefig("solarXPos.png")

p2 = plot()
plot!(p2, data_reference.t, data_reference.q[:,2], label = "EarthRef yPos")
plot!(p2, data_sindy.t, data_sindy.q[:,2], label = "EarthId yPos", xlabel = "Time", ylabel = "Y Position")

p4 = plot()
plot!(p4, data_reference.t, data_reference.q[:,4], label = "SunRef yPos")
plot!(p4, data_sindy.t, data_sindy.q[:,4], label = "SunId yPos", xlabel = "Time", ylabel = "Y Position")

title = plot(title = "True vs Predicted Y Positions", grid = false, showaxis = false, bottom_margin = -50Plots.px)
display(plot(title, p2, p4, layout = @layout([A{0.1h}; [B C]]), size=(850, 600), show=true, reuse=false, linewidth = 1.5))
savefig("solarYPos.png")

# plot momenta
p5 = plot()
plot!(p5, data_reference.t, data_reference.q[:,5], label = "EarthRef xMom")
plot!(p5, data_sindy.t, data_sindy.q[:,5], label = "EarthId xMom", xlabel = "Time", ylabel = "X momentum")

p7 = plot()
plot!(p7, data_reference.t, data_reference.q[:,7], label = "SunRef xMom")
plot!(p7, data_sindy.t, data_sindy.q[:,7], label = "SunId xMom", xlabel = "Time", ylabel = "X momentum")

title = plot(title = "True vs Predicted X Momenta", grid = false, showaxis = false, bottom_margin = -50Plots.px)
display(plot(title, p5, p7, layout = @layout([A{0.1h}; [B C]]), size=(850, 600), show=true, reuse=false, linewidth = 1.5))
savefig("solarXMom.png")

p6 = plot()
plot!(p6, data_reference.t, data_reference.q[:,6], label = "EarthRef yMom")
plot!(p6, data_sindy.t, data_sindy.q[:,6], label = "EarthId yMom", xlabel = "Time", ylabel = "Y momentum")

p8 = plot()
plot!(p8, data_reference.t, data_reference.q[:,8], label = "SunRef yMom")
plot!(p8, data_sindy.t, data_sindy.q[:,8], label = "SunId yMom", xlabel = "Time", ylabel = "Y momentum")

title = plot(title = "True vs Predicted Y Momenta", grid = false, showaxis = false, bottom_margin = -50Plots.px)
display(plot(title, p6, p8, layout = @layout([A{0.1h}; [B C]]), size=(850, 600), show=true, reuse=false, linewidth = 1.5))
savefig("solarYMom.png")

# save coefficients to file
using DelimitedFiles
solar_system = "solar_system_files"
if !isdir(solar_system)
    mkdir(solar_system)
end
solar_system_file = joinpath(solar_system, "solar_system_coeffs.csv")
solar_system_arr = []
push!(solar_system_arr, vector_field.coefficients)
writedlm(solar_system_file, solar_system_arr, ',')




