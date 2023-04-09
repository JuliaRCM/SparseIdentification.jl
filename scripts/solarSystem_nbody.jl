# This script solves the 2D n-body problem for a number of planets

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

include("solarsystem.jl") 

gr()


# --------------------
# Setup
# --------------------

println("Setting up...")

# (q₁,q₂,q₃,q₄,q₅,q₆,p₁,p₂,p₃,p₄,p₅,p₆) system, 3 planets and each with 2 dims positions and 2 dims momenta
const nd = 8

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 2

# maximum wave number of trig basis for function library to explore
const trig_wave_num = 0

# maximum wave number of state difference basis for function library to explore
const diffs_power = 2

# Get states information of 3 planets
venus = solar_system[:venus]
earth = solar_system[:earth]
sun = solar_system[:sun]

# mass of each planet
m₂ = venus.m * 1e24 # [kg]
m₁ = earth.m * 1e24 # [kg]
m₃ = sun.m * 1e24  # [kg]
# m = [m₁, m₂, m₃] 
m = [m₁, m₂]

# gravitational constant
G = 6.6743e-11 # m³kg-¹s-²

# Gradient function of the 2D hamiltonian
dq_i_dt(mᵢ,pᵢ) =  pᵢ./mᵢ
dp_i_dt(mᵢ,mⱼ,xᵢ,xⱼ) = -G .* mᵢ .* mⱼ .* (xᵢ - xⱼ) ./ (abs(xᵢ - xⱼ).^3)

# momentum gradient function of the 2D hamiltonian
dp_i_dt(mᵢ,mⱼ,xᵢ,xⱼ) = -G * mᵢ .* mⱼ .* (xᵢ - xⱼ)./(abs(xᵢ - xⱼ).^3)

function grad_pos_ana!(dq,q,p,m,t) 
        dq .= [p[1]./m[1]; p[2]./m[1];
              p[3]./m[2]; p[4]./m[2];]
              # p[5]./m[3]; p[6]./m[3];]
end

function grad_mom_ana!(dp,q,p,m,t) 
        dp .= [-G .* (m[1] .* m[2] .* (q[1] - q[3]) ./ (abs(q[1] - q[3]).^3)); # .+ m[1] .* m[3] .* (q[1] - q[5]) ./ (abs(q[1] - q[5]).^3));
            -G .* (m[1] .* m[2] .* (q[2] - q[4]) ./ (abs(q[2] - q[4]).^3)); #.+ m[1] .* m[3] .* (q[2] - q[6]) ./ (abs(q[2] - q[6]).^3));
            -G .* (m[2] .* m[1] .* (q[3] - q[1]) ./ (abs(q[3] - q[1]).^3)); #.+ m[2] .* m[3] .* (q[3] - q[5]) ./ (abs(q[3] - q[5]).^3));
            -G .* (m[2] .* m[1] .* (q[4] - q[2]) ./ (abs(q[4] - q[2]).^3));] #.+ m[2] .* m[3] .* (q[4] - q[6]) ./ (abs(q[4] - q[6]).^3));]
           # -G .* (m[3] .* m[1] .* (q[5] - q[1]) ./ (abs(q[5] - q[1]).^3) .+ m[3] .* m[2] .* (q[5] - q[3]) ./ (abs(q[5] - q[3]).^3));
           # -G .* (m[3] .* m[1] .* (q[6] - q[2]) ./ (abs(q[6] - q[2]).^3) .+ m[3] .* m[2] .* (q[6] - q[4]) ./ (abs(q[6] - q[4]).^3));]
end        

# Initial conditions
q₀ = [venus.x[1:2]; earth.x[1:2];] .* 1e3 #km to m      sun.x[1:2];
p₀ = [venus.v[1:2] .* m[1]; earth.v[1:2] .* m[2];] .* 1e3 ./ (3600.0 .* 24.0) #km/d to m/s    sun.v[1:2] .* m[3];

centering_q₀ = maximum(q₀)
centering_p₀ = maximum(p₀)
q₀ = q₀ ./ centering_q₀
p₀ = p₀ ./ centering_p₀

tstep = 1.0
tspan = (0.0, 1e4)
trange = range(tspan[begin], step = tstep, stop = tspan[end])


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob_reference = DynamicalODEProblem(grad_pos_ana!, grad_mom_ana!, q₀, p₀, tspan, m)
data_reference = ODE.solve(prob_reference, KahanLi6(), abstol=1e-7, reltol=1e-7, saveat = trange, tstops = trange)
x_ref = data_reference.u


# choose SINDy method
# TODO: not sure if statement about λ and noise_level below is true anymore, because we need a high noise_level to data here but not necessarily the same value for λ 
# (λ parameter must be close to noise value so that only coeffs with value around the noise are sparsified away)
# **** grad_pos_ana! is just a place in this case b/c we generate noisy data directly in the script
method_params = HamiltonianSINDy(grad_pos_ana!, λ = 0.05, noise_level = 0.05, integrator_timeStep = 0.05, polyorder = polyorder, trigonometric = trig_wave_num, diffs_power=diffs_power)

# add noise to data
y_noisy = [_x .+ method_params.noise_level .* randn(size(_x)) for _x in x_ref]

# wrapper function to make ẋ reference data 
function grad_ana(x,m)
    # dummy values
    dq = zeros(4) #zeros(6)
    dp = zeros(4) #zeros(6)
    t = 0
    p = zeros(4) #zeros(6)
    q = zeros(4) #zeros(6)

    ẋ_ref_pos = grad_pos_ana!(dq, q, x[5:8], m, t)#grad_pos_ana!(dq, q, x[7:12], m, t)
    ẋ_ref_mom = grad_mom_ana!(dp, x[1:4], p, m, t)#grad_mom_ana!(dp, x[1:6], p, m, t)

    return [ẋ_ref_pos; ẋ_ref_mom]
end
ẋ_ref = [grad_ana(_x, m) for _x in x_ref]

# flatten data for TrainingData struct
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
vectorfield = VectorField(method_params, tdata)

println(vectorfield.coefficients)


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

tstep = 1.0
tspan = (0.0, 1e4)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

prob_reference = DynamicalODEProblem(grad_pos_ana!, grad_mom_ana!, q₀, p₀, tspan, m)
data_reference = ODE.solve(prob_reference, KahanLi6(), abstol=1e-7, reltol=1e-7, saveat = trange, tstops = trange)

# TODO: figure out what ode problem to use here for symplectic integration
prob_sindy = ODEProblem(vectorfield, x[1], tspan)
data_sindy = ODE.solve(prob_sindy, Tsit5(), abstol=1e-7, reltol=1e-7, saveat = trange, tstops = trange) 

# plot positions
plot!(p1, data_reference.t, data_reference[1,:], label = "VenusRef xPos")
plot!(p1, data_sindy.t, data_sindy[1,:], markershape=:xcross, label = "VenusId xPos")

plot!(p3, data_reference.t, data_reference[3,:], label = "EarthRef xPos")
plot!(p3, data_sindy.t, data_sindy[3,:], markershape=:xcross, label = "EarthId xPos")

plot!(p5, data_reference.t, data_reference[5,:], label = "SunRef xPos")
plot!(p5, data_sindy.t, data_sindy[5,:], markershape=:xcross, label = "SunId xPos")

plot!(xlabel = "Time", ylabel = "x_pos", size=(1000,1000))
display(plot(p1, p3, p5, title="Analytical vs Calculated x Positions"))

plot!(p2, data_reference.t, data_reference[2,:], label = "VenusRef yPos")
plot!(p2, data_sindy.t, data_sindy[2,:], markershape=:xcross, label = "VenusId yPos")

plot!(p4, data_reference.t, data_reference[4,:], label = "EarthRef yPos")
plot!(p4, data_sindy.t, data_sindy[4,:], markershape=:xcross, label = "EarthId yPos")

plot!(p6, data_reference.t, data_reference[6,:], label = "SunRef yPos")
plot!(p6, data_sindy.t, data_sindy[6,:], markershape=:xcross, label = "SunId yPos")

plot!(xlabel = "Time", ylabel = "y_pos", size=(1000,1000))
display(plot(p2, p4, p6, title="Analytical vs Calculated y Positions"))

# plot momenta
plot!(p7, data_reference.t, data_reference[7,:], label = "VenusRef xMom")
plot!(p7, data_sindy.t, data_sindy[7,:], markershape=:xcross, label = "VenusId xMom")

plot!(p9, data_reference.t, data_reference[9,:], label = "EarthRef xMom")
plot!(p9, data_sindy.t, data_sindy[9,:], markershape=:xcross, label = "EarthId xMom")

plot!(p11, data_reference.t, data_reference[11,:], label = "SunRef xMom")
plot!(p11, data_sindy.t, data_sindy[11,:], markershape=:xcross, label = "SunId xMom")

plot!(xlabel = "Time", ylabel = "x_mom", size=(1000,1000))
display(plot(p7, p9, p11, title="Analytical vs Calculated x Momenta"))

plot!(p8, data_reference.t, data_reference[8,:], label = "VenusRef yMom")
plot!(p8, data_sindy.t, data_sindy[8,:], markershape=:xcross, label = "VenusId yMom")

plot!(p10, data_reference.t, data_reference[10,:], label = "EarthRef yMom")
plot!(p10, data_sindy.t, data_sindy[10,:], markershape=:xcross, label = "EarthId yMom")

plot!(p12, data_reference.t, data_reference[12,:], label = "SunRef yMom")
plot!(p12, data_sindy.t, data_sindy[12,:], markershape=:xcross, label = "SunId yMom")

plot!(xlabel = "Time", ylabel = "y_mom", size=(1000,1000))
display(plot(p8, p10, p12, title="Analytical vs Calculated y Momenta"))












# test ref plots
p1 = plot(xlabel = "Time", ylabel = "position")
plot!(p1, data_reference.t, data_reference[1,:], label = "Venus x-position")
p2 = plot(xlabel = "Time", ylabel = "position")
plot!(p2, data_reference.t, data_reference[2,:], label = "Venus y-position")
p3 = plot(xlabel = "Time", ylabel = "position")
plot!(p3, data_reference.t, data_reference[3,:], label = "Earth x-position")
p4 = plot(xlabel = "Time", ylabel = "position")
plot!(p4, data_reference.t, data_reference[4,:], label = "Earth y-position")
p5 = plot(xlabel = "Time", ylabel = "position")
plot!(p5, data_reference.t, data_reference[5,:], label = "Sun x-position")
p6 = plot(xlabel = "Time", ylabel = "position")
plot!(p6, data_reference.t, data_reference[6,:], label = "Sun y-position")
display(plot(p1, p2, p3, p4, p5, p6, title="Analytical positions"))


p7 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p7, data_reference.t, data_reference[7,:], label = "Venus x-momentum")
p8 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p8, data_reference.t, data_reference[8,:], label = "Venus y-momentum")
p9 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p9, data_reference.t, data_reference[9,:], label = "Earth x-momentum")
p10 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p10, data_reference.t, data_reference[10,:], label = "Earth y-momentum")
p11 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p11, data_reference.t, data_reference[11,:], label = "Sun x-momentum")
p12 = plot(xlabel = "Time", ylabel = "momentum")
plot!(p12, data_reference.t, data_reference[12,:], label = "Sun y-momentum")
display(plot(p7, p8, p9, p10, p11, p12, title="Analytical momentum"))
