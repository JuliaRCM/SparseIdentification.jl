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
using Plots
using Random
using SparseIdentification
using GeometricIntegrators
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

# mass of each planet
m₁ = earth.m #* 1e24 # [kg]
m₂ = sun.m #* 1e24  # [kg]

# named tuple
m = (m₁ = m₁, m₂ = m₂)

# gravitational constant
G = 6.6743e-11 # m³kg-¹s-²
# G = 9.983431049193709e8  # km³(10^24 kg)⁻¹days⁻²

function gradient_analytical!(dx,x,m,t) 
    q = x[1:4]
    p = x[5:8]
    m₁ = m[1]
    m₂ = m[2]

    dx .= [p[1]./m₁; p[2]./m₁; 0; 0;
          -G .* (m₁ .* m₂ .* (q[1] - q[3]) ./ (abs(q[1] - q[3]).^3)); 
          -G .* (m₁ .* m₂ .* (q[2] - q[4]) ./ (abs(q[2] - q[4]).^3)); 
          0; 0;]
end

# Initial conditions
q₀ = [earth.x[1:2]; 0; 0;] #.* 1e3 #km to m    
p₀ = [earth.v[1:2] .* m[1]; 0; 0;] #.* 1e3 ./ (3600.0 .* 24.0) #km/d to m/s    
x₀ = [q₀; p₀]

tstep = 5e1
tspan = (0.0, 1e7)
trange = range(tspan[begin], step = tstep, stop = tspan[end])


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob_reference = GeometricIntegrators.ODEProblem((dx, t, x, params) -> gradient_analytical!(dx, x, params, t), tspan, tstep, x₀, parameters=m)
data_reference = integrate(prob_reference, Gauss(2))
x_ref = data_reference.q

# choose SINDy method
# **** gradient_analytical! is just a place holder in this case b/c we generate noisy data directly in the script #TODO: already replaced in new version
method = HamiltonianSINDy(gradient_analytical!, λ = 5e-8, noise_level = 0.00, polyorder = polyorder, diffs_power = diffs_power)

# add noise to data
y_noisy = [_x .+ method.noise_level .* randn(size(_x)) for _x in x_ref]

# make ẋ reference data 
t = 0.0
dx = zeros(nd)
ẋ_ref = [gradient_analytical!(dx,_x, m, t) for _x in x_ref]

# Flatten data for TrainingData struct
# position data
x = Float64[]
x = [vcat(x, vec(_x)) for _x in x_ref]

# momentum data
ẋ = Float64[]
ẋ = [vcat(ẋ, vec(_ẋ)) for _ẋ in ẋ_ref]

# noisy position data 
y = Float64[]
y = [vcat(y, vec(_y)) for _y in y_noisy]


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# collect training data
tdata = TrainingData(x, ẋ, y)

# compute vector field
vectorfield = VectorField(method, tdata, solver = BFGS()) 

println(vectorfield.coefficients)


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

tstep = 1e5
tspan = (0.0, 1e6)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

prob_reference = DynamicalODEProblem(grad_pos_ana!, grad_mom_ana!, q₀, p₀, tspan, m)
data_reference = ODE.solve(prob_reference, KahanLi6(), abstol=1e-7, reltol=1e-7, saveat = trange, tstops = trange)

prob_sindy = GeometricIntegrators.ODEProblem((dx, t, x, params) -> vectorfield(dx, x, params, t), tspan, tstep, x[1])
data_sindy = integrate(prob_sindy, Gauss(2))


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


























# sun, and 2 planets plots
# plot positions
# p1 = plot(xlabel = "Time", ylabel = "position")
# plot!(p1, data_reference.t, data_reference[1,:], label = "VenusRef xPos")
# plot!(p1, data_sindy.t, data_sindy.q[:,1], label = "VenusId xPos")

# p3 = plot(xlabel = "Time", ylabel = "position")
# plot!(p3, data_reference.t, data_reference[3,:], label = "EarthRef xPos")
# plot!(p3, data_sindy.t, data_sindy.q[:,3], label = "EarthId xPos")

# p5 = plot(xlabel = "Time", ylabel = "position")
# plot!(p5, data_reference.t, data_reference[5,:], label = "SunRef xPos")
# plot!(p5, data_sindy.t, data_sindy.q[:,5], label = "SunId xPos")

# plot!(xlabel = "Time", ylabel = "x_pos", size=(1000,1000))
# display(plot(p1, p3, p5, title="Analytical vs Calculated x Positions"))

# p2 = plot(xlabel = "Time", ylabel = "position")
# plot!(p2, data_reference.t, data_reference[2,:], label = "VenusRef yPos")
# plot!(p2, data_sindy.t, data_sindy.q[:,2], label = "VenusId yPos")

# p4 = plot(xlabel = "Time", ylabel = "position")
# plot!(p4, data_reference.t, data_reference[4,:], label = "EarthRef yPos")
# plot!(p4, data_sindy.t, data_sindy.q[:,4], label = "EarthId yPos")

# p6 = plot(xlabel = "Time", ylabel = "position")
# plot!(p6, data_reference.t, data_reference[6,:], label = "SunRef yPos")
# plot!(p6, data_sindy.t, data_sindy.q[:,6], label = "SunId yPos")

# plot!(xlabel = "Time", ylabel = "y_pos", size=(1000,1000))
# display(plot(p2, p4, p6, title="Analytical vs Calculated y Positions"))

# # plot momenta
# p7 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p7, data_reference.t, data_reference[7,:], label = "VenusRef xMom")
# plot!(p7, data_sindy.t, data_sindy.q[:,7], label = "VenusId xMom")

# p9 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p9, data_reference.t, data_reference[9,:], label = "EarthRef xMom")
# plot!(p9, data_sindy.t, data_sindy.q[:,9], label = "EarthId xMom")

# p11 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p11, data_reference.t, data_reference[11,:], label = "SunRef xMom")
# plot!(p11, data_sindy.t, data_sindy.q[:,11], label = "SunId xMom")

# plot!(xlabel = "Time", ylabel = "x_mom", size=(1000,1000))
# display(plot(p7, p9, p11, title="Analytical vs Calculated x Momenta"))

# p8 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p8, data_reference.t, data_reference[8,:], label = "VenusRef yMom")
# plot!(p8, data_sindy.t, data_sindy.q[:,8], label = "VenusId yMom")

# p10 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p10, data_reference.t, data_reference[10,:], label = "EarthRef yMom")
# plot!(p10, data_sindy.t, data_sindy.q[:,10], label = "EarthId yMom")

# p12 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p12, data_reference.t, data_reference[12,:], label = "SunRef yMom")
# plot!(p12, data_sindy.t, data_sindy.q[:,12], label = "SunId yMom")

# plot!(xlabel = "Time", ylabel = "y_mom", size=(1000,1000))
# display(plot(p8, p10, p12, title="Analytical vs Calculated y Momenta"))















# # test ref plots
# p1 = plot(xlabel = "Time", ylabel = "position")
# plot!(p1, data_reference.t, data_reference[1,:], label = "Venus x-position")
# p2 = plot(xlabel = "Time", ylabel = "position")
# plot!(p2, data_reference.t, data_reference[2,:], label = "Venus y-position")
# p3 = plot(xlabel = "Time", ylabel = "position")
# plot!(p3, data_reference.t, data_reference[3,:], label = "Earth x-position")
# p4 = plot(xlabel = "Time", ylabel = "position")
# plot!(p4, data_reference.t, data_reference[4,:], label = "Earth y-position")
# p5 = plot(xlabel = "Time", ylabel = "position")
# plot!(p5, data_reference.t, data_reference[5,:], label = "Sun x-position")
# p6 = plot(xlabel = "Time", ylabel = "position")
# plot!(p6, data_reference.t, data_reference[6,:], label = "Sun y-position")
# display(plot(p1, p2, p3, p4, p5, p6, title="Analytical positions"))


# p7 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p7, data_reference.t, data_reference[7,:], label = "Venus x-momentum")
# p8 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p8, data_reference.t, data_reference[8,:], label = "Venus y-momentum")
# p9 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p9, data_reference.t, data_reference[9,:], label = "Earth x-momentum")
# p10 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p10, data_reference.t, data_reference[10,:], label = "Earth y-momentum")
# p11 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p11, data_reference.t, data_reference[11,:], label = "Sun x-momentum")
# p12 = plot(xlabel = "Time", ylabel = "momentum")
# plot!(p12, data_reference.t, data_reference[12,:], label = "Sun y-momentum")
# display(plot(p7, p8, p9, p10, p11, p12, title="Analytical momentum"))




#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################EXTRA CODE######################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

# # Gradient function of the 2D hamiltonian
# dq_i_dt(mᵢ,pᵢ) =  pᵢ./mᵢ
# dp_i_dt(mᵢ,mⱼ,xᵢ,xⱼ) = -G .* mᵢ .* mⱼ .* (xᵢ - xⱼ) ./ (abs(xᵢ - xⱼ).^3)

# # momentum gradient function of the 2D hamiltonian
# dp_i_dt(mᵢ,mⱼ,xᵢ,xⱼ) = -G * mᵢ .* mⱼ .* (xᵢ - xⱼ)./(abs(xᵢ - xⱼ).^3)

# function grad_pos_ana!(dq,q,p,m,t) 
#         dq .= [p[1]./m[1]; p[2]./m[1];
#               p[3]./m[2]; p[4]./m[2];]
#               # p[5]./m[3]; p[6]./m[3];]
# end
# function grad_pos_ana!(dq,q,p,m,t) 
#     dq .= [p[1]./m[1]; p[2]./m[1];
#           0; 0;]
#           # p[5]./m[3]; p[6]./m[3];]
# end

# function grad_mom_ana!(dp,q,p,m,t) 
#         dp .= [-G .* (m[1] .* m[2] .* (q[1] - q[3]) ./ (abs(q[1] - q[3]).^3)); # .+ m[1] .* m[3] .* (q[1] - q[5]) ./ (abs(q[1] - q[5]).^3));
#             -G .* (m[1] .* m[2] .* (q[2] - q[4]) ./ (abs(q[2] - q[4]).^3)); #.+ m[1] .* m[3] .* (q[2] - q[6]) ./ (abs(q[2] - q[6]).^3));
#             -G .* (m[2] .* m[1] .* (q[3] - q[1]) ./ (abs(q[3] - q[1]).^3)); #.+ m[2] .* m[3] .* (q[3] - q[5]) ./ (abs(q[3] - q[5]).^3));
#             -G .* (m[2] .* m[1] .* (q[4] - q[2]) ./ (abs(q[4] - q[2]).^3));] #.+ m[2] .* m[3] .* (q[4] - q[6]) ./ (abs(q[4] - q[6]).^3));]
#            # -G .* (m[3] .* m[1] .* (q[5] - q[1]) ./ (abs(q[5] - q[1]).^3) .+ m[3] .* m[2] .* (q[5] - q[3]) ./ (abs(q[5] - q[3]).^3));
#            # -G .* (m[3] .* m[1] .* (q[6] - q[2]) ./ (abs(q[6] - q[2]).^3) .+ m[3] .* m[2] .* (q[6] - q[4]) ./ (abs(q[6] - q[4]).^3));]
# end       

# function grad_mom_ana!(dp,q,p,m,t) 
#     dp .= [-G .* (m[1] .* m[2] .* (q[1] - q[3]) ./ (abs(q[1] - q[3]).^3)); 
#            -G .* (m[1] .* m[2] .* (q[2] - q[4]) ./ (abs(q[2] - q[4]).^3)); 
#            0; 
#            0;]
# end

# prob_reference = DynamicalODEProblem(grad_pos_ana!, grad_mom_ana!, q₀, p₀, tspan, m)
# data_reference = ODE.solve(prob_reference, KahanLi6(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)
# x_ref = data_reference.u



# center the data so it is not so big and the SINDy algorithm can work better with things such as noise_level
# centering_q₀ = maximum(q₀)
# centering_p₀ = maximum(p₀)
# centering_m = maximum(m)
# q₀ = q₀ ./ centering_q₀
# p₀ = p₀ ./ centering_p₀
# m = m ./ centering_m


# function grad_ana(x,m)
#     # dummy values
#     dq = zeros(4) #zeros(6)
#     dp = zeros(4) #zeros(6)
#     t = 0
#     p = zeros(4) #zeros(6)
#     q = zeros(4) #zeros(6)

#     ẋ_ref_pos = grad_pos_ana!(dq, q, x[5:8], m, t)#grad_pos_ana!(dq, q, x[7:12], m, t)
#     ẋ_ref_mom = grad_mom_ana!(dp, x[1:4], p, m, t)#grad_mom_ana!(dp, x[1:6], p, m, t)

#     return [ẋ_ref_pos; ẋ_ref_mom]
# end







#############################################################################################################################################################################
#############################################################################################################################################################################
#############################################################################################################################################################################
# using RuntimeGeneratedFunctions
# using Symbolics
# RuntimeGeneratedFunctions.init(@__MODULE__)
# include("../src/methods/hamiltonian.jl")
# export _prod, calculate_nparams
# # Change to a matrix
# testẋ = hcat(ẋ...)

# # dimension of system
# d = size(tdata.x[begin], 1) ÷ 2

# # returns function that builds hamiltonian gradient through symbolics
# fθ = ΔH_func_builder(d, method_params.polyorder, method_params.trignometric, 
# method_params.diffs_power, method_params.trig_state_diffs)


# # dimension of system
# first_dim = size(x[begin],1)

# # binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
# nparam = calculate_nparams(first_dim, method_params.polyorder, method_params.trignometric, method_params.diffs_power, method_params.trig_state_diffs)

# # coeffs initialized to a vector of zeros b/c easier to optimize ones for our case
# coeffs = ones(nparam)

# f = zeros(eltype(coeffs), size(x[begin],1))

# lib_grads = zeros(size(x_ref,1),size(x[begin],1))

# # Change to a matrix
# testx = hcat(x...)

# for i in axes(testx,2)
#     fθ(f, testx[:,i], coeffs)
#     lib_grads[i,:] .= f
# end

# # initial guess: least-squares
# Ξ = lib_grads \ testx'

# for _ in 1:10
#     # find coefficients below λ threshold
#     smallinds = abs.(Ξ) .< method_params.λ

#     # check if there are any small coefficients != 0 left
#     all(Ξ[smallinds] .== 0) && break

#     # set all small coefficients to zero
#     Ξ[smallinds] .= 0

#     # Regress dynamics onto remaining terms to find sparse Ξ
#     for ind in axes(ẋ,1)
#         biginds = .~(smallinds[:,ind])
#         Ξ[biginds,ind] .= lib_grads[:,biginds] \ testx[biginds,:]'
#     end
# end
# Ξ
#############################################################################################################################################################################
#############################################################################################################################################################################
#############################################################################################################################################################################
