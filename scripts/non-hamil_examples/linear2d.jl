
using DifferentialEquations
using Distributions
using ODE
using Plots
using Random
using SparseIdentification

gr()


# --------------------
# Setup
# --------------------

# generate basis
#  - search space up to fifth order polynomials
#  - no trigonometric functions
basis = CompoundBasis(polyorder = 5, trigonometric = 0)

# initial data
x₀ = [2., 0.]

# 2D system 
nd = length(x₀)

# vector field
const A = [-0.1  2.0
           -2.0 -0.1]

rhs(xᵢₙ,p,t) = A*xᵢₙ
# rhs(dx,t,xᵢₙ,p) = rhs(xᵢₙ,p,t)

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# x = Array(data)
# stored as matrix with dims [nd,ntime]

num_samp = 15

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output
# s depend on size of nd (total dims), 4 in the case here so we use samp_range x samp_range x samp_range x samp_range
s = collect(Iterators.product(fill(samp_range, nd)...))

# compute vector field from x state values
x = [collect(s[i]) for i in eachindex(s)]

x = hcat(x...)

# compute vector field from x state values at each timestep
# stored as matrix with dims [nd,ntime]
ẋ = zero(x)
for i in axes(ẋ,2)
    ẋ[:,i] .= A*x[:,i]
end

# choose SINDy method
method = SINDy(lambda = 0.05, noise_level = 0.0, coeff = 0.52, batch_size = 18)

# add noise to ẋ
ẋnoisy = ẋ .+ method.noise_level .* randn(size(ẋ))

# collect training data
tdata = TrainingData(Float32.(x), Float32.(ẋnoisy))

# println("x = ", tdata.x)
# println("ẋ = ", tdata.ẋ)


# ----------------------------------------
# Identify SINDy Vector Field
# ----------------------------------------

println("Computing Vector Field...")

# compute vector field
# vectorfield = VectorField(method, basis, tdata)
# vectorfield = VectorField(method, basis, data; solver = OptimSolver())
vectorfield, model = VectorField(method, basis, tdata, solver = NNSolver())


# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

tstep = 0.01
tspan = (0.0, 25.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

# prob_reference = GeometricIntegrators.ODEProblem((dx, t, x, params) -> rhs(dx,t,x), tspan, tstep, x₀)
# data_reference = GeometricIntegrators.integrate(prob_reference, Gauss(1))

prob = ODEProblem(rhs, x₀, tspan)
data = ODE.solve(prob, abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)

#----------------------------------------
# Note: Use Rx₀ instead of x₀ in prob_approx if just x₀ doesn't work
R = x₀ / model[1].W(x₀)
#----------------------------------------

prob_approx = ODEProblem(vectorfield, x₀, tspan)
# prob_approx = ODEProblem(vectorfield, x₀, tspan)
xid = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange) 


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

# en_data = [model[1].W(x) for x in (data.u)]
# en_data = hcat(en_data...)
# en_xid = [model[1].W(x) for x in (xid.u)]
# en_xid = hcat(en_xid...)

p1 = plot()
plot!(p1, data.t, data[1,:], label = "Data")
plot!(p1, xid.t, xid[1,:], label = "Identified")

p2 = plot()
plot!(p2, data.t, data[2,:], label = "Data")
plot!(p2, xid.t, xid[2,:], label = "Identified")

display(plot(p1, p2))
savefig("linear2d.png")

p3 = plot(data[1,:], data[2,:], label="true")
p3 = scatter!(xid[1,:], xid[2,:], label="approx", linestyle =:dash, mc=:red, ms=2, ma=0.5, xlabel ="X1", ylabel="X2")
display(plot(p3, show = true, reuse = false))
savefig("linear2d_fig2.png")