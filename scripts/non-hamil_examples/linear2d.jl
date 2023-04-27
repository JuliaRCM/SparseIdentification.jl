
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

tstep = 0.01
# tspan = (0.0,0.05)
tspan = (0.0,25.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

# vector field
const A = [-0.1  2.0
           -2.0 -0.1]

rhs(x,p,t) = A*x


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob = ODEProblem(rhs, x₀, tspan)
data = ODE.solve(prob, abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)
x = Array(data)
# stored as matrix with dims [nd,ntime]

# compute vector field from x state values at each timestep
# stored as matrix with dims [nd,ntime]
ẋ = zero(x)
for i in axes(ẋ,2)
    ẋ[:,i] .= A*x[:,i]
end


# collect training data
tdata = TrainingData(x, ẋ)

# println("x = ", tdata.x)
# println("ẋ = ", tdata.ẋ)


# ----------------------------------------
# Identify SINDy Vector Field
# ----------------------------------------

# choose SINDy method
method = SINDy(lambda = 0.05, noise_level = 0.05)

# compute vector field
vectorfield = VectorField(method, basis, tdata)
# vectorfield = VectorField(method, basis, data; solver = OptimSolver())

#println(vectorfield.coefficients)

#println("   maximum(vectorfield.coefficients) = ", maximum(vectorfield.coefficients))


# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

prob_approx = ODEProblem(vectorfield, x₀, tspan)
xid = ODE.solve(prob_approx, ode4(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange) 


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

p1 = plot()
plot!(p1, data.t, data[1,:], label = "Data")
plot!(p1, xid.t, xid[1,:], label = "Identified")

p2 = plot()
plot!(p2, data.t, data[2,:], label = "Data")
plot!(p2, xid.t, xid[2,:], label = "Identified")

plot(p1, p2)
savefig("linear2d.png")

p3 = plot(data[1,:], data[2,:], label="true")
p3 = scatter!(xid[1,:], xid[2,:], label="approx", linestyle =:dash, mc=:red, ms=2, ma=0.5, xlabel ="X1", ylabel="X2")
display(plot(p3, show = true, reuse = false))
savefig("linear2d_fig2.png")