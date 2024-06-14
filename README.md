# SparseIdentification

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaRCM.github.io/SparseIdentification.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaRCM.github.io/SparseIdentification.jl/dev/)
[![Build Status](https://github.com/JuliaRCM/SparseIdentification.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaRCM/SparseIdentification.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaRCM/SparseIdentification.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaRCM/SparseIdentification.jl)

## Overview

SparseIdentification.jl is a Julia package designed to implement the Sparse Identification of Nonlinear Dynamical systems (SINDy) algorithm for Hamiltonian and Newtonian Systems. SINDy is a powerful method for discovering governing equations from data, particularly useful for systems where the underlying model is sparse. This package re-implements the SINDy algorithm in Julia, providing an extension to the method for Hamiltonian Systems as well as for intrinsic coordinate discovery.

## Features

- **Fast Computation**: Utilizes Julia's high-performance capabilities for numerical optimization and root finding.
- **Robust to Noise**: The algorithm includes mechanisms to handle noisy data effectively.
- **Flexibility**: Supports various optimization methods and is easily extendable.
- **Integration with Julia Packages**: Leverages Julia packages like Flux.jl for machine learning tasks and Zygote.jl for automatic differentiation.

## Installation

To install SparseIdentification.jl, use the Julia package manager:

```julia
import Pkg
Pkg.add("SparseIdentification")
```

## Usage

Please refer to scripts folder to see example usages of the package. Here is a basic example of how to use SparseIdentification.jl:

```julia
using SparseIdentification

# initial data
x₀ = [2., 0.]

# 2D system 
nd = length(x₀)

# vector field
const A = [-0.1  2.0
           -2.0 -0.1]

rhs(xᵢₙ,p,t) = A*xᵢₙ

# number of samples
num_samp = 15

# example sample input range
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output
# s depend on size of nd (total dims), 2 in the case here so we use samp_range x samp_range
s = collect(Iterators.product(fill(samp_range, nd)...))

# compute vector field from x state values
x = [collect(s[i]) for i in eachindex(s)]

x = hcat(x...)

# compute vector field from x state values at each timestep
ẋ = zero(x)
for i in axes(ẋ,2)
    ẋ[:,i] .= A*x[:,i]
end

# collect training data (noise is added in the method itself)
data = TrainingData(Float32.(x), Float32.(ẋ))

# example specifications of SINDy method
method = SINDy(lambda = 0.05, noise_level = 0.01, nloops = 5)

# generate basis
#  - search space up to fifth order polynomials
#  - no trigonometric functions
basis = CompoundBasis(polyorder = 5, trigonometric = 0)

# select a solver type JuliaLeastSquare(), OptimSolver(), or NNsolver() and pass it to the VectorField
solverType = JuliaLeastSquare()
vectorfield = VectorField(method, basis, data, solver = solverType)

# Display the identified coefficients or basis
println(vectorfield.coefficients)
```

## Intrinsic Coordinate Identification with SINDy

The package also supports intrinsic coordinate identification using SINDy, as demonstrated with the [Lorenz System](scripts/non-hamil_examples/ex2_lorenz.jl) example. This involves training models with Flux.jl and leveraging automatic differentiation with Zygote.jl.

## API Documentation

For more detailed information about the functions in SparseIdentification.jl, see the [API Documentation](https://juliarcm.github.io/SparseIdentification.jl/latest/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact

If you have any questions or need support, please contact us at [email](mailto:nigelbrk@gmail.com).