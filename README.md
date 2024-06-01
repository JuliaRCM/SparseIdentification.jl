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

# Define the system
Θ = [x^2 for x in -10:0.1:10]  # Example basis functions
ẋ = [2x + 1 for x in -10:0.1:10]  # Example gradient data

# collect training data
data = TrainingData(Float32.(x), Float32.(ẋ))

# generate a basis
basis = CompoundBasis(polyorder = 3, trigonometric = 0)

# Perform SINDy
method = SINDyMethod(noise_level=0.01, lambda=0.05, nloops=5)

# select a solver type JuliaLeastSquare(), OptimSolver(), or NNsolver() and pass it to the VectorField
solverType = JuliaLeastSquare()
vectorfield = VectorField(method, basis, data, solver = solverType)

# Display the identified coefficients
println(vectorfield.Ξ)
```

## Intrinsic Coordinate Identification with SINDy

The package also supports intrinsic coordinate identification using SINDy, as demonstrated with the [Lorenz System](scripts/non-hamil_examples/ex2_lorenz.jl) example. This involves training models with Flux.jl and leveraging automatic differentiation with Zygote.jl.

## API Documentation

For more detailed information about the functions in SparseIdentification.jl, see the [API Documentation](https://juliarcm.github.io/SparseIdentification.jl/latest/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact

If you have any questions or need support, please contact us at [email](mailto:nigelbrk@gmail.com).