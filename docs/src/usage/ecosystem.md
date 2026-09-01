# JuliaGNI Integration

The package is built on the JuliaGNI stack and composes with it in both directions: an integrated
solution can be turned into training data, and an identified system can be turned back into a
problem an integrator accepts.

## The shape of the API

Identification mirrors integration. Where `GeometricIntegrators` has

```julia
solution = integrate(problem, method)
```

this package has

```julia
result = identify(problem, method)
```

`TrainingData` and `TrajectoryData` are `GeometricBase.AbstractProblem`s, and the methods are
`GeometricBase.AbstractMethod`s, so the ecosystem's accessors work on them: `nsamples`,
`statedimension`, `timestep`, `datatype`, `arrtype`, `parameters`, `functions`.

```@docs; canonical=false
SparseIdentification.identify
SparseIdentification.IdentificationProblem
SparseIdentification.SparsificationMethod
```

## Closing the loop

The full round trip — integrate a known system, identify it from the solution, and integrate the
identified system:

```@example gni
using SparseIdentification
using GeometricIntegrators
using GeometricProblems.HarmonicOscillator

# 1. a reference problem, integrated
reference = hodeproblem()
solution  = integrate(reference, ImplicitMidpoint())

# 2. its solution as training data — for a Hamiltonian system the state is z = (q, p)
data = TrajectoryData(solution)
(nsamples(data), statedimension(data))
```

```@example gni
# 3. identify
method = HamiltonianSINDy(λ = 0.01, integrator_timestep = timestep(data), polyorder = 2)
result = identify(data, method)
```

```@example gni
# 4. back to a problem, and integrate it
identified = HODEProblem(result, (0.0, 1.0), 0.01, [0.5], [0.0])
idsolution = integrate(identified, ImplicitMidpoint())

typeof(idsolution).name.name
```

Because the identified problem carries the identified Hamiltonian as an invariant, the energy
behaviour of the integrated solution can be checked with `GeometricSolutions`' own diagnostics
rather than by hand.

```@docs; canonical=false
SparseIdentification.TrainingData
SparseIdentification.TrajectoryData
```

## Method traits

The ecosystem's trait functions are answered where they mean something for a regression method.
The substantive ones are whether the *identified model* is structure-preserving:

```@example gni
using SparseIdentification: issymplectic, isenergypreserving

sindy = SINDy(CompoundBasis(polyorder = 2); λ = 0.05)
ham   = HamiltonianSINDy(λ = 0.05, polyorder = 2)

(issymplectic(sindy), issymplectic(ham))
```

That is the difference between the two methods stated as a property rather than as prose.

!!! note "These six traits are not exported"
    `isexplicit`, `isimplicit`, `issymmetric`, `issymplectic`, `isenergypreserving` and
    `isstifflyaccurate` are extended here but reached qualified, as above.

    `GeometricIntegratorsBase` defines its *own* generic functions of those six names rather than
    extending `GeometricBase`'s stubs, and exports them. A session with both
    `using SparseIdentification` and `using GeometricIntegrators` would therefore see two different
    bindings under one name and resolve it to neither. Not exporting them is the same choice
    `SimpleSolvers` makes for `status` and `isconverged`, for the same reason.

`issymmetric`, `isstifflyaccurate` and `order` describe a Runge–Kutta tableau and have no meaning
for a regression, so they are left as `missing`. `GeometricBase.isAbstractMethod` consequently
returns `false` for these methods — which is correct, since it is a conformance check for
*integrator* methods, not a general one.

## What is deliberately not reused

`GeometricBase.GeometricData` wraps a `NamedTuple` of series tagged by system and data type. It
is not used here: it carries no dimensions or sample count, has no outer constructor, and adding
one from this package would be type piracy. `TrainingData` and `TrajectoryData` validate their
shapes at construction and answer the ecosystem's accessors, which is what the wrapper would have
been for.
