using Aqua
using SparseIdentification
using Test

# Piracy is the check that matters here: this package adds methods to `SimpleSolvers.solve`, and
# those are legitimate only because the dispatching argument is one of our own solver types.
# Those same methods are what `test_ambiguities` guards: `Θ` is annotated `AbstractMatrix` so that
# they cannot be ambiguous against `SimpleSolvers.solve`'s own `LinearSolver`, `Linesearch` and
# `LinesearchProblem` methods, none of which is an `AbstractMatrix`.
@testset "Aqua" begin
    Aqua.test_ambiguities(SparseIdentification)
    Aqua.test_piracies(SparseIdentification)
    Aqua.test_stale_deps(SparseIdentification)
    Aqua.test_undefined_exports(SparseIdentification)
    Aqua.test_project_extras(SparseIdentification)
    Aqua.test_deps_compat(SparseIdentification)
end
