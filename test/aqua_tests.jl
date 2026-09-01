using Aqua
using SparseIdentification
using Test

# Piracy is the check that matters here: this package adds methods to `SimpleSolvers.solve`, and
# those are legitimate only because the dispatching argument is one of our own solver types.
# `test_ambiguities` is not run, for the reason given in GeometricOptimizers' own aqua_tests.jl —
# it reports large numbers of LinearAlgebra pairs that are not this package's to fix.
@testset "Aqua" begin
    Aqua.test_piracies(SparseIdentification)
    Aqua.test_stale_deps(SparseIdentification)
    Aqua.test_undefined_exports(SparseIdentification)
    Aqua.test_project_extras(SparseIdentification)
    Aqua.test_deps_compat(SparseIdentification)
end
