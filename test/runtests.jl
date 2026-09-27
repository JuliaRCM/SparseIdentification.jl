using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Basis" include("basis.jl")
    @safetestset "Extended Bases" include("basis_extended.jl")
    @safetestset "Training Data" include("trainingdata.jl")
    @safetestset "Solvers" include("solvers.jl")
    @safetestset "SINDy" include("methods/sindy.jl")
    @safetestset "Hamiltonian SINDy" include("methods/hamiltonian.jl")
    @safetestset "JuliaGNI Conformance" include("integration/conformance.jl")
end
if "slow" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
