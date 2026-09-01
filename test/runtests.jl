using SafeTestsets

@safetestset "Basis                                                                           " begin
    include("basis_tests.jl")
end
@safetestset "Training Data                                                                   " begin
    include("trainingdata_tests.jl")
end
@safetestset "Solvers                                                                         " begin
    include("solvers_tests.jl")
end
@safetestset "SINDy                                                                           " begin
    include("sindy_tests.jl")
end
@safetestset "Hamiltonian SINDy                                                               " begin
    include("hamiltonian_tests.jl")
end
@safetestset "JuliaGNI Conformance                                                            " begin
    include("conformance_tests.jl")
end
@safetestset "Aqua                                                                            " begin
    include("aqua_tests.jl")
end
