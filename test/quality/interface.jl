using GeometricBase
using InteractiveUtils: subtypes
using SparseIdentification
using Test

# One instance of each concrete method type of the package, built as the tests build them, plus
# the instance from a constructor of no argument where the type has one.
const METHODS = [
    SINDy(CompoundBasis(polyorder = 2); λ = 0.05),
    HamiltonianSINDy(λ = 0.05, polyorder = 2),
    HamiltonianSINDy(CompoundBasis(polyorder = 2); λ = 0.05, integrator_timestep = 0.01)
]

for T in (SINDy, HamiltonianSINDy)
    hasmethod(T, Tuple{}) && push!(METHODS, T())
end

function owned_method_types(T = GeometricBase.AbstractMethod, found = Any[])
    for S in subtypes(T)
        if !isabstracttype(Base.unwrap_unionall(S)) &&
           Base.moduleroot(parentmodule(S)) === SparseIdentification
            push!(found, S)
        end
        owned_method_types(S, found)
    end
    found
end

@testset "GeometricBase interface" begin
    @testset "$i: $(nameof(typeof(m)))" for (i, m) in enumerate(METHODS)
        @test isempty(GeometricBase.interface_violations(m))
    end

    # A new method type without an instance above turns this red.
    @testset "Every owned method type is covered: $T" for T in owned_method_types()
        @test any(m -> m isa T, METHODS)
    end
end
