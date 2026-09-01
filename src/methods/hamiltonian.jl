
##########################################################
# Symbolic construction of a parametrised Hamiltonian
##########################################################

" makes polynomial combinations of basis "
function hamiltonian_poly(z, order, inds...)
    ham = []

    if order == 0
        Num(1)
    elseif order == length(inds)
        ham = vcat(ham, _prod([z[i] for i in inds]...))
    else
        start_ind = length(inds) == 0 ? 1 : inds[end]
        for j in start_ind:length(z)
            ham = vcat(ham, hamiltonian_poly(z, order, inds..., j))
        end
    end

    return ham
end

" collects and sums only polynomial combinations of basis "
function hamiltonian(z, a, order)
    ham = []

    for i in 1:order
        ham = vcat(ham, hamiltonian_poly(z, i))
    end

    sum(collect(a .* ham))
end

" collects and sums polynomial and trigonometric combinations of basis "
function hamil_trig(z, a, order, trig_wave_num)
    ham = []

    # Polynomial basis
    for i in 1:order
        ham = vcat(ham, hamiltonian_poly(z, i))
    end

    # Trigonometric basis
    for k in 1:trig_wave_num
        ham = vcat(ham, vcat(sin.(k*z)), vcat(cos.(k*z)))
    end

    ham = sum(collect(a .* ham))

    return ham
end

"""
    HamiltonianFunctions

The compiled functions of a parametrised Hamiltonian `H(z; a)`, in the form
`GeometricEquations` expects.

# Fields

  - `H`: the Hamiltonian, callable as `H(t, q, p, params)` with `params.a` the coefficients
  - `v`: `q̇ = ∂H/∂p`, callable as `v(v, t, q, p, params)`
  - `f`: `ṗ = -∂H/∂q`, callable as `f(f, t, q, p, params)`
  - `ż`: the combined field `J∇H`, callable as `ż(out, z, a)` — the form the regression uses
  - `d`: the number of degrees of freedom
  - `nparam`: the number of coefficients

Splitting `v` and `f` is what lets an identified system become a `HODEProblem`; the combined `ż`
is kept because the fitting loop evaluates the whole field at once.
"""
struct HamiltonianFunctions{HT, VT, FT, ZT}
    H::HT
    v::VT
    f::FT
    ż::ZT
    d::Int
    nparam::Int
end

"""
    hamiltonian_functions(d, polyorder, trig_wave_num)

Build the symbolic Hamiltonian `H(z; a) = Σₖ aₖ φₖ(z)` over `d` degrees of freedom and compile it,
together with its symplectic gradient, into [`HamiltonianFunctions`](@ref).

The constant term is omitted: it contributes nothing to `∇H` and is therefore unidentifiable.
"""
function hamiltonian_functions(d::Int, polyorder::Int, trig_wave_num::Int)
    nparam = calculate_nparams(d, polyorder, trig_wave_num)

    @variables a[1:nparam]
    @variables q[1:d]
    @variables p[1:d]
    @variables t
    z = vcat(q, p)

    H = trig_wave_num > 0 ? hamil_trig(z, a, polyorder, trig_wave_num) :
        hamiltonian(z, a, polyorder)

    Dz = Differential.(z)
    ∇H = [expand_derivatives(dz(H)) for dz in Dz]

    # q̇ = ∂H/∂p and ṗ = -∂H/∂q
    v_expr = ∇H[(d + 1):(2d)]
    f_expr = -∇H[1:d]

    # The combined field the regression evaluates, as a function of (out, z, a).
    ż = @RuntimeGeneratedFunction(build_function(vcat(v_expr, f_expr), z, a)[2])

    # The GeometricEquations calling conventions. `a` is passed inside `params`, so the generated
    # code takes it as an ordinary argument and a wrapper unpacks it — the same one-definition,
    # two-callers split EulerLagrange uses.
    v_raw = @RuntimeGeneratedFunction(build_function(v_expr, q, p, a)[2])
    f_raw = @RuntimeGeneratedFunction(build_function(f_expr, q, p, a)[2])
    H_raw = @RuntimeGeneratedFunction(build_function(H, q, p, a))

    Hfun = (t, q, p, params) -> H_raw(q, p, params.a)
    vfun = (v, t, q, p, params) -> v_raw(v, q, p, params.a)
    ffun = (f, t, q, p, params) -> f_raw(f, q, p, params.a)

    HamiltonianFunctions(Hfun, vfun, ffun, ż, d, nparam)
end

"""
    hamilGrad_func_builder(d, polyorder, trig_wave_num)

The symplectic gradient `J∇H` of the parametrised Hamiltonian, callable as `out = f(out, z, a)`.

A thin wrapper over [`hamiltonian_functions`](@ref) for the regression, which needs only the
combined field.
"""
function hamilGrad_func_builder(d, polyorder, trig_wave_num)
    hamiltonian_functions(d, polyorder, trig_wave_num).ż
end
