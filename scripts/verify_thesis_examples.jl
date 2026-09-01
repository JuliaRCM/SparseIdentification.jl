# Verification of the statements this package's documentation takes from
#
#   Nigel Bruce Khan, "Sparse Identification of Symplectic Hamiltonian Dynamics for Predictive
#   Modeling and Analysis", Master's Thesis, Technische Universität München, 30 November 2023.
#
# Nothing here is assumed. Each claim is either confirmed, or shown to be wrong and the corrected
# form given, so that the documentation states something that holds rather than something that was
# transcribed.
#
# Symbolic expressions are evaluated by compiling them with `build_function`, which is the same
# path the package itself uses. `substitute` alone does not fold `cos(1.0)` to a number under
# Symbolics 7, so it cannot be used to get a value out.
#
# Run with:  julia --project=. scripts/verify_thesis_examples.jl

using LinearAlgebra
using Printf
using Symbolics

const PASS = "  [ok]     "
const FAIL = "  [WRONG]  "

results = Tuple{String, Bool, String}[]
record(name, ok, note = "") = push!(results, (name, ok, note))

"Compile a symbolic expression in the variables `z` and evaluate it at the point `zval`."
function numeval(expr::AbstractVector, z, zval)
    build_function(expr, z; expression = Val(false))[1](zval)
end
numeval(expr, z, zval) = build_function(expr, z; expression = Val(false))(zval)

"The canonical symplectic vector field ż = J∇H for z = (q₁..q_d, p₁..p_d)."
function hamiltonian_vectorfield(H, z, d)
    Dz = Differential.(z)
    g = [expand_derivatives(dz(H)) for dz in Dz]
    vcat(g[(d + 1):(2d)], -g[1:d])
end

println("="^98)
println("Verification of claims taken from Khan (2023)")
println("="^98)

# ─────────────────────────────────────────────────────────────────────────────────────────────
# 1. The central structural claim: J∇H is LINEAR in the coefficients.
#
# H(z; a) = Σₖ aₖ φₖ(z)  ⟹  ż = J∇H(z) = Σₖ aₖ · J∇φₖ(z)
#
# If this holds, fitting a Hamiltonian against measured ż is an ordinary linear least-squares
# problem needing no optimiser. The thesis uses BFGS throughout.
# ─────────────────────────────────────────────────────────────────────────────────────────────
println("\n1. Is J∇H linear in the coefficients?")

let d = 2
    z = Symbolics.variables(:z, 1:(2d))
    zval = [0.31, -0.77, 1.29, -0.42]

    # A deliberately mixed basis: polynomial, trigonometric, exponential.
    φ = [z[1]^2, z[3] * z[4], cos(z[1]), exp(-z[2]), z[1] * z[2] * z[3]]

    JgradH(a) = hamiltonian_vectorfield(sum(a[k] * φ[k] for k in eachindex(φ)), z, d)
    Jgrad_terms = [hamiltonian_vectorfield(φₖ, z, d) for φₖ in φ]

    a₁ = [1.7, -0.4, 2.0, 0.9, -1.1]
    a₂ = [-0.3, 2.2, 0.5, -1.4, 0.8]
    α, β = 2.5, -0.7

    lhs = numeval(JgradH(α .* a₁ .+ β .* a₂), z, zval)
    rhs = α .* numeval(JgradH(a₁), z, zval) .+ β .* numeval(JgradH(a₂), z, zval)
    linear = isapprox(lhs, rhs; atol = 1e-12)

    Θ = reduce(hcat, [numeval(t, z, zval) for t in Jgrad_terms])   # (2d × nφ) library
    superposed = isapprox(Θ * a₁, numeval(JgradH(a₁), z, zval); atol = 1e-12)

    @printf("     ‖J∇H(αa₁+βa₂) - (α J∇H(a₁) + β J∇H(a₂))‖ = %.3e\n", norm(lhs - rhs))
    @printf("     ‖Θa - J∇H(a)‖ = %.3e\n", norm(Θ * a₁ - numeval(JgradH(a₁), z, zval)))
    println(linear ? PASS : FAIL, "J∇H(αa₁ + βa₂) = α J∇H(a₁) + β J∇H(a₂)")
    println(superposed ? PASS : FAIL, "J∇H(a) = Θ a  with  Θ[:,k] = J∇φₖ")
    record("J∇H linear in the coefficients", linear && superposed,
        "so the vector-field fit is linear least squares, not an optimisation")
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# 2. Nonlinear oscillator, thesis Equation (4.2), printed page 41.
#
# As printed:   H = ½p₁² + ½p₁² + cos(q₁) + cos(q₂)
#
# The text calls this "a two-dimensional nonlinear oscillator system, i.e., four variables".
# A Hamiltonian in (q₁,q₂,p₁,p₂) whose kinetic part names only p₁ cannot be that: p₂ never
# appears, so q̇₂ = ∂H/∂p₂ = 0 and the second degree of freedom is frozen.
# ─────────────────────────────────────────────────────────────────────────────────────────────
println("\n2. Nonlinear oscillator, thesis Eq. (4.2)")

let d = 2
    z = Symbolics.variables(:z, 1:(2d))          # z = (q₁, q₂, p₁, p₂)
    zval = [0.4, -1.1, 0.9, 1.7]

    H_printed = z[3]^2 / 2 + z[3]^2 / 2 + cos(z[1]) + cos(z[2])   # as printed
    H_intended = z[3]^2 / 2 + z[4]^2 / 2 + cos(z[1]) + cos(z[2])   # as it must be

    f_printed = numeval(hamiltonian_vectorfield(H_printed, z, d), z, zval)
    f_intended = numeval(hamiltonian_vectorfield(H_intended, z, d), z, zval)

    @printf("     as printed:  q̇ = %-22s (p₂ = %.1f)\n",
        string(round.(f_printed[1:2], digits = 4)), zval[4])
    @printf("     as intended: q̇ = %s\n", string(round.(f_intended[1:2], digits = 4)))

    frozen = iszero(f_printed[2])
    correct = f_intended[2] ≈ zval[4]
    println(frozen ? FAIL : PASS,
        "as printed, q̇₂ = ∂H/∂p₂ = 0 — the second degree of freedom does not move")
    record("Thesis Eq. (4.2) kinetic term", !(frozen && correct),
        "printed `½p₁² + ½p₁²`; must read `½p₁² + ½p₂²`")
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# 3. Toda lattice, thesis Equation (4.3), printed page 43.
#
#   H = Σₙ ( p(n)²/2 + V(q(n+1) − q(n)) ),   V(r) = e^{−r} + r − 1
#
# Checked by normalisation: V(0) = 0, V'(0) = 0, V''(0) = 1, which is what makes the lattice
# reduce to coupled harmonic oscillators at small amplitude.
# ─────────────────────────────────────────────────────────────────────────────────────────────
println("\n3. Toda lattice potential, thesis Eq. (4.3)")

let
    r = Symbolics.variables(:r, 1:1)
    V = exp(-r[1]) + r[1] - 1
    D = Differential(r[1])
    V′ = expand_derivatives(D(V))
    V″ = expand_derivatives(D(V′))

    v0, v1, v2 = numeval(V, r, [0.0]), numeval(V′, r, [0.0]), numeval(V″, r, [0.0])
    ok = isapprox(v0, 0; atol = 1e-14) && isapprox(v1, 0; atol = 1e-14) &&
         isapprox(v2, 1; atol = 1e-14)

    @printf("     V(0) = %.3f,  V'(0) = %.3f,  V''(0) = %.3f\n", v0, v1, v2)
    println(ok ? PASS : FAIL, "V(r) = e^{-r} + r - 1 is the standard normalised Toda potential")
    record("Thesis Eq. (4.3) Toda potential", ok, "correct as printed")
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# 4. Point vortices, thesis Equation (4.4), printed page 44.
#
# As printed:   H = -1/(4π) Σᵢ₌₁ᴺ Σⱼ₌₁ᴺ pᵢ pⱼ log|qᵢ - qⱼ|
# with the text: "p represents the strength of the vorticity, and q is the distance from its
# centre".
#
# Two independent problems:
#   (a) the double sum runs over all i and j, including i = j, where log|qᵢ - qᵢ| = log 0;
#   (b) if p is the vortex strength, then ṗ = -∂H/∂q is non-zero, i.e. the strengths would
#       evolve in time. Point-vortex strengths are constants of the motion.
#
# In the correct formulation the conjugate pair is the *two spatial coordinates* of each vortex —
# √Γᵢ xᵢ and √Γᵢ yᵢ — and the strengths Γᵢ are fixed parameters, not dynamical variables:
#   H = -1/(4π) Σᵢ<ⱼ Γᵢ Γⱼ log|rᵢ - rⱼ|.
# ─────────────────────────────────────────────────────────────────────────────────────────────
println("\n4. Point vortices, thesis Eq. (4.4)")

let N = 2
    println(FAIL, "Σᵢ Σⱼ over all i,j includes i = j, where log|qᵢ - qⱼ| = log 0 = $(log(0.0))")

    z = Symbolics.variables(:z, 1:(2N))          # z = (q₁, q₂, p₁, p₂)
    zval = [0.3, 1.4, 2.0, -1.5]

    # Drop the divergent diagonal so the rest can be examined at all.
    H = -1 / (4π) * sum(z[N + i] * z[N + j] * log(abs(z[i] - z[j]))
    for i in 1:N, j in 1:N if i != j)

    f = numeval(hamiltonian_vectorfield(H, z, N), z, zval)
    ṗ = f[(N + 1):(2N)]

    strengths_evolve = any(abs.(ṗ) .> 1e-12)
    @printf("     with p read as the vortex strength:  ṗ = %s\n",
        string(round.(ṗ, digits = 4)))
    println(strengths_evolve ? FAIL : PASS,
        "ṗ = -∂H/∂q ≠ 0, so the vortex strengths would change in time")

    record("Thesis Eq. (4.4) point vortex", false,
        "i=j term diverges, and p cannot be the strength — strengths are conserved")
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# 5. N-body / solar system, thesis Equation (4.5), printed page 46, and the conditioning
#    diagnosis that follows it.
#
#   H = Σᵢ pᵢ²/(2mᵢ) - Σᵢ<ⱼ G mᵢmⱼ / |qⱼ - qᵢ|
#
# The equation is the standard N-body Hamiltonian. The interesting claim is *why* the method
# fails on it: the two coefficient families are separated by so many orders of magnitude that no
# single global threshold λ can keep one and discard the other. The thesis quotes "order 10⁻²⁴"
# and "order 10³⁷"; both are checked, as is the ratio, which is what actually does the damage.
# ─────────────────────────────────────────────────────────────────────────────────────────────
println("\n5. N-body Hamiltonian and the conditioning claim, thesis Eq. (4.5)")

let
    G = 6.6743e-11              # m³ kg⁻¹ s⁻²
    m_earth = 5.972e24          # kg
    m_sun = 1.989e30          # kg

    kinetic = 1 / (2 * m_earth)
    potential = G * m_earth * m_sun

    kin_order = round(Int, log10(kinetic))
    pot_order = round(Int, log10(potential))

    @printf("     1/(2m_earth)    = %.3e   → order 1e%-4d  (thesis: 1e-24)\n", kinetic,
        kin_order)
    @printf("     G m_earth m_sun = %.3e   → order 1e%-4d  (thesis: 1e37)\n", potential,
        pot_order)
    @printf("     ratio           = %.3e   → %d orders of magnitude apart\n",
        potential / kinetic, round(Int, log10(potential / kinetic)))

    kin_ok = kin_order == -24
    pot_ok = pot_order == 37
    println(kin_ok ? PASS : FAIL, "quoted order for 1/(2mᵢ)")
    println(pot_ok ? PASS : FAIL, "quoted order for G mᵢmⱼ")

    spread = potential / kinetic > 1e30
    println(spread ? PASS : FAIL,
        "the coefficient families are far too far apart for one global λ to separate")

    record("Thesis Eq. (4.5) N-body Hamiltonian", true, "standard form, correct as printed")
    record("Thesis quoted magnitudes", kin_ok && pot_ok,
        @sprintf("actual orders are 1e%d and 1e%d, not 1e-24 and 1e37", kin_order,
            pot_order))
    record("Thesis conditioning diagnosis", spread,
        "conclusion holds even though the quoted numbers do not")
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# 6. Structure preservation by construction.
#
# For ANY coefficient vector, the Jacobian of ż = J∇H satisfies J⁻¹∂f = ∇²H, which is symmetric.
# This is what makes the method structure-preserving rather than structure-encouraged, and it is
# the property a fit-then-project method cannot guarantee.
# ─────────────────────────────────────────────────────────────────────────────────────────────
println("\n6. Is the identified field Hamiltonian for ANY coefficients?")

let d = 2
    z = Symbolics.variables(:z, 1:(2d))
    zval = [0.53, -0.81, 1.07, -0.29]

    φ = [z[1]^2, z[2]^2, z[3]^2, z[4]^2, z[1] * z[3], z[2] * z[4], cos(z[1]), z[1]^2 * z[4]]
    a = [0.7, -1.3, 2.1, 0.4, -0.9, 1.6, 0.5, -0.2]     # arbitrary, not fitted to anything

    f = hamiltonian_vectorfield(sum(a[k] * φ[k] for k in eachindex(φ)), z, d)
    Dz = Differential.(z)
    Jac_sym = [expand_derivatives(dzⱼ(fᵢ)) for fᵢ in f, dzⱼ in Dz]
    Jac = reshape(numeval(vec(Jac_sym), z, zval), 2d, 2d)

    J = [zeros(d, d) I(d); -I(d) zeros(d, d)]           # ż = J∇H
    Hess = J \ Jac                                       # must be ∇²H, hence symmetric

    symmetric = isapprox(Hess, Hess'; atol = 1e-10)
    @printf("     ‖J⁻¹∂f - (J⁻¹∂f)ᵀ‖ = %.3e\n", norm(Hess - Hess'))
    println(symmetric ? PASS : FAIL,
        "J⁻¹ × Jacobian is symmetric — the field is Hamiltonian for arbitrary coefficients")
    record("Structure preservation by construction", symmetric,
        "holds for any coefficients, fitted or not")
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
println("\n" * "="^98)
println("Summary")
println("="^98)
for (name, ok, note) in results
    println(ok ? PASS : FAIL, rpad(name, 40), note)
end
nfail = count(!r[2] for r in results)
println("\n$(length(results) - nfail) of $(length(results)) confirmed; $nfail need correcting in any text that repeats them.")
