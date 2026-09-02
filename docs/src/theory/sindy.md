# Sparse Identification

SINDy — Sparse Identification of Nonlinear Dynamics — recovers the governing equations of a
dynamical system from measurements of its state, by solving a *sparse regression* problem over a
fixed library of candidate functions. It was introduced by Brunton, Proctor and Kutz
[BruntonProctorKutz2016](@ref).

## The idea

The premise is a physical one: **the governing equations of most systems are sparse in a sensible
basis.** The Lorenz equations have seven terms. The Navier–Stokes equations have a handful. If you
write down a library of a few hundred plausible candidate terms and ask which combination explains
the data, the honest answer usually involves very few of them.

That turns model discovery into a variable-selection problem, which is tractable, rather than a
search over all possible functions, which is not.

## The regression

Collect ``m`` snapshots of an ``n``-dimensional state into a matrix, and the corresponding time
derivatives alongside:

```math
X = \begin{pmatrix} x(t_1)^\top \\ \vdots \\ x(t_m)^\top \end{pmatrix} \in \mathbb{R}^{m \times n},
\qquad
\dot X = \begin{pmatrix} \dot x(t_1)^\top \\ \vdots \\ \dot x(t_m)^\top \end{pmatrix} .
```

Choose a library of ``p`` candidate functions and evaluate every one of them on every snapshot:

```math
\Theta(X) = \begin{pmatrix}
  \mid & \mid & & \mid \\
  \theta_1(X) & \theta_2(X) & \cdots & \theta_p(X) \\
  \mid & \mid & & \mid
\end{pmatrix} \in \mathbb{R}^{m \times p} .
```

A typical library holds a constant, the state itself, all monomials up to some degree, and
trigonometric terms. The ansatz is then linear:

```math
\dot X = \Theta(X)\, \Xi , \qquad \Xi \in \mathbb{R}^{p \times n} .
```

Column ``j`` of ``\Xi`` selects which library terms appear in the equation for ``\dot x_j``. The
whole problem is to find a **sparse** ``\Xi``.

Take ``m \gg p``: many more snapshots than candidates. The system is then heavily overdetermined,
which is what makes the fit robust to noise.

## Sequentially thresholded least squares

The sparsity is imposed by STLSQ, which is deliberately simple:

1. Fit ``\Xi \leftarrow \Theta \backslash \dot X`` by ordinary least squares.
2. Set every coefficient with ``|\Xi_{ij}| < \lambda`` to exactly zero.
3. Refit each column by least squares **using only the surviving terms** of that column.
4. Repeat from 2 until the support stops changing.

Two points are easy to get wrong:

- ``\lambda`` is a **hard threshold on coefficient magnitude, applied after the fit**. It is not an
  ``\ell^1`` penalty and it never appears in the objective being minimised. This is what
  distinguishes STLSQ from LASSO.
- The refit is done **per state component**, not on the whole matrix at once, because different
  components generally retain different library terms.

Discarded terms stay discarded. Each iteration works on a strictly smaller support, which is why
the algorithm terminates quickly — usually within two or three passes.

```@docs; canonical=false
SparseIdentification.SINDy
```

## What is actually guaranteed

Zhang and Schaeffer [ZhangSchaeffer2019](@ref) analysed exactly this algorithm. Writing ``A =
\Theta`` and ``b`` for one column of ``\dot X``, STLSQ is a fixed-point iteration on

```math
F(x) = \lVert Ax - b \rVert_2^2 + \lambda^2 \lVert x \rVert_0 ,
```

the ``\ell^0``-penalised least-squares objective. They prove, for ``A`` of full column rank with
``m \ge p``:

- the iteration **terminates in at most ``p`` steps**;
- ``F`` **strictly decreases** at every non-stationary step;
- the limit is a **local minimiser** of ``F``, and every global minimiser is a fixed point.

Both of the first two are directly testable, and each has a testset in `test/sindy_tests.jl`:
"Thresholding is exact and terminating" pins the step bound, and "The thresholding objective
decreases" recovers the iterates through `nloops` and checks ``F`` along them.

Note the guarantee is *local*. STLSQ converging does not certify that you found the true model —
only that you found a local minimiser of a sparsity-penalised objective. Recovery of the true
support additionally needs the library to actually contain the true terms and to be reasonably
well conditioned on your data.

## The ridge variant

Where the library is collinear — high-degree polynomials on clustered data will do it — the
least-squares refit is ill-conditioned. The standard remedy adds an ``\ell^2`` term,

```math
F_1(x) = \lVert Ax - b \rVert_2^2 + \gamma \lVert x \rVert_2^2 + \lambda^2 \lVert x \rVert_0 ,
```

implemented by augmenting ``A`` with ``\sqrt{\gamma}\, I`` and ``b`` with zeros. ``\gamma`` and
``\lambda`` are **independent knobs composed by matrix augmentation**, not a trade-off: ``\gamma``
conditions each refit, ``\lambda`` selects the support. It is not implemented here — [`SINDy`](@ref)
exposes ``\lambda`` alone, and there is no ``\gamma``.

## Where the derivatives come from

The formulation needs ``\dot X``. Sometimes it is measured; usually it is not, and must be
estimated from ``X`` by numerical differentiation — which amplifies noise badly. This is the single
largest practical weakness of the formulation above, and it has two established answers:

- **Denoise the derivative**, e.g. by total-variation regularised differentiation, as the original
  paper does.
- **Never differentiate at all.** Integrate the equation against a smooth compactly-supported test
  function ``\varphi`` and move the derivative onto ``\varphi`` by parts:
  ```math
  -\int \varphi'(t)\, x(t)\, dt = \int \varphi(t)\, f(x(t))\, dt .
  ```
  This is Weak SINDy [MessengerBortz2021](@ref), and it is substantially more noise-robust. It is
  not yet implemented here.

The Hamiltonian side of this package offers a third answer specific to it — matching the flow map
over one time step instead of the vector field — described in
[Fitting without derivative data](@ref).

## Limitations worth knowing before you start

- **The answer must be in the library.** SINDy cannot discover a term you did not offer it. A
  polynomial library will never find ``\cos(q)`` or ``1/\lVert q \rVert``.
- **Coordinates matter.** The paper's own failure cases are a nonlinearly-transformed Lorenz system
  and a glycolytic oscillator, both sparse in *some* basis but not the one supplied.
- **A single global ``\lambda`` assumes comparable coefficient magnitudes.** When the true
  coefficients span many orders of magnitude, no threshold separates signal from noise. See
  [When It Fails](@ref) for a worked case where this is fatal.
