# References

## The method implemented here

- [Khan2023](@id Khan2023)
  **Nigel Bruce Khan**, *Sparse Identification of Symplectic Hamiltonian Dynamics for Predictive
  Modeling and Analysis*. Master's Thesis, Technische Universität München, 30 November 2023.
  Examiner: Univ.-Prof. Dr. Eric Sonnendrücker; scientific advisor: Dr. Michael Kraus.
  [mediaTUM 1747893](https://mediatum.ub.tum.de/1747893)

  The source of the Hamiltonian-SINDy method and of this package. Introduces both the
  Hamiltonian-SINDy algorithm and an Auto-Encoder-Hamiltonian-SINDy variant that discovers
  canonical conjugate coordinates jointly with the dynamics; only the former is implemented here.

  Statements taken from this thesis are checked in `scripts/verify_thesis_examples.jl` rather than
  transcribed. Three of its equations do not hold as printed — see
  [Nonlinear Oscillator](@ref) and [When It Fails](@ref).

## Foundational SINDy

- [BruntonProctorKutz2016](@id BruntonProctorKutz2016)
  **Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz**, *Discovering governing equations from
  data by sparse identification of nonlinear dynamical systems*. Proceedings of the National
  Academy of Sciences **113**(15), 3932–3937, 2016.
  [doi:10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113) ·
  [arXiv:1509.03580](https://arxiv.org/abs/1509.03580)

  The original method: the library formulation ``\dot X = \Theta(X)\Xi``, sequentially thresholded
  least squares, and the Lorenz and damped-oscillator examples reproduced in this documentation.

- [ZhangSchaeffer2019](@id ZhangSchaeffer2019)
  **Linan Zhang, Hayden Schaeffer**, *On the Convergence of the SINDy Algorithm*. Multiscale
  Modeling & Simulation **17**(3), 948–972, 2019.
  [doi:10.1137/18M1189828](https://doi.org/10.1137/18M1189828) ·
  [arXiv:1805.06445](https://arxiv.org/abs/1805.06445)

  The convergence analysis: STLSQ as a fixed-point iteration on
  ``\lVert Ax-b\rVert_2^2 + \lambda^2\lVert x\rVert_0``, termination in at most ``p`` steps, strict
  decrease, and convergence to a local minimiser. Also the ridge variant (STRidge) and the reason
  ``\lambda`` and ``\gamma`` compose rather than trade off.

## Extensions referenced in the text

- [MessengerBortz2021](@id MessengerBortz2021)
  **Daniel A. Messenger, David M. Bortz**, *Weak SINDy: Galerkin-Based Data-Driven Model
  Selection*. Multiscale Modeling & Simulation **19**(3), 1474–1497, 2021.
  [doi:10.1137/20M1343166](https://doi.org/10.1137/20M1343166) ·
  [arXiv:2005.04339](https://arxiv.org/abs/2005.04339)

  The weak formulation, which integrates against compactly supported test functions and so never
  differentiates noisy data. The principal answer to the derivative-estimation problem discussed in
  [Sparse Identification](@ref). Not yet implemented here.

- [Champion2019](@id Champion2019)
  **Kathleen Champion, Bethany Lusch, J. Nathan Kutz, Steven L. Brunton**, *Data-driven discovery
  of coordinates and governing equations*. PNAS **116**(45), 22445–22451, 2019.
  [doi:10.1073/pnas.1906995116](https://doi.org/10.1073/pnas.1906995116) ·
  [arXiv:1904.02107](https://arxiv.org/abs/1904.02107)

  The autoencoder–SINDy coupling that the thesis's Auto-Encoder-Hamiltonian-SINDy builds on.

- [Fasel2022](@id Fasel2022)
  **Urban Fasel, J. Nathan Kutz, Bingni W. Brunton, Steven L. Brunton**, *Ensemble-SINDy: Robust
  sparse model discovery in the low-data, high-noise limit, with active learning and control*.
  Proceedings of the Royal Society A **478**(2260), 20210904, 2022.
  [doi:10.1098/rspa.2021.0904](https://doi.org/10.1098/rspa.2021.0904) ·
  [arXiv:2111.10992](https://arxiv.org/abs/2111.10992)

- [Kaptanoglu2022](@id Kaptanoglu2022)
  **Alan A. Kaptanoglu et al.**, *PySINDy: A comprehensive Python package for robust sparse system
  identification*. Journal of Open Source Software **7**(69), 3994, 2022.
  [doi:10.21105/joss.03994](https://doi.org/10.21105/joss.03994)

  The reference implementation in Python, and a useful checklist of optimizers and feature
  libraries.

## Structure-preserving identification

- [LeeTraskStinis2021](@id LeeTraskStinis2021)
  **Kookjin Lee, Nathaniel Trask, Panos Stinis**, *Structure-preserving Sparse Identification of
  Nonlinear Dynamics for Data-driven Modeling*. Proceedings of Mathematical and Scientific Machine
  Learning, PMLR **190**, 65–80, 2022. [arXiv:2109.05364](https://arxiv.org/abs/2109.05364)

  A more general bracket-based framework covering both Poisson (conservative) and metric
  (dissipative) structure.

- [Greydanus2019](@id Greydanus2019)
  **Samuel Greydanus, Misko Dzamba, Jason Yosinski**, *Hamiltonian Neural Networks*. NeurIPS 2019.
  [arXiv:1906.01563](https://arxiv.org/abs/1906.01563)

  The neural counterpart: parametrise ``H`` by a network rather than a sparse basis. Accurate but
  not interpretable, which is the trade-off sparse identification exists to avoid.

## Software

- **GeometricIntegrators.jl** — symplectic and variational integrators.
  <https://github.com/JuliaGNI/GeometricIntegrators.jl>
- **GeometricProblems.jl** — the reference test systems used in this documentation.
  <https://github.com/JuliaGNI/GeometricProblems.jl>
- **Symbolics.jl** — the symbolic layer that builds the basis and differentiates the Hamiltonian.
  <https://github.com/JuliaSymbolics/Symbolics.jl>
