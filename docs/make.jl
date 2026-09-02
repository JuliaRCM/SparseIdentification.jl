using SparseIdentification
using Documenter

DocMeta.setdocmeta!(SparseIdentification, :DocTestSetup, :(using SparseIdentification);
    recursive = true)

# The changelog is the single record of how the package got here; the docs render a copy of it
# rather than keeping a second, drifting account.
cp(
    normpath(@__FILE__, "../../CHANGELOG.md"), normpath(@__FILE__, "../src/releasenotes.md");
    force = true)

makedocs(;
    modules = [SparseIdentification],
    authors = "Michael Kraus <michael.kraus@ipp.mpg.de> and contributors",
    repo = "https://github.com/JuliaRCM/SparseIdentification.jl/blob/{commit}{path}#{line}",
    sitename = "SparseIdentification.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaRCM.github.io/SparseIdentification.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = 500_000
    ),
    # Every error class is fatal except missing docstrings, so a broken doctest, `@example` or
    # cross-reference fails the build.
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "Theory" => [
            "Hamiltonian Systems" => "theory/hamiltonian_systems.md",
            "Sparse Identification" => "theory/sindy.md",
            "Hamiltonian SINDy" => "theory/hamiltonian_sindy.md"
        ],
        "Usage" => [
            "Getting Started" => "usage/getting_started.md",
            "Basis Libraries" => "usage/basis.md",
            "Choosing λ" => "usage/sparsification.md",
            "JuliaGNI Integration" => "usage/ecosystem.md"
        ],
        "Examples" => [
            "Linear 2D Oscillator" => "examples/linear_oscillator.md",
            "Lorenz Attractor" => "examples/lorenz.md",
            "Nonlinear Oscillator" => "examples/nonlinear_oscillator.md",
            "Toda Lattice" => "examples/toda_lattice.md"
        ],
        "When It Fails" => "limitations.md",
        "Library" => "library.md",
        "References" => "references.md",
        "Release Notes" => "releasenotes.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaRCM/SparseIdentification.jl",
    devurl = "latest",
    devbranch = "main"
)
