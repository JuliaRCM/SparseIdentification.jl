using SparseIdentification
using Documenter

DocMeta.setdocmeta!(SparseIdentification, :DocTestSetup, :(using SparseIdentification); recursive=true)

makedocs(;
    modules=[SparseIdentification],
    authors="Michael Kraus",
    repo="https://github.com/JuliaRCM/SparseIdentification.jl/blob/{commit}{path}#{line}",
    sitename="SparseIdentification.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaRCM.github.io/SparseIdentification.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Library" => "library.md",
    ],
)

deploydocs(;
    repo   = "github.com/JuliaRCM/SparseIdentification.jl",
    devurl = "latest",
    devbranch = "main",
)
