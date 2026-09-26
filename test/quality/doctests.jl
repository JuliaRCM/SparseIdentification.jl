using Documenter
using SparseIdentification

# Documenter evaluates a page's `@meta` block in `Main`, and this file runs in a module of its own.
@eval Main import SparseIdentification

DocMeta.setdocmeta!(
    SparseIdentification, :DocTestSetup, :(using SparseIdentification);
    recursive = true)

doctest(SparseIdentification)
