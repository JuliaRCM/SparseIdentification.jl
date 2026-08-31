# Release Notes

All notable changes to SparseIdentification.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually
changed, so that a compat-only bump can be told apart from a rename or a change in results.

This file was started on 2026-08-31 and deliberately holds no entries. Nothing has been
released yet — there are no tags, and `Project.toml` stands at `0.1.0` — so the development
history that predates this file is in `git log` alone. It is named as a gap rather than
reconstructed, because a changelog assembled after the fact loses exactly the reasoning that
makes it worth keeping.

## [Unreleased] — targeting 0.1.0

### New Features

### Bug Fixes

### Breaking Changes

- **Minimum Julia is now 1.10**, raised from the declared 1.6. 1.10 is the LTS and the floor across
  the whole tree; 1.6 was declared but never tested, and CI was still running a `1.6 · 1.9 ·
  ^1.10.0-0` matrix. CI now derives its lower matrix entry from this field, so a declared floor that
  nobody tests is no longer possible.

## Open Issues

- **The package does not load.** `PreallocationTools` 0.4.34 moved its `ForwardDiff` methods into an
  extension, so `ForwardDiff` is undefined there unless the extension loads; `MKL_jll` reaches the
  environment through `LinearSolve`. Both are upstream. The pinned `ForwardDiff = "0.10"` and
  `Symbolics = "5"` bounds are part of the picture and want revisiting together. Recorded
  2026-08-31.
