"""
    ExampleSearch

Search & presentation framework for a thesis "running example" illustrating the
precision ladder of neural network verification algorithms on tiny hand-checkable
networks (1 input, two hidden ReLU layers, 1 output):

    (i)    interval propagation                 NV.Box()
    (ii)   symbolic interval propagation        DPNFV(max_vars=0, method=:OuterBoundsNeurify)
    (iii)  symbolic intervals, separate relax.  DPNFV(max_vars=0, method=:NeurifyRelax)
    (iii')      + fresh variables               DPNFV(max_vars>0, method=:NeurifyRelax)
                                               (also evaluated: :DeepPolyRelax)
    (iv)   zonotope propagation                 NV.Ai2z() = Ai2{Zonotope} (DeepZ, Thm 3.1)
    (v)    backsubstitution (CROWN/DeepPoly)    aCROWN(), n_steps=0
    (vi)   backsubstitution + poly first layer  PolyCROWN(DiffNNPolySym(...)), n_steps=0
    (vii)  alpha-CROWN                          aCROWN(), n_steps>0
    (viii) NNPoly with optimized relaxations    PolyCROWN(...), n_steps>0

Desired ordering of the concrete upper bounds (looser > tighter):

    interval > sym. interval > sym. interval (sep.) > {+ fresh vars, zonotope}
    sym. interval > CROWN(0) > NNPoly(0)   (CROWN(0) beats the *plain* forward
                                            sym. interval; no ordering w.r.t.
                                            the sep./fresh-var/zonotope rungs)
    CROWN(opt) < CROWN(0), NNPoly(opt) < NNPoly(0), NNPoly(opt) < CROWN(opt)

The search only accepts networks where *all* intermediate quantities (DPNFV
per-layer bounds & symbolic coefficients, zonotope centers/generators, CROWN
Λ/γ and the final bounds) are fractions with small denominators — with an
exemption for NNPoly (poly) and for the optimized variants.

Usage:

    include("debug/example_search/ExampleSearch.jl")
    .Main.ExampleSearch.pilot()
    .Main.ExampleSearch.fuzz(n_trials=50_000)

Found candidates are persisted in `candidates.jld2` next to this file and can
be inspected with `print_candidate(idx)` / `list_candidates()` /
`print_bound_representation(idx, solver)` / `print_zono_trace(...)` /
`print_niceness(...)`; plots via `plot_candidate` (requires Plots).
"""
module ExampleSearch

using Random, LinearAlgebra, Dates
using JLD2
using Flux
using LazySets
import NNPoly as NP
import NeuralVerification as NV

include("rational_mode.jl")
include("zonotope.jl")
include("networks.jl")
include("bounds.jl")
include("niceness.jl")
include("constraints.jl")
include("registry.jl")
include("search.jl")
include("polish.jl")
include("tex_export.jl")

# Presentation & plotting (optional: requires Plots.jl)
try
    include("present.jl")
catch e
    @warn "Skipping presentation/plotting (present.jl) — Plots.jl not loadable" exception=e
end

end # module
