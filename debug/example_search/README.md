# Example Search — thesis "running example" networks

Search & presentation framework for tiny networks (1 input, two hidden ReLU layers of
width 2, 1 output) that illustrate the **precision ladder of verification algorithms**
with hand-checkable small-denominator fractions.

## The ladder

Concrete upper bounds must strictly decrease along the rungs (tolerance `tol = 1e-2`):

| rung | method | solver call |
|------|--------|-------------|
| (i)   | interval propagation | `NV.Box()` |
| (ii)  | symbolic interval propagation (single relaxation) | `DPNFV(max_vars=0, method=:OuterBoundsNeurify)` |
| (iii) | symbolic interval propagation (separate lower/upper relaxations) | `DPNFV(max_vars=0, method=:NeurifyRelax)` |
| (iii')| + fresh variables | `DPNFV(max_vars=10, method=:NeurifyRelax)` (also stored: `:DeepPolyRelax`) |
| (iv)  | zonotope propagation (DeepZ / AI², Thm 3.1) | `NV.Ai2z()` — re-implemented type-generically in `zonotope.jl` |
| (v)   | backsubstitution (CROWN) | `aCROWN()`, `n_steps=0` |
| (vi)  | backsubstitution with poly first layer (NNPoly) | `PolyCROWN(DiffNNPolySym(common_generators=true), poly_layers=1)`, `n_steps=0` |
| (vii) | optimized αCROWN | `aCROWN()`, `n_steps=25` |
| (viii)| optimized NNPoly | `PolyCROWN(...)`, `n_steps=25` |

Required orderings: `(i) > (ii) > (iii) > {(iii'), (iv)}` (forward refinement
chain) and `(ii) > (v) > (vi)` — CROWN(0) beats the *plain* forward symbolic
interval propagation (no fresh vars, no separate bounds) and PolyCROWN(0); no
ordering is required between CROWN(0) and the separated/fresh-var/zonotope
rungs. Optimized rungs: `(vii) < (v)`, `(viii) < (vi)`, `(viii) < (vii)`.

Classification: `:accepted` (all links hold), `:near_miss` (exactly one link fails —
kept in the registry so nothing is lost), `:nice_fail` (ladder-perfect but ugly numbers).
The `:nice_fail`/`:near_miss` tags are pooled during the run and the best ones by
denominator (`max_den_seen`, then violation count) are saved at the end; for
`:near_miss` the best candidate of each distinct failed link is always kept.

## Niceness criterion

For the thesis example, *every hand-checkable quantity* must be a fraction with small
denominator (default `max_den = 32`): all final bounds of rungs (i)–(v), DPNFV per-layer
bounds and symbolic bound matrices `L`/`U`, zonotope centers/generators (computed in
exact `Rational{BigInt}` arithmetic), and the aCROWN(0) coefficients `Λ, λ, Γ, γ`.
Exempt: NNPoly polynomials and the optimized rungs (vii)/(viii).

## Files

| file | contents |
|------|----------|
| `ExampleSearch.jl` | module entry point, includes everything below |
| `rational_mode.jl` | exact-`Rational` support (slope-function method extension, conversions) |
| `zonotope.jl` | Theorem-3.1 ReLU relaxation on zonotopes, type-generic, with per-layer trace & pretty printer (`print_zono_trace`); validated against `NV.Ai2z` |
| `networks.jl` | network builders (`make_network`, `make_crown_chain`), sampler, layer rescaling |
| `bounds.jl` | staged upper-bound evaluation for rungs (i)–(viii) |
| `niceness.jl` | collection of all intermediate quantities + denominator scoring |
| `constraints.jl` | ladder link definitions, `classify_ladder` |
| `registry.jl` | `candidates.jld2` store + auto-generated `CANDIDATES.md` |
| `search.jl` | staged fuzzing driver (`fuzz`, `pilot`) |
| `polish.jl` | simulated-annealing refinement of registry seeds (`polish`) |
| `tex_export.jl` | exact-rational re-derivation of all rungs (i)–(vi) as LaTeX (`propagation_tex`, `write_propagation_tex`), with `validate_propagation` cross-checking every replication against the real solvers |
| `present.jl` | bound-representation printing & relaxation plots (requires Plots) |

## Usage

```julia
include("debug/example_search/ExampleSearch.jl")

# inspect a candidate
ExampleSearch.print_candidate(idx)
ExampleSearch.print_bound_representation(idx, NP.DPNFV(max_vars=10))
ExampleSearch.print_zono_trace(ExampleSearch.make_network(c.Ws, c.bs), ExampleSearch.default_input_set())
ExampleSearch.print_niceness(c.Ws, c.bs)

# yield estimation, then the search
ExampleSearch.pilot(n_trials=2000, structured_W1=true)
ExampleSearch.pilot(n_trials=2000, wide_alphabet=true, input_widths=[1.0, 2.0])
ExampleSearch.fuzz(n_trials=200_000; seed=101, structured_W1=true, save=true)

# plots (needs Plots)
ExampleSearch.plot_candidate(c, Dict("DPNFV" => Dict("max_vars" => [0, 10]),
                                     "aCROWN" => Dict("n_steps" => [0, 25]),
                                     "pCROWN" => Dict("n_steps" => [0, 25])))
```

## Local polish

Pools that are only a step or two away from `:accepted` (e.g. a near-miss whose
single link failure is an exact tie, plus one niceness offender) can be
*polished* instead of re-sampled globally:

```julia
ExampleSearch.polish(651; n_iter=20_000, seed=651)
```

`polish(seed_idx)` runs simulated annealing over the dyadic neighborhood of a
registry seed — weight/bias nudges by small dyadic `δ`, plus function- and
bound-preserving per-neuron / per-layer power-of-two rescalings — minimizing

    30·(failed links) + 12·(niceness violations)
        + 8·log₂(max_den/32)₊ + 3·log₂(max numerator/32)₊

so the search trades link fixes against niceness fixes in one balanced
objective (the numerator term captures hand-checkability, not just
denominators). Evaluation is staged like `fuzz` (a move is dropped right after
the early links when its link-failure count already exceeds the current
corridor of +1). Fully-accepted candidates are saved to the registry as they
appear; the best solution is saved at the end if it beats the seed.

## Notes

* All bounds are invariant under `rescale_layer(i, c)` (ReLU positive homogeneity), which
  is used post-hoc to clear denominators of constants/generators. Chord slopes `λ = u/(u-l)`
  are invariant under this rescaling, so niceness of slopes must hold *before* rescaling.
* The search found networks where all error generators cancel exactly in the output
  (`genmat(Zout)` becomes empty) — the dependency-effect structure that makes the
  zonotope rung exact.
* The old script `debug/find_compare_examples.jl` is kept for reference only.
