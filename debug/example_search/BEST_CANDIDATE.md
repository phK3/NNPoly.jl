# Thesis flagship: #817 (= #816 with output bias shifted b3 → b3 − 8); companions: #813 (num-optimal), #816 (den-optimal)

(frozen 2026-10-31, after refinement rounds 2–3; #817 added 2026-11-02)

## #817 — thesis flagship: property y ≤ 0 provable only by rung (viii)

Registered 2026-11-02 by shifting **#816**'s output bias `b3 → b3 − 8`
(+7 → −1); all other weights identical. Every upper bound shifts by exactly
−8, so:

* all 9 ladder links stay strict (margins are shift-invariant),
* denominators are unchanged (integer shift): 0/123 violations, max den 7,
* rungs (i)–(vii) all keep a positive upper bound, and only rung **(viii)**
  NNPoly (optimised slopes) crosses zero: 35 | 31/3 | 47/5 | 11/2 | 7/3 |
  6 | 36/13 | 133069/44449 | **−45991/56255 ≤ 0**.

So `y ≤ 0` on `[-1, 1]` is provable by the final NNPoly polynomials only —
rung (vi) NNPoly(0) still gives +36/13.

```
W1 = [-0.75; -14.0]     b1 = [0.25, 2.0]
W2 = [-48.0 1.75        b2 = [24.0, -4.0]
      -28.0 1.75]
W3 = [-0.5 1.5]         b3 = [-1.0]          input x ∈ [-1, 1]
```

Both #813 and #816 below are `:accepted`, all 9 ladder links strict,
niceness 0/123 violations, 123/123 items denominator ≤ 8, and both are
**proven frontier vertices** of the same network family under the b3-shift
lever (see "Frontier" below).

## #813 — numerator-optimal: (0, 0, 10, 76)

Best worst-numerator in the registry: **76** (`outer0_L[2] = −76/3`),
max denominator 10, 120/123 items ≤ 8. Bound ladder partially negative
(output is negative on the input set).

```
W1 = [-0.75; -7.0]      b1 = [0.25, 1.0]
W2 = [-24.0 1.75        b2 = [12.0, -8.0]
      -56.0 7.0]
W3 = [-1.0 0.75]        b3 = [-3.5]          input x ∈ [-1, 1]
```

## #816 — denominator-optimal: (0, 0, 7, 89)

Max denominator **7**, max numerator **89**, all 123 items ≤ 8, and — like
#814 before it — a fully positive bound ladder:
43 | 55/3 → 17.4 → 13.5 → 10.0 | 10.33 | 14.0 | 10.77 | 10.994 | 7.182.
(Also the smallest integer-friendly b3 = +7.)

```
W1 = [-0.75; -14.0]     b1 = [0.25, 2.0]
W2 = [-48.0 1.75        b2 = [24.0, -4.0]
      -28.0 1.75]
W3 = [-0.5 1.5]         b3 = [+7.0]          input x ∈ [-1, 1]
```

## Frontier (b3-shift lever, exact sweep)

The two states are the vertices of two b3-shift ridges of one family
(#813 shifted by dyadic δ):

* den ≤ 10 ridge: min numerator 76 at b3 = −3.5 (#813)
* den ≤ 7 ridge: min numerator 89 at b3 = +7 (#816); the ridge degrades
  by +6 numerator per half-shift in *both* directions (measured over
  δ ∈ [−6, 25])

Refinement rounds: 2×20k-step SA from #813 → no improvement (robust local
opt); 2×20k-step SA from #814 with a den-7 hard bar → both arms converged
to exactly (0,0,7,89) = #816.

## Lineage

#651 → tie-split `b1[2] -= 1/4` → SA (d63c919c) → #808 → SA refinement
(c334125b, arms 1/2) → #810/#811 → b3-shift sweeps (+12/+7) → #812/#813 →
#813 + δ=+10.5 with SA rescale round (a79c61e5) → **#816**.

## Files

Per entry N ∈ {813, 816, 817}: `exampleN.jld2` (registry record), `EXAMPLEN.md`
(dump), `propagation_idxN.tex` (exact propagation table; #817's summary table
also lists the optimised rungs (vii)/(viii)). #814 (0,0,7,95)
and #812 (#0,0,10,89) remain as registered alternates.
