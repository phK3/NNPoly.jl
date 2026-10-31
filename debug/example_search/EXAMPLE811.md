# Candidate: registry entry 811  (refinement of #808 via polish, job c334125b)

Worst items: zono_layer3_pre_c = -107//6 (largest numerator, den <= 10).
Backups: example811.jld2 / EXAMPLE811.md / propagation_idx811.tex

==============================================================================
Candidate #811  [ACCEPTED]  saved 2026-10-31T13:19:39.874
------------------------------------------------------------------------------
W1 = [-0.75; -7.0;;]
b1 = [0.25, 1.0]
W2 = [-24.0 1.75; -56.0 7.0]
b2 = [12.0, -8.0]
W3 = [-1.0 0.75]
b3 = [-10.5]
------------------------------------------------------------------------------
Input set: [-1.0, 1.0]
Bounds (looser > tighter):
  (i)   interval            : 25.5
  (ii)  sym. interval       : 0.8333333333333339
  (iii) sym. interval (sep.): -0.09999999999999964
  (iii')     + fresh vars   : -4.0   [DeepPolyRelax: -7.5]
  (iv)  zonotope            : -7.16666666666667
  (v)   CROWN (0 steps)     : -3.5
  (vi)  NNPoly (0 steps)    : -6.7307692307692335
  (vii) aCROWN (optimized)  : -6.5062543582853305
  (viii)NNPoly (optimized)  : -11.701875006327203
Niceness: 0/123 violations, max denominator = 10, score = 116.6999999999999999999999999999999999999999999999999999999999999999999999999996
Notes: refineShift arm: polished from delta-2.5-shifted #808 (job c334125b, seed 3002)
==============================================================================
