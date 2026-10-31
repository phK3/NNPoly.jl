# Candidate: registry entry 651  (corrected-ladder run, seed 203)

Near-miss under the corrected ladder: fails only crown_gt_pcrown (exact tie (v) = (vi) = 93/7).
Niceness: 122/123 items within max_den = 32 (115 items have den <= 8);
single offender: fv_L[2] = -187/45 (den 45). Backups: example651.jld2 / EXAMPLE651.md / propagation_idx651.tex.

==============================================================================
Candidate #651  [NEAR-MISS]  saved 2026-10-30T21:16:16.533  (trial 150790)
------------------------------------------------------------------------------
W1 = [-1.5; -1.5;;]
b1 = [0.5, 0.5]
W2 = [-6.0 3.0; -12.0 16.0]
b2 = [4.0, 0.0]
W3 = [-1.0 1.0]
b3 = [1.0]
------------------------------------------------------------------------------
Input set: [-1.0, 1.0]
Bounds (looser > tighter):
  (i)   interval            : 33.0
  (ii)  sym. interval       : 19.666666666666664
  (iii) sym. interval (sep.): 18.333333333333336
  (iii')     + fresh vars   : 17.0   [DeepPolyRelax: 11.0]
  (iv)  zonotope            : 15.666666666666666
  (v)   CROWN (0 steps)     : 13.285714285714285
  (vi)  NNPoly (0 steps)    : 13.285714285714285
  (vii) aCROWN (optimized)  : 10.5758288473492
  (viii)NNPoly (optimized)  : 9.129976846053909
Failed ladder links: crown_gt_pcrown
Niceness: 1/123 violations, max denominator = 45, score = 139.5
   ✗ fv_L[2]                       -187//45  (den = 45)
Notes: near-miss (auto-rescaled as rescaled)
==============================================================================
