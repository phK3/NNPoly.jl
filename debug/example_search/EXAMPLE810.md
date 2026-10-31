# Candidate: registry entry 810  (refinement of #808 via polish, job c334125b)

Worst items: fv_L[2] = -124//3 (largest numerator, den <= 10).
Backups: example810.jld2 / EXAMPLE810.md / propagation_idx810.tex

==============================================================================
Candidate #810  [ACCEPTED]  saved 2026-10-31T13:19:39.600
------------------------------------------------------------------------------
W1 = [-24.0; -1.75;;]
b1 = [8.0, 0.25]
W2 = [-0.75 7.0; -0.875 14.0]
b2 = [8.0, -4.0]
W3 = [-1.5 3.75]
b3 = [-24.0]
------------------------------------------------------------------------------
Input set: [-1.0, 1.0]
Bounds (looser > tighter):
  (i)   interval            : 66.0
  (ii)  sym. interval       : 4.0000000000000036
  (iii) sym. interval (sep.): -1.5999999999999979
  (iii')     + fresh vars   : -3.9999999999999964   [DeepPolyRelax: -0.75]
  (iv)  zonotope            : -10.0
  (v)   CROWN (0 steps)     : 1.5
  (vi)  NNPoly (0 steps)    : -6.576923076923078
  (vii) aCROWN (optimized)  : -10.078775293042698
  (viii)NNPoly (optimized)  : -14.300442988671454
Niceness: 0/123 violations, max denominator = 10, score = 92.19999999999999999999999999999999999999999999999999999999999999999999999999956
Notes: refine808 arm: polished from #808 (job c334125b, below-bar objective soft_den=8 soft_num=64, seed 3001)
==============================================================================
