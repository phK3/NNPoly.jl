# Candidate: registry entry 802  (polish run job d63c919c, arm split-W1)
Warm-start W1[1]+=1.0 arm best: ladder-perfect, 2 niceness violations (max den 38).

==============================================================================
Candidate #802  [NICE-FAIL]  saved 2026-10-30T22:43:39.671
------------------------------------------------------------------------------
W1 = [-320.0; -0.75;;]
b1 = [64.0, 0.25]
W2 = [-0.078125 12.0; -0.390625 120.0]
b2 = [12.0, -8.0]
W3 = [-1.0625 0.5]
b3 = [-1.5]
------------------------------------------------------------------------------
Input set: [-1.0, 1.0]
Bounds (looser > tighter):
  (i)   interval            : 54.5
  (ii)  sym. interval       : 26.0
  (iii) sym. interval (sep.): 20.72222222222222
  (iii')     + fresh vars   : 14.416666666666664   [DeepPolyRelax: 16.875]
  (iv)  zonotope            : 14.000000000000007
  (v)   CROWN (0 steps)     : 13.5
  (vi)  NNPoly (0 steps)    : 10.386627906976743
  (vii) aCROWN (optimized)  : 11.45634324086077
  (viii)NNPoly (optimized)  : 9.647601621391074
Niceness: 2/123 violations, max denominator = 38, score = 206.5
   ✗ acrown0_Λ                     -489//38  (den = 38)
   ✗ sep0_U[2]                     629//36  (den = 36)
Notes: polish-best from warm-start
==============================================================================
