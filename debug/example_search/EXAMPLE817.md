# Candidate: registry entry 817  (thesis flagship: output-bias-shifted #816)

b3 -> b3 - 8 (7 -> -1), all other weights identical: every upper bound shifts by exactly -8,
all 9 ladder links strict (margins are shift-invariant), 0/123 niceness violations,
max denominator 7, all-positive bound ladder on rungs (i)-(vii).
Rung (viii) NNPoly (optimised slopes) is the ONLY rung whose upper bound is <= 0
(-0.817545... = -45991/56255); rung (vi) NNPoly (0 steps) is 36/13 = 2.769... > 0.
So the property y <= 0 on [-1, 1] is provable only by the final NNPoly polynomials.

==============================================================================
Candidate #817  [ACCEPTED]  saved 2026-11-02T16:22:05.784
------------------------------------------------------------------------------
W1 = [-0.75; -14.0;;]
b1 = [0.25, 2.0]
W2 = [-48.0 1.75; -28.0 1.75]
b2 = [24.0, -4.0]
W3 = [-0.5 1.5]
b3 = [-1.0]
------------------------------------------------------------------------------
Input set: [-1.0, 1.0]
Bounds (looser > tighter):
  (i)   interval            : 35.0
  (ii)  sym. interval       : 10.333333333333334
  (iii) sym. interval (sep.): 9.4
  (iii')     + fresh vars   : 5.500000000000001   [DeepPolyRelax: 2.0]
  (iv)  zonotope            : 2.333333333333332
  (v)   CROWN (0 steps)     : 6.0
  (vi)  NNPoly (0 steps)    : 2.7692307692307683
  (vii) aCROWN (optimized)  : 2.9937456417146677
  (viii)NNPoly (optimized)  : -0.8175451078157225
Niceness: 0/123 violations, max denominator = 7, score = 96.7
Notes: output-bias shift of #816: b3 -> b3 - 8 (7 -> -1), everything else identical. Every upper bound shifts by exactly -8, so rungs (i)-(vii) stay positive and only rung (viii) NNPoly(optimized) = -0.8175 <= 0 proves y <= 0. Denominators unchanged vs #816: 0/123 violations, max_den = 7. Intended thesis flagship.
==============================================================================
