# Candidate: registry entry 809  (polish run job d63c919c, arm #451)
#451 arm best: niceness-clean (max den 32), one exact-tie link failure (v)=(vi)=64.

==============================================================================
Candidate #809  [NEAR-MISS]  saved 2026-10-30T23:16:32.185
------------------------------------------------------------------------------
W1 = [-0.0625; -6.0;;]
b1 = [0.0, 0.0]
W2 = [-48.0 1.0; 64.0 -1.0]
b2 = [6.5, 0.0]
W3 = [3.5 8.75]
b3 = [-7.75]
------------------------------------------------------------------------------
Input set: [-2.0, 2.0]
Bounds (looser > tighter):
  (i)   interval            : 127.0
  (ii)  sym. interval       : 84.0
  (iii) sym. interval (sep.): 78.0
  (iii')     + fresh vars   : 57.0   [DeepPolyRelax: 84.99999999999999]
  (iv)  zonotope            : 52.5
  (v)   CROWN (0 steps)     : 64.0
  (vi)  NNPoly (0 steps)    : 64.0
  (vii) aCROWN (optimized)  : 43.513061520724136
  (viii)NNPoly (optimized)  : 39.30096110678271
Failed ladder links: crown_gt_pcrown
Niceness: 0/121 violations, max denominator = 32, score = 111.8000000000000000000000000000000000000000000000000000000000000000000000000004
Notes: polish-best from #451
==============================================================================
