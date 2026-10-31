# Flagship candidate: registry entry 31

Backed up before the ranked-pool search (seed 202). Full registry snapshot: candidates_backup_400.jld2; standalone record: example31.jld2; TeX: propagation_idx31.tex.

==============================================================================
Candidate #31  [NICE-FAIL]  saved 2026-10-30T08:50:54.174  (trial 2330)
------------------------------------------------------------------------------
W1 = [-1.0; -2.0;;]
b1 = [0.0, 0.5]
W2 = [-0.5 0.5; -2.0 1.0]
b2 = [0.5, 0.0]
W3 = [-2.0 0.5]
b3 = [-0.5]
------------------------------------------------------------------------------
Input set: [-1.0, 1.0]
Bounds (looser > tighter):
  (i)   interval            : 0.75
  (ii)  sym. interval       : 0.033653846153846256
  (iii) sym. interval (sep.): -0.0625
  (iii')     + fresh vars   : -0.53125   [DeepPolyRelax: 0.0]
  (iv)  zonotope            : -0.5625
  (v)   CROWN (0 steps)     : 0.0
  (vi)  NNPoly (0 steps)    : -0.75
  (vii) aCROWN (optimized)  : -1.1041754999511926
  (viii)NNPoly (optimized)  : -1.188327850601669
Niceness: 17/113 violations, max denominator = 208, score = 656.5
   ✗ ub_outer0                     7//208  (den = 208)
   ✗ outer0_ubs_layer3             7//208  (den = 208)
   ✗ outer0_U[2]                   -133//208  (den = 208)
   ✗ fv_L[1]                       65//92  (den = 92)
   ✗ fv_L[2]                       -249//92  (den = 92)
   ✗ zono_layer2_pre_c[1]          49//64  (den = 64)
   ✗ zono_layer2_pre_G[3]          7//64  (den = 64)
   ✗ zono_layer2_post_c[1]         49//64  (den = 64)
   ✗ zono_layer2_post_G[3]         7//64  (den = 64)
   ✗ zono_layer3_pre_c             -121//64  (den = 64)
==============================================================================
