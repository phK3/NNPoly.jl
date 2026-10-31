
# ==========================================================================================
# The precision ladder and its constraint checking.
#
# Desired ordering of concrete upper bounds (looser > tighter):
#
#   box > outer0 > sep0 > {fv, zono}    (forward refinement chain)
#   outer0 > acrown0 > pcrown0          (CROWN(0) beats the *plain* forward
#                                        symbolic interval propagation)
#   acrown_opt < acrown0,  pcrown_opt < pcrown0,  pcrown_opt < acrown_opt
#
# I.e. the forward rungs (i)–(iv) form a strict chain; CROWN(0) is expected to
# beat the plain forward symbolic interval propagation (no fresh vars, no
# separate bounds) and PolyCROWN(0). The ordering of CROWN(0) w.r.t. the
# separated / fresh-var / zonotope rungs does *not* matter — no link compares
# them, and no link compares fv vs zono.
#
# A candidate is
#   :accepted  if all links hold with margin ≥ tol,
#   :near_miss if exactly one link is violated,
#   :rejected  otherwise.
# ==========================================================================================

"""
Links among the forward rungs only — checkable from `evaluate_forward` output.
"""
function forward_links(m; tol=1e-2)
    return (
        i_gt_ii     = m.ub_box    > m.ub_outer0 + tol,
        ii_gt_iii   = m.ub_outer0 > m.ub_sep0   + tol,
        iii_gt_fv   = m.ub_sep0   > m.ub_fv     + tol,
        iii_gt_zono = m.ub_sep0   > m.ub_zono   + tol,
    )
end

"""
Links checkable before the optimized runs: forward links plus the CROWN(0)/NNPoly(0)
links. Requires `ub_acrown0`/`ub_pcrown0` in addition to the forward metrics.
"""
function early_links(m; tol=1e-2)
    return (
        forward_links(m; tol=tol)...,
        # (ii) > (v): CROWN(0) beats the *plain* forward symbolic interval
        #             propagation (no fresh vars, no separate bounds). No
        #             ordering is imposed between CROWN(0) and the separated /
        #             fresh-var / zonotope rungs.
        outer0_gt_crown = m.ub_outer0 > m.ub_acrown0 + tol,
        # (v) > (vi)
        crown_gt_pcrown = m.ub_acrown0 > m.ub_pcrown0 + tol,
    )
end

"""
Names and definitions of all ladder links. Each link checks that the *looser*
method's bound exceeds the *tighter* method's bound by at least `tol`.
Requires all nine bounds (forward + CROWN(0) + optimized).
"""
function ladder_links(m; tol=1e-2)
    return (
        early_links(m; tol=tol)...,
        # optimized versions improve on the unoptimized ones
        acrown_opt_improves = m.ub_acrown_opt < m.ub_acrown0 - tol,
        pcrown_opt_improves = m.ub_pcrown_opt < m.ub_pcrown0 - tol,
        # NNPoly(opt) beats CROWN(opt)
        pcrown_opt_beats_acrown_opt = m.ub_pcrown_opt < m.ub_acrown_opt - tol,
    )
end

"""
Classify a candidate against the ladder.

Returns `(tag, failed_links)` with tag ∈ (:accepted, :near_miss, :rejected).
"""
function classify_ladder(m; tol=1e-2)
    links = ladder_links(m; tol=tol)
    failed = Symbol[name for (name, ok) in pairs(links) if !ok]
    if isempty(failed)
        return (:accepted, Symbol[])
    elseif length(failed) == 1
        return (:near_miss, failed)
    else
        return (:rejected, failed)
    end
end

"""
Only the links that can be checked *before* the (expensive) optimized runs.
"""
function early_links_ok(m; tol=1e-2)
    return all(early_links(m; tol=tol))
end

"""
Number of violated early links (used for cheap early rejection in the search).
"""
function count_failed_early(m; tol=1e-2)
    return sum(1 for ok in early_links(m; tol=tol) if !ok)
end
