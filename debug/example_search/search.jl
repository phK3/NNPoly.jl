
# ==========================================================================================
# Staged random search for candidate networks.
#
# Stages (cheap first, so most candidates are rejected within microseconds):
#   1. forward rungs (i)–(iv) + forward ladder links
#   2. CROWN(0)/NNPoly(0) rungs (v)/(vi) + their ladder links
#   3. niceness analysis of all intermediate quantities
#   4. optimized rungs (vii)/(viii) + final ladder classification
#
# Candidates that pass everything are saved as :accepted, candidates failing
# exactly one ladder link as :near_miss, and ladder-perfect candidates with
# ugly intermediate numbers as :nice_fail.
#
# For accepted/near-miss candidates, a denominator-clearing layer rescaling
# (which preserves all bounds) is applied before saving.
# ==========================================================================================

"""
Pick the rescale variant (including the identity) with the best niceness score.
All variants describe the *same* network function, so bounds are unchanged.
"""
function best_rescale(Ws, bs, input_set; max_den=32)
    best = (Ws=copy(Ws), bs=copy(bs), s=niceness_report(Ws, bs, input_set; max_den=max_den))
    for (Wsᵥ, bsᵥ, name) in rescale_variants(Ws, bs)
        s = niceness_report(Wsᵥ, bsᵥ, input_set; max_den=max_den)
        if s.score < best.s.score
            best = (Ws=Wsᵥ, bs=bsᵥ, s=s)
        end
    end
    return best
end

"""
    fuzz(; kwargs...)

Staged random search for running-example networks.

Keyword arguments:
* `n_trials`   — number of sampled networks
* `seed`       — RNG seed (stored in the registry for reproducibility)
* `tol`        — minimum margin required by every ladder link
* `n_steps`    — optimisation steps for rungs (vii)/(viii)
* `max_den`    — maximum denominator for "nice" fractions
* `h1`, `h2`   — hidden layer widths (default 2, 2)
* `weights`, `biases`, `b3_pool` — sampling alphabets
* `wide_alphabet` — use the widened dyadic alphabets (`WIDE_WEIGHTS`/`WIDE_BIASES`/
                    `WIDE_B3`); ignored when `weights`/`biases` are given explicitly
* `structured_W1` — bias the first-layer weights towards cancellation-friendly patterns
* `nice_bias`   — keep layer 1 to a single crossing-capable neuron (others one-sided),
                  reducing denominators of all downstream quantities
* `input_widths` — list of input half-widths c; each trial uses the symmetric set
                   [-c, c] with c drawn uniformly from this list (default [1.0]).
                   The drawn c is stored per candidate as `input_c`.
* `save`       — persist found candidates to the registry
* `max_nice_fail` — number of :nice_fail candidates to keep, ranked by denominator
* `max_near_miss` — number of :near_miss candidates to keep, ranked by denominator
                    (the best candidate of each distinct failed link is always kept)
* `progress_cb(trial, n_trials, stats)` — called every `progress_every` trials
* `stop_cb()`  — return true to stop the search cooperatively
"""
function fuzz(; n_trials=10_000, seed=1, tol=1e-2, n_steps=25, max_den=32,
              h1=2, h2=2, weights=nothing, biases=nothing, b3_pool=nothing,
              structured_W1=false, nice_bias=false, wide_alphabet=false,
              input_widths=[1.0],
              save=true, max_nice_fail=50, max_near_miss=150,
              progress_cb=nothing, stop_cb=nothing, progress_every=500)
    rng = MersenneTwister(seed)
    weights = isnothing(weights) ? (wide_alphabet ? WIDE_WEIGHTS : DEFAULT_WEIGHTS) : weights
    biases  = isnothing(biases)  ? (wide_alphabet ? WIDE_BIASES  : DEFAULT_BIASES)  : biases
    b3_pool = isnothing(b3_pool) ? (wide_alphabet ? WIDE_B3 : [-0.5, 0.0, 0.5]) : b3_pool

    stats = Dict{Symbol, Int}(:trials => 0, :stage1_reject => 0, :stage2_reject => 0,
                              :nice_fail => 0, :accepted => 0, :near_miss => 0,
                              :near_miss_overflow => 0, :opt_reject => 0,
                              :nice_fail_pool => 0, :near_miss_pool => 0)
    link_counts = Dict{Symbol, Int}()
    # candidates are pooled in memory and ranked by denominator quality at the
    # end of the run (instead of storing the first ones found)
    nice_pool = Vector{NamedTuple}()
    nm_pool = Vector{NamedTuple}()

    for trial in 1:n_trials
        !isnothing(stop_cb) && stop_cb() && break

        input_c = Float64(rand(rng, input_widths))
        input_set = Hyperrectangle(low=[-input_c], high=[input_c])
        Ws, bs = sample_candidate(rng; h1=h1, h2=h2, weights=weights, biases=biases,
                                  b3_pool=b3_pool,
                                  structured_W1=structured_W1, nice_bias=nice_bias)
        stats[:trials] += 1

        ### stage 1: forward rungs
        m_fwd = evaluate_forward(Ws, bs, input_set)
        fwd_links = forward_links(m_fwd; tol=tol)
        n_fail_fwd = count(!ok for ok in values(fwd_links))
        if n_fail_fwd >= 2
            stats[:stage1_reject] += 1
            _count_links!(link_counts, fwd_links)
            continue
        end

        ### stage 2: CROWN(0) / NNPoly(0)
        m = merge(m_fwd, evaluate_crown_init(Ws, bs, input_set))
        links = early_links(m; tol=tol)
        n_fail_early = count(!ok for ok in values(links))
        if n_fail_early >= 2
            stats[:stage2_reject] += 1
            _count_links!(link_counts, links)
            continue
        end

        ### stage 3: niceness of all intermediate quantities
        nice = niceness_report(Ws, bs, input_set; max_den=max_den)

        ### stage 4: optimized rungs (only worth it for survivors)
        m = merge(m, evaluate_crown_opt(Ws, bs, input_set; n_steps=n_steps))
        tag, failed = classify_ladder(m; tol=tol)
        _count_links!(link_counts, ladder_links(m; tol=tol))

        if tag == :rejected
            stats[:opt_reject] += 1
        elseif tag == :near_miss
            # kept regardless of niceness (so nothing is lost — e.g. candidate-14-style nets);
            # pooled, ranked by denominator, best max_near_miss saved at the end
            stats[:near_miss] += 1
            save && push!(nm_pool, (Ws=Ws, bs=bs, m=m, input_c=input_c, trial=trial,
                                    failed=failed, nice=nice))
        elseif nice.n_violations > 0
            # ladder-perfect but ugly intermediate numbers: pooled, ranked by
            # denominator, best max_nice_fail saved at the end
            stats[:nice_fail] += 1
            save && push!(nice_pool, (Ws=Ws, bs=bs, m=m, input_c=input_c, trial=trial,
                                      nice=nice))
        else
            stats[:accepted] += 1
            if save
                r = best_rescale(Ws, bs, input_set; max_den=max_den)
                save_candidate!(merge((Ws=r.Ws, bs=r.bs, input_c=input_c), m); tag=:accepted, notes="",
                                failed_links=failed, niceness=r.s, trial=trial, seed=seed)
            end
        end

        if !isnothing(progress_cb) && trial % progress_every == 0
            progress_cb(trial, n_trials, stats)
        end
    end

    if save
        counts = _save_ranked_pools!(nice_pool, nm_pool; max_nice_fail=max_nice_fail,
                                     max_near_miss=max_near_miss, max_den=max_den, seed=seed)
        stats[:nice_fail_pool] = counts.n_nice
        stats[:near_miss_pool] = counts.n_nm
        export_markdown()   # final CANDIDATES.md refresh
    end

    return (stats=stats, link_counts=link_counts, seed=seed, n_trials=n_trials)
end

"""
Rank the pooled candidates by denominator quality and save the best ones.

Ranking key: `max_den_seen` (smaller is better), tie-broken by the number of
niceness violations and the aggregate score. Near-misses are selected with
diversity: the best candidate of each distinct failed link type is kept first,
remaining slots go to the globally best-ranked ones.

Ranking uses the raw (pre-rescale) niceness report; the auto-rescale is applied
afterwards to the selected near-misses, so stored denominators only improve on
the ranked ones.
"""
function _save_ranked_pools!(nice_pool, nm_pool; max_nice_fail, max_near_miss,
                             max_den, seed)
    rank_key(r) = (r.nice.max_den_seen, r.nice.n_violations, r.nice.score)

    ### nice-fail pool: globally best-ranked ladder-perfect candidates (stored raw,
    ### as before — this tag is not rescaled)
    nice_ranked = sort(nice_pool; by=rank_key)
    n_nice = min(max_nice_fail, length(nice_ranked))
    for r in nice_ranked[1:n_nice]
        save_candidate!(merge((Ws=r.Ws, bs=r.bs, input_c=r.input_c), r.m);
                        tag=:nice_fail, failed_links=Symbol[], niceness=r.nice,
                        trial=r.trial, seed=seed)
    end

    ### near-miss pool: best per distinct failed link first, then fill by ranking
    nm_ranked = sort(nm_pool; by=rank_key)
    chosen = Set{Int}()
    seen_links = Set{Symbol}()
    for (i, r) in enumerate(nm_ranked)
        l = isempty(r.failed) ? :none : r.failed[1]
        if !(l in seen_links)
            push!(seen_links, l)
            push!(chosen, i)
        end
    end
    for i in eachindex(nm_ranked)
        length(chosen) >= max_near_miss && break
        push!(chosen, i)   # sets deduplicate
    end
    n_nm = 0
    for i in sort(collect(chosen))[1:min(max_near_miss, end)]
        r = nm_ranked[i]
        input_set = Hyperrectangle(low=[-r.input_c], high=[r.input_c])
        res = best_rescale(r.Ws, r.bs, input_set; max_den=max_den)
        save_candidate!(merge((Ws=res.Ws, bs=res.bs, input_c=r.input_c), r.m);
                        tag=:near_miss,
                        notes="near-miss (auto-rescaled as $(rescale_note(r.Ws, r.bs, res.Ws)))",
                        failed_links=r.failed, niceness=res.s, trial=r.trial, seed=seed)
        n_nm += 1
    end
    return (n_nice=n_nice, n_nm=n_nm)
end

function _count_links!(link_counts, links)
    for (name, ok) in pairs(links)
        !ok && (link_counts[name] = get(link_counts, name, 0) + 1)
    end
    return link_counts
end

function rescale_note(Ws, bs, Ws_new)
    Ws == Ws_new ? "identity" : "rescaled"
end

"""
Quick pilot run (no saving) to estimate per-stage yield before a long search.
"""
function pilot(; n_trials=2000, kwargs...)
    t = @elapsed result = fuzz(; n_trials=n_trials, save=false, kwargs...)
    s = result.stats
    n = max(s[:trials], 1)
    println("Pilot: $n trials in $(round(t, digits=1)) s")
    println("  stage-1 rejects : ", s[:stage1_reject], "  (", round(100*s[:stage1_reject]/n, digits=1), "%)")
    println("  stage-2 rejects : ", s[:stage2_reject], "  (", round(100*s[:stage2_reject]/n, digits=1), "%)")
    println("  opt rejects     : ", s[:opt_reject],     "  (", round(100*s[:opt_reject]/n, digits=1), "%)")
    println("  accepted        : ", s[:accepted],      "  (", round(100*s[:accepted]/n, digits=1), "%)")
    println("  near-miss       : ", s[:near_miss],     "  (", round(100*s[:near_miss]/n, digits=1), "%)")
    s[:near_miss_overflow] > 0 && println("    (near-miss cap overflow: ", s[:near_miss_overflow], " not saved)")
    println("  nice-fail       : ", s[:nice_fail],     "  (", round(100*s[:nice_fail]/n, digits=1), "%)")
    if !isempty(result.link_counts)
        println("  failed-link distribution:")
        for (name, cnt) in sort(collect(result.link_counts); by=x -> -x[2])
            println("    ", name, ": ", cnt)
        end
    end
    return result
end
