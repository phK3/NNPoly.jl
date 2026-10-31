
# ==========================================================================================
# Persistent candidate registry.
#
# Every accepted / near-miss / nice-fail candidate is appended to
# `candidates.jld2` next to this file, together with its metrics, a niceness
# summary, a timestamp and optional notes. A human-readable `CANDIDATES.md` is
# regenerated on every save.
# ==========================================================================================

const REGISTRY_FILE = joinpath(@__DIR__, "candidates.jld2")
const MARKDOWN_FILE = joinpath(@__DIR__, "CANDIDATES.md")

"""
Load the candidate registry (a `Dict{Int, NamedTuple}`), or an empty dict.
"""
function load_registry()
    isfile(REGISTRY_FILE) || return Dict{Int, NamedTuple}()
    return JLD2.load_object(REGISTRY_FILE)
end

"""
Input set a record was evaluated on: the symmetric `[-c, c]` (stored as
`input_c`), falling back to the default `[-1, 1]` for older records.
"""
function input_set_of(c)
    c_in = haskey(c, :input_c) ? c.input_c : 1.0
    return Hyperrectangle(low=[-c_in], high=[c_in])
end

"""
Append a candidate to the registry. `m` must be a NamedTuple of metrics
containing at least `Ws`, `bs` and the bound values.

The whole registry file is rewritten on every save (fine for the ~10² entries
this search produces); `md_every` throttles the `CANDIDATES.md` regeneration.

Returns the index of the new entry.
"""
function save_candidate!(m::NamedTuple; tag=:accepted, notes="", failed_links=Symbol[],
                         niceness=nothing, trial=nothing, seed=nothing, md_every=10)
    reg = load_registry()
    idx = isempty(reg) ? 1 : maximum(keys(reg)) + 1
    rec = merge(
        (idx = idx, tag = tag, notes = notes, failed_links = failed_links,
         niceness = niceness, saved = string(now()), trial = trial, seed = seed),
        m,
    )
    reg[idx] = rec
    JLD2.save_object(REGISTRY_FILE, reg)
    (idx % md_every == 0) && export_markdown()
    return idx
end

"""
Retrieve a stored candidate record by index.
"""
function get_saved_candidate(idx)
    reg = load_registry()
    haskey(reg, idx) || error("No saved candidate with index $idx")
    return reg[idx]
end

"""
List all stored candidates as a vector of records (newest last).
"""
list_candidates(; tag=nothing) = isempty(load_registry()) ? NamedTuple[] :
    [reg for (idx, reg) in sort(collect(load_registry()); by=k -> k[1]) if isnothing(tag) || reg.tag == tag]

"""
Delete a candidate from the registry (e.g. to prune the near-miss list).
"""
function delete_candidate!(idx)
    reg = load_registry()
    haskey(reg, idx) || error("No saved candidate with index $idx")
    delete!(reg, idx)
    JLD2.save_object(REGISTRY_FILE, reg)
    export_markdown()
    return nothing
end

"""
Re-evaluate the zonotope bound (`ub_zono`) of every stored record with the
*current* zonotope implementation and re-classify the ladder link status.

The stored `ub_zono` values are snapshots of the implementation at save time;
run this after a change to the zonotope relaxation (e.g. the generator fix that
gives every crossing neuron its own error generator). The tag is re-derived
uniformly: ladder-perfect records keep their `:accepted`/`:nice_fail`
distinction from the stored niceness report, one failed link is `:near_miss`,
two or more are `:rejected` (kept in the registry, not deleted).

Returns the number of records whose stored values changed.
"""
function refresh_zono!(; tol=1e-2, verbose=true)
    reg = load_registry()
    n_changed = 0
    for (idx, c) in sort(collect(reg); by=k -> k[1])
        iset = input_set_of(c)
        ub_zono = Float64(zono_upper_bound(make_network(c.Ws, c.bs), iset)[1])
        m = (ub_box=c.ub_box, ub_outer0=c.ub_outer0, ub_sep0=c.ub_sep0,
             ub_fv=c.ub_fv, ub_zono=ub_zono, ub_acrown0=c.ub_acrown0,
             ub_pcrown0=c.ub_pcrown0, ub_acrown_opt=c.ub_acrown_opt,
             ub_pcrown_opt=c.ub_pcrown_opt)
        tag, failed = classify_ladder(m; tol=tol)
        tag = tag == :accepted && !isnothing(c.niceness) && c.niceness.n_violations > 0 ?
              :nice_fail : tag
        changed = abs(ub_zono - Float64(c.ub_zono)) > 1e-9 || tag != c.tag || failed != c.failed_links
        if changed
            n_changed += 1
            if verbose
                println("#$idx: ub_zono ", round(Float64(c.ub_zono); sigdigits=5), " -> ",
                        round(ub_zono; sigdigits=5), "   tag ", c.tag, " -> ", tag,
                        isempty(failed) ? "" : "   failed: ", join(string.(failed), ", "))
            end
        end
        reg[idx] = merge(c, (ub_zono=ub_zono, tag=tag, failed_links=failed))
    end
    JLD2.save_object(REGISTRY_FILE, reg)
    export_markdown()
    verbose && println("updated ", n_changed, " of ", length(reg), " records")
    return n_changed
end

"""
Re-derive the ladder classification of every stored record from its *stored*
bounds, under the current link definitions in `constraints.jl`.

Run this after changing the ladder semantics (e.g. replacing the
`fvzono_gt_crown` link by `outer0_gt_crown`). With `niceness=true` the stored
niceness report is also recomputed with the current `collect_niceness`
(fixes records saved before the zonotope generator fix, whose stored reports
are stale).

Returns `(n_retagged, n_niceness_changed)`.
"""
function refresh_ladder!(; tol=1e-2, niceness=false, verbose=true)
    reg = load_registry()
    n_retagged = 0
    n_nice_changed = 0
    for (idx, c) in sort(collect(reg); by=k -> k[1])
        m = (ub_box=c.ub_box, ub_outer0=c.ub_outer0, ub_sep0=c.ub_sep0,
             ub_fv=c.ub_fv, ub_zono=c.ub_zono, ub_acrown0=c.ub_acrown0,
             ub_pcrown0=c.ub_pcrown0, ub_acrown_opt=c.ub_acrown_opt,
             ub_pcrown_opt=c.ub_pcrown_opt)
        tag, failed = classify_ladder(m; tol=tol)
        nice = c.niceness
        if niceness
            newnice = niceness_report(c.Ws, c.bs, input_set_of(c))
            if newnice.n_violations != nice.n_violations ||
               newnice.max_den_seen != nice.max_den_seen
                n_nice_changed += 1
            end
            nice = newnice
        end
        tag = tag == :accepted && !isnothing(nice) && nice.n_violations > 0 ? :nice_fail : tag
        if tag != c.tag || failed != c.failed_links
            n_retagged += 1
            verbose && println("#$idx: tag ", c.tag, " -> ", tag,
                               isempty(failed) ? "" : "   failed: ", join(string.(failed), ", "))
        end
        reg[idx] = merge(c, (tag=tag, failed_links=failed, niceness=nice))
    end
    JLD2.save_object(REGISTRY_FILE, reg)
    export_markdown()
    verbose && println("retagged ", n_retagged, " of ", length(reg), " records; niceness changed for ",
                       n_nice_changed)
    return (n_retagged=n_retagged, n_niceness_changed=n_nice_changed)
end

const _TAG_LABELS = Dict(:accepted => "ACCEPTED", :near_miss => "NEAR-MISS", :nice_fail => "NICE-FAIL")

"""
Pretty-print a stored candidate: network parameters, all bounds, ladder links,
niceness summary and notes.
"""
function print_candidate(idx; io=stdout)
    c = get_saved_candidate(idx)
    println(io, "="^78)
    println(io, "Candidate #", idx, "  [", get(_TAG_LABELS, c.tag, string(c.tag)), "]",
            "  saved ", c.saved, (isnothing(c.trial) ? "" : "  (trial $(c.trial))"))
    println(io, "-"^78)
    for i in 1:3
        println(io, "W", i, " = ", c.Ws[i])
        println(io, "b", i, " = ", c.bs[i])
    end
    println(io, "-"^78)
    iset = input_set_of(c)
    println(io, "Input set: [", low(iset)[1], ", ", high(iset)[1], "]")
    println(io, "Bounds (looser > tighter):")
    println(io, "  (i)   interval            : ", c.ub_box)
    println(io, "  (ii)  sym. interval       : ", c.ub_outer0)
    println(io, "  (iii) sym. interval (sep.): ", c.ub_sep0)
    println(io, "  (iii')     + fresh vars   : ", c.ub_fv, "   [DeepPolyRelax: ", c.ub_fv_dp, "]")
    println(io, "  (iv)  zonotope            : ", c.ub_zono)
    println(io, "  (v)   CROWN (0 steps)     : ", c.ub_acrown0)
    println(io, "  (vi)  NNPoly (0 steps)    : ", c.ub_pcrown0)
    println(io, "  (vii) aCROWN (optimized)  : ", c.ub_acrown_opt)
    println(io, "  (viii)NNPoly (optimized)  : ", c.ub_pcrown_opt)
    if !isempty(c.failed_links)
        println(io, "Failed ladder links: ", join(c.failed_links, ", "))
    end
    if !isnothing(c.niceness)
        n = c.niceness
        println(io, "Niceness: ", n.n_violations, "/", n.n_items, " violations, ",
                "max denominator = ", n.max_den_seen, ", score = ", round(n.score, digits=1))
        for w in n.worst
            println(io, "   ✗ ", rpad(w.name, 30), frac_str(w.value), "  (den = ", w.den, ")")
        end
    end
    if !isempty(c.notes)
        println(io, "Notes: ", c.notes)
    end
    println(io, "="^78)
    return c
end

"""
Regenerate the human-readable `CANDIDATES.md` from the registry.
"""
function export_markdown(path=MARKDOWN_FILE)
    reg = load_registry()
    open(path, "w") do io
        println(io, "# Candidate networks for the thesis running example")
        println(io)
        println(io, "Auto-generated by `debug/example_search`. Orderings (looser > tighter):")
        println(io, "interval > sym. interval > sym. interval (sep.) > {fresh vars, zonotope};")
        println(io, "sym. interval > CROWN(0) > NNPoly(0) (CROWN(0) vs. sep./fv/zono: unconstrained).")
        println(io)
        println(io, "| idx | tag | input | box | outer0 | sep0 | fv | fv(dp) | zono | aCROWN0 | pCROWN0 | aCROWN opt | pCROWN opt | max den |")
        println(io, "|----:|-----|------:|----:|-------:|-----:|---:|-------:|-----:|--------:|--------:|-----------:|-----------:|--------:|")
        for idx in sort(collect(keys(reg)))
            c = reg[idx]
            md = isnothing(c.niceness) ? "—" : string(c.niceness.max_den_seen)
            c_in = haskey(c, :input_c) ? c.input_c : 1.0
            println(io, "| ", idx, " | ", get(_TAG_LABELS, c.tag, string(c.tag)), " | ", c_in, " | ",
                    join([c.ub_box, c.ub_outer0, c.ub_sep0, c.ub_fv, c.ub_fv_dp, c.ub_zono,
                          c.ub_acrown0, c.ub_pcrown0, c.ub_acrown_opt, c.ub_pcrown_opt], " | "),
                    " | ", md, " |")
        end
        println(io)
        println(io, "Details: `ExampleSearch.print_candidate(idx)`, network parameters are in the JLD2 registry.")
    end
    return path
end
