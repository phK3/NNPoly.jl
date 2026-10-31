
# ==========================================================================================
# Local polish: simulated-annealing / basin-hopping refinement of registry seeds.
#
# Global sampling (`fuzz`) has shown that ladder-perfect AND nice candidates are
# extremely rare jointly, but the pools contain seeds that are only one or two
# steps away (e.g. #651: an exact tie (v) = (vi) and a single niceness offender,
# fv_L[2] = -187//45). This module searches the *local* dyadic neighborhood of
# such seeds with simulated annealing, minimizing a balanced objective that
# measures the distance to the target state — ladder-perfect AND nice:
#
#     S = 30 · (failed ladder links) + 12 · (niceness violations)
#         + 8 · log₂(max denominator / max_den)₊ + 3 · log₂(max numerator / max_den)₊
#
# All four weighted penalties are non-negative; smaller is better. Raw-key
# components are still reported (see `polish_key` in the return value).
#
# Moves (all preserve the dyadic alphabet, keeping the exact-rational niceness
# analysis valid):
#   * weight/bias nudge — an entry grows/shrinks by a small dyadic δ
#   * neuron rescale    — pre-activations of ONE hidden neuron scaled by a power
#                         of two (ReLU is positively homogeneous: the network
#                         function and all bounds are unchanged)
#   * layer rescale     — `rescale_layer` of one hidden layer (ditto)
#   * kick              — 2–4 moves at once (basin hopping)
#
# Evaluation is staged like `fuzz`: a move is discarded right after the early
# links when its failure count exceeds the current corridor (the optimized
# rungs and niceness analysis are skipped). A corridor of +1 link failure is
# kept, so paths that *briefly* break a link while clearing denominators can
# still be explored.
# ==========================================================================================

const _POLISH_NUDGES = (0.03125, 0.03125, 0.0625, 0.0625, 0.125, 0.125,
                        0.25, 0.5, 1.0, 2.0)
const _POLISH_RESCALES = (0.5, 2.0)

const _POLISH_W_LINK  = 30.0   # per failed ladder link
const _POLISH_W_VIOL  = 12.0   # per niceness violation
const _POLISH_W_DEN   = 8.0    # per doubling of the worst denominator beyond the bar
const _POLISH_W_NUM   = 3.0    # per doubling of the worst numerator above the bar

"""
Largest numerator among all tracked niceness items (exact recovery via
`frac_of`), as a plain `Int`. Hand-checkability cares about numerator
magnitudes, not just denominators.
"""
function _polish_max_num(items)
    isempty(items) && return 0
    # saturate: exact-rational numerators can exceed Int64 after dyadic
    # nudges, and anything beyond 10^9 is equally "ugly" for the objective
    m = maximum(abs(numerator(frac_of(i.value))) for i in items)
    return Int(min(m, 10^9))
end

"""
Niceness with items retained: fresh `collect_niceness` + summary + max numerator.
"""
function _polish_niceness(Ws, bs, input_set; max_den)
    items = collect_niceness(Ws, bs, input_set)
    return (summary = niceness_summary(items; max_den = max_den),
            items = items, max_num = _polish_max_num(items))
end

"""
Balanced polish objective (smaller = closer to the target: ladder-perfect AND
nice, everything hand-checkable). The raw key is
`(k1, k2, max_den, max_num)` = (failed links, niceness violations, worst
denominator, worst numerator).
"""
function polish_objective(k; max_den=32)
    s = _POLISH_W_LINK * k[1] + _POLISH_W_VIOL * k[2]
    s += k[3] > max_den ? _POLISH_W_DEN * (log(2, k[3]) - log(2, max_den)) : 0.0
    s += k[4] > max_den ? _POLISH_W_NUM * (log(2, k[4]) - log(2, max_den)) : 0.0
    return s
end

"""
Full staged evaluation of one polish trial.

Returns `nothing` when the candidate can be rejected *before* the expensive
rungs: after the early links its failure count exceeds `k1_ref + 1`, or after
the optimized rungs it exceeds `k1_ref`. Returns the metrics, the failed-link
list and the polish key `(k1, n_violations, max_den, max_num)` otherwise.
"""
function _polish_eval(Ws, bs, input_set; tol, max_den, k1_ref=typemax(Int))
    m = evaluate_forward(Ws, bs, input_set)
    m = merge(m, evaluate_crown_init(Ws, bs, input_set))
    n_early = count(!ok for ok in values(early_links(m; tol=tol)))
    # (subtract rather than add to k1_ref: comparing n_early - 1 > k1_ref is
    #  overflow-safe when k1_ref == typemax(Int))
    n_early - 1 > k1_ref && return nothing

    m = merge(m, evaluate_crown_opt(Ws, bs, input_set))
    links = ladder_links(m; tol=tol)
    k1 = count(!ok for ok in values(links))
    k1 > k1_ref && return nothing

    nice = _polish_niceness(Ws, bs, input_set; max_den=max_den)
    # saturate both like the numerator: BigInt dens can exceed Int64
    key = (k1, nice.summary.n_violations,
           Int(min(nice.summary.max_den_seen, big(10)^9)), nice.max_num)
    failed = Symbol[name for (name, ok) in pairs(links) if !ok]
    return (m = m, nice = nice, key = key, failed = failed)
end

"""
Apply `n_moves` random polish moves (mutating `Ws`/`bs` in place).
"""
function _polish_apply_moves!(rng::AbstractRNG, Ws, bs; n_moves=1)
    L = length(Ws)
    for _ in 1:n_moves
        kind = rand(rng, 1:4)
        if kind == 1      # weight entry nudge
            i = rand(rng, 1:L)
            r, c = size(Ws[i])
            Ws[i][rand(rng, 1:r), rand(rng, 1:c)] +=
                rand(rng, _POLISH_NUDGES) * (rand(rng) < 0.5 ? -1 : 1)
        elseif kind == 2  # bias nudge
            i = rand(rng, 1:L)
            bs[i][rand(rng, 1:length(bs[i]))] +=
                rand(rng, _POLISH_NUDGES) * (rand(rng) < 0.5 ? -1 : 1)
        elseif kind == 3  # single hidden-neuron rescale (function-preserving)
            i = rand(rng, 1:L-1)
            j = rand(rng, 1:size(Ws[i], 1))
            f = rand(rng, _POLISH_RESCALES)
            Ws[i][j, :]   .= f .* Ws[i][j, :]
            bs[i][j]     *= f
            Ws[i+1][:, j] ./= f
        else              # whole hidden-layer rescale (function-preserving)
            i = rand(rng, 1:L-1)
            f = rand(rng, _POLISH_RESCALES)
            Ws[i] .*= f
            bs[i] .*= f
            Ws[i+1] ./= f
        end
    end
    return Ws, bs
end

"""
Save a polish result to the registry with the standard ladder tags.
"""
function _polish_save!(Ws, bs, input_c, ev; tol, seed, from, tag)
    ladder_tag, failed = classify_ladder(ev.m; tol=tol)
    ladder_tag = ladder_tag == :accepted && ev.nice.summary.n_violations > 0 ?
                 :nice_fail : ladder_tag
    tag_str = string(tag, " from ", from)
    return save_candidate!(merge((Ws=Ws, bs=bs, input_c=input_c), ev.m);
                           tag=ladder_tag, notes=tag_str,
                           failed_links=failed, niceness=ev.nice.summary,
                           trial=nothing, seed=seed)
end

"""
    polish(seed_idx; kwargs...)
    polish(Ws, bs, input_c; kwargs...)

Simulated-annealing polish of a registry seed (or of an explicit warm-start
network): search the local dyadic neighborhood for candidates that are
ladder-perfect AND nice (all tracked fractions have denominators ≤ `max_den`),
minimizing `polish_objective`.

Keyword arguments:
* `n_iter`   — number of polishing steps
* `seed`     — RNG seed for the moves (stored for reproducibility)
* `tol`      — minimum margin required by every ladder link
* `max_den`  — niceness bar (denominator)
* `T0`, `Tend` — annealing temperature range (geometric schedule)
* `kick_prob` — probability of applying 2–4 moves instead of a single one
* `patience` — restart from the best solution with `T = T0` after this many
                non-improving steps
* `save`     — save fully-accepted candidates to the registry as they appear,
                and the best solution at the end if it beats the seed
* `progress_cb(iter, n_iter, stats)` — called every `progress_every` steps
* `stop_cb()` — return true to stop cooperatively

Returns `(seed_idx, seed, best, best_key, seed_key, best_obj, seed_obj, stats)`.
"""
function polish(seed_idx::Int; kwargs...)
    c0 = get_saved_candidate(seed_idx)
    input_c = haskey(c0, :input_c) ? c0.input_c : 1.0
    return _polish_run(c0.Ws, c0.bs, input_c; seed_idx=seed_idx, kwargs...)
end

function polish(Ws0, bs0, input_c::Real; kwargs...)
    return _polish_run(Ws0, bs0, Float64(input_c); seed_idx=nothing, kwargs...)
end

function _polish_run(Ws0, bs0, input_c; n_iter=5000, seed=1, tol=1e-2, max_den=32,
                     T0=6.0, Tend=0.1, kick_prob=0.25, patience=1000,
                     save=true, progress_cb=nothing, stop_cb=nothing, progress_every=500,
                     seed_idx=nothing)
    rng = MersenneTwister(seed)
    from = isnothing(seed_idx) ? "warm-start" : "#$(seed_idx)"
    input_set = Hyperrectangle(low=[-input_c], high=[input_c])
    Ws0 = [copy(W) for W in Ws0]
    bs0 = [copy(b) for b in bs0]

    seed_ev = _polish_eval(Ws0, bs0, input_set; tol=tol, max_den=max_den)
    seed_obj = polish_objective(seed_ev.key; max_den=max_den)
    cur = (Ws=Ws0, bs=bs0, ev=seed_ev, obj=seed_obj)
    best = cur
    T, cool = T0, (Tend / T0)^(1 / max(n_iter, 1))
    since_improve = 0
    # save at most one accepted solution per run (unless a strictly better one
    # turns up): once the annealing wanders into the accepted region, saving
    # every accepted hit would flood the registry with near-duplicates
    saved_accepted_obj = nothing

    stats = Dict{Symbol, Int}(:iter => 0, :pruned => 0, :evals => 0,
                              :improved => 0, :accepted => 0)

    for it in 1:n_iter
        !isnothing(stop_cb) && stop_cb() && break

        # corridor: allow at most one extra link failure beyond the current
        # solution, so "temporarily break a link to clear a denominator" paths
        # stay reachable while the bulk of bad moves stays cheap to reject
        k1_ref = cur.ev.key[1]
        Ws = [copy(W) for W in cur.Ws]
        bs = [copy(b) for b in cur.bs]
        n_moves = rand(rng) < kick_prob ? rand(rng, 2:4) : 1
        _polish_apply_moves!(rng, Ws, bs; n_moves=n_moves)
        ev = _polish_eval(Ws, bs, input_set; tol=tol, max_den=max_den, k1_ref=k1_ref)
        stats[:iter] += 1

        if isnothing(ev)
            stats[:pruned] += 1
        else
            stats[:evals] += 1
            obj = polish_objective(ev.key; max_den=max_den)
            Δ = obj - cur.obj
            if Δ <= 0 || rand(rng) < exp(-Δ / max(T, eps()))
                cur = (Ws=Ws, bs=bs, ev=ev, obj=obj)
            end
            if obj < best.obj
                best = cur
                since_improve = 0
                stats[:improved] += 1
            else
                since_improve += 1
            end
            if ev.key[1] == 0 && ev.key[2] == 0
                stats[:accepted] += 1
                if save && (isnothing(saved_accepted_obj) || obj < saved_accepted_obj)
                    idx = _polish_save!(Ws, bs, input_c, ev; tol=tol, seed=seed,
                                        from=from, tag="polish-accepted")
                    saved_accepted_obj = obj
                    println("[polish ", from, "] ACCEPTED at iter ", it,
                            "  key = ", ev.key, "  -> registry #", idx)
                end
            end
        end

        since_improve >= patience && (cur = best; since_improve = 0; T = T0)
        T = max(T * cool, Tend)

        if !isnothing(progress_cb) && it % progress_every == 0
            progress_cb(it, n_iter, stats)
        end
    end

    # save the best solution if it beats the seed on the objective (and is not
    # an already-saved accepted solution)
    if save && best.obj < seed_obj && best.obj != saved_accepted_obj
        idx = _polish_save!(best.Ws, best.bs, input_c, best.ev; tol=tol, seed=seed,
                            from=from, tag="polish-best")
        println("[polish ", from, "] saved best -> registry #", idx,
                "   key = ", best.ev.key, "  obj = ", round(best.obj, digits=2),
                " (seed obj ", round(seed_obj, digits=2), ")")
    end

    return (seed_idx=seed_idx, seed=seed, best=best, best_key=best.ev.key,
            seed_key=seed_ev.key, best_obj=best.obj, seed_obj=seed_obj, stats=stats)
end
