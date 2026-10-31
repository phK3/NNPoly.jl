
# ==========================================================================================
# Niceness analysis: are all (hand-checkable) quantities fractions with small
# denominators?
#
# Tracked quantities:
#   * final concrete bounds of rungs (i)–(v)  [rung (vi) NNPoly and the optimized
#     rungs (vii)/(viii) are exempt — NNPoly is not expected to produce nice bounds]
#   * DPNFV: per-layer concrete pre-activation bounds + final symbolic bound
#     matrices L, U (per method used in the ladder)
#   * zonotope: per-layer centers and generators (computed in exact Rational{BigInt})
#   * aCROWN(0): coefficients Λ, λ, Γ, γ of the linear bounding functions
#
# Scoring: every tracked value is recovered as an exact fraction via
# `rationalize(·, tol)`; a value is "nice" if its denominator ≤ `max_den`.
# ==========================================================================================

"""
Recover the exact fraction of a (floating point) value.
"""
frac_of(x::Rational; tol=1e-9) = x
frac_of(x::Real; tol=1e-9) = rationalize(Float64(x), tol=tol)

denominator_of(x; tol=1e-9) = denominator(frac_of(x; tol=tol))

const _NICE_ITEM_TAG = Symbol[]

"""
Add `(name, value)` (or a vector/matrix of values) to the items list.
"""
function add_nice_items!(items, name, values; exempt=false)
    vals = values isa AbstractArray ? vec(values) : [values]
    for (i, v) in enumerate(vals)
        v === nothing && continue
        (v isa Real && isnan(v)) && continue
        (v isa Real && isinf(v)) && continue
        den = denominator_of(v)
        push!(items, (name = length(vals) == 1 ? String(name) : "$(name)[$i]",
                      value = v, den = den, exempt = exempt))
    end
    return items
end

"""
Collect all tracked intermediate quantities for a candidate network.

Returns a vector of `(name, value, den, exempt)` tuples.
"""
function collect_niceness(Ws, bs, input_set=default_input_set(); tol=1e-9)
    items = NamedTuple[]

    net = make_network(Ws, bs)
    net_npi = NV.NetworkNegPosIdx(net)

    ### rung (i): box — final bound (intermediates are trivially nice for the alphabet)
    add_nice_items!(items, :ub_box, box_upper_bound(net, input_set))

    ### rungs (ii), (iii), (iii'): DPNFV per-layer bounds + final symbolic matrices
    for (tag, mv, method) in ((:outer0, 0, :OuterBoundsNeurify),
                              (:sep0, 0, :NeurifyRelax),
                              (:fv, 10, :NeurifyRelax),
                              (:fv_dp, 10, :DeepPolyRelax))
        d = dpnfv_details(net_npi, input_set; max_vars=mv, method=method)
        add_nice_items!(items, Symbol(:ub_, tag), d.ub)
        for (k, (lb, ub)) in enumerate(zip(d.lbs_per_layer, d.ubs_per_layer))
            add_nice_items!(items, Symbol(tag, :_lbs_layer, k), lb)
            add_nice_items!(items, Symbol(tag, :_ubs_layer, k), ub)
        end
        add_nice_items!(items, Symbol(tag, :_L), d.L)
        add_nice_items!(items, Symbol(tag, :_U), d.U)
    end

    ### rung (iv): zonotope centers/generators, exact rational arithmetic
    Ws_r, bs_r = rationalize_network(Ws, bs)
    net_r = make_network(Ws_r, bs_r)
    input_r = Hyperrectangle(low=[Rational{BigInt}(low(input_set)[1])],
                             high=[Rational{BigInt}(high(input_set)[1])])
    _, trace = zono_propagate(net_r, input_r, Rational{BigInt})
    for t in trace
        add_nice_items!(items, Symbol(:zono_layer, t.layer, :_pre_c), center(t.pre))
        add_nice_items!(items, Symbol(:zono_layer, t.layer, :_pre_G), genmat(t.pre))
        add_nice_items!(items, Symbol(:zono_layer, t.layer, :_lo), t.lo)
        add_nice_items!(items, Symbol(:zono_layer, t.layer, :_hi), t.hi)
        add_nice_items!(items, Symbol(:zono_layer, t.layer, :_λ), t.λ)
        add_nice_items!(items, Symbol(:zono_layer, t.layer, :_μ), t.μ)
        add_nice_items!(items, Symbol(:zono_layer, t.layer, :_post_c), center(t.post))
        add_nice_items!(items, Symbol(:zono_layer, t.layer, :_post_G), genmat(t.post))
    end
    Zout = trace[end].post  # final layer has Id activation: post-activation == output set
    # note: genmat may have 0 columns when all error generators cancel exactly in the output
    add_nice_items!(items, :ub_zono, (center(Zout) + vec(sum(abs.(genmat(Zout)), dims=2)))[1])

    ### rung (v): aCROWN(0) coefficients
    d = acrown0_details(Ws, bs, input_set)
    add_nice_items!(items, :acrown0_Λ, d.Λ)
    add_nice_items!(items, :acrown0_λ, d.λ)
    add_nice_items!(items, :acrown0_Γ, d.Γ)
    add_nice_items!(items, :acrown0_γ, d.γ)
    add_nice_items!(items, :ub_acrown0, d.ubs[end][1])

    return items
end

"""
Summarize a niceness item list: violations, worst denominators, aggregate score.

A candidate is "nice" if no non-exempt item has a denominator above `max_den`.
Exempt items (currently none — the exempted rungs are simply not collected)
are reported but do not cause violations.
"""
function niceness_summary(items; max_den=32)
    violations = filter(i -> !i.exempt && i.den > max_den, items)
    max_den_seen = isempty(items) ? 0 : maximum(i.den for i in items)
    worst = sort(violations, by=i -> -i.den)[1:min(10, length(violations))]
    # aggregate score: 0 per integer, log2(den) for nice fractions, 20 + log2(den) beyond max_den
    score = 0.0
    for i in items
        penalty = i.den == 1 ? 0.0 :
                  (i.den <= max_den ? log2(i.den) : 20.0 + log2(i.den))
        score += penalty
    end
    # note: the full `violations` list is intentionally not returned (it is large);
    # `worst` holds the 10 biggest offenders.
    return (n_items=length(items), n_violations=length(violations), max_den_seen=max_den_seen,
            worst=worst, score=score)
end

"""
    niceness_report(Ws, bs; max_den=32, input_set)

Full niceness report for a candidate network.
"""
function niceness_report(Ws, bs, input_set=default_input_set(); max_den=32)
    items = collect_niceness(Ws, bs, input_set)
    return niceness_summary(items; max_den=max_den)
end

"""
Print a niceness report grouped by category, marking violations.
"""
function print_niceness(Ws, bs, input_set=default_input_set(); max_den=32, io=stdout)
    items = collect_niceness(Ws, bs, input_set)
    s = niceness_summary(items; max_den=max_den)
    println(io, "Niceness report (max_den = ", max_den, "): ",
            s.n_violations, " violations out of ", s.n_items, " items, ",
            "score = ", round(s.score, digits=1))
    for i in items
        mark = i.den > max_den ? " ✗ (den = $(i.den))" : ""
        println(io, "  ", rpad(i.name, 32), frac_str(i.value), mark)
    end
    return s
end
