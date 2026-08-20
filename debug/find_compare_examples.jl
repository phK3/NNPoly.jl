using Random
using NNPoly, NeuralVerification, Flux, LazySets
using Plots

const NP = NNPoly
const NV = NeuralVerification

#= 
    Search for tiny 1-input / 1-output networks with two ReLU layers such that

      * DPNFV(max_vars=0) is worse than DPNFV(max_vars > 0)
      * PolyCROWN is tighter than aCROWN
      * PolyCROWN with optimization steps is tighter than aCROWN with optimization steps

    The search is intentionally restricted to very small integer-like weights and
    biases so the resulting network is easy to hand-calculate and explain in a thesis.
=#

function make_network(Ws::Vector{Matrix{Float64}}, bs::Vector{Vector{Float64}})
    layers = Vector{NV.Layer{<:NV.ActivationFunction, Float64}}()
    for i in 1:length(Ws)-1
        push!(layers, NV.Layer(Ws[i], bs[i], NV.ReLU()))
    end
    push!(layers, NV.Layer(Ws[end], bs[end], NV.Id()))
    return NV.Network(layers)
end

function make_crown_chain(Ws::Vector{Matrix{Float64}}, bs::Vector{Vector{Float64}}; poly=false)
    layers = Any[]
    for i in 1:length(Ws)-1
        if i == 1 && poly
            α = similar(bs[i], length(bs[i]), 2, 2)
        else
            α = similar(bs[i], length(bs[i]), 1)
        end
        push!(layers, NP.CROWNLayer(Ws[i], bs[i], NV.ReLU(), α))
    end
    push!(layers, NP.CROWNLayer(Ws[end], bs[end], NV.Id(), similar(bs[end], 0)))
    return Chain(layers...)
end

function last_upper_bound(bounds_vec)
    return Float64(bounds_vec[end][1])
end

function dpnfv_upper_bound(net_npi, input_set; max_vars)
    s = NP.init_symbolic_interval_fvheur(net_npi, input_set, max_vars=max_vars)
    ŝ = NV.forward_network(NP.DPNFV(max_vars=max_vars), net_npi, s)
    lbs, ubs = NP.bounds(ŝ)
    return last_upper_bound(ubs)
end

function acrown_upper_bound(net_crown, input_set; n_steps)
    params = NP.OptimisationParams(
        n_steps=n_steps,
        timeout=30.0,
        patience=10,
        print_freq=1000,
        verbosity=0,
        start_lr=0.1,
        decay=0.98,
    )
    _, _, ubs = NP.optimise_bounds(NP.aCROWN(), net_crown, input_set, params=params, loss_fun=NP.upper_bound_loss)
    return last_upper_bound(ubs)
end

function pcrown_upper_bound(net_crown, input_set; n_steps)
    params = NP.OptimisationParams(
        n_steps=n_steps,
        timeout=30.0,
        patience=10,
        print_freq=1000,
        verbosity=0,
        start_lr=0.1,
        decay=0.98,
    )
    _, _, ubs = NP.optimise_bounds(
        NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true), poly_layers=1),
        net_crown,
        input_set,
        params=params,
        loss_fun=NP.upper_bound_loss
    )
    return last_upper_bound(ubs)
end

function sample_random_candidate(rng; max_width=2)
    # single input, two hidden ReLU layers, single output
    h1 = rand(rng, 2:max_width)
    h2 = rand(rng, 2:max_width)

    W1 = Float64.(rand(rng, [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], h1, 1))
    b1 = Float64.(rand(rng, [-1.0, -0.5, 0.0, 0.5, 1.0], h1))

    W2 = Float64.(rand(rng, [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], h2, h1))
    b2 = Float64.(rand(rng, [-1.0, -0.5, 0.0, 0.5, 1.0], h2))

    W3 = Float64.(rand(rng, [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], 1, h2))
    b3 = Float64.(rand(rng, [-0.5, 0.0, 0.5], 1))

    return [W1, W2, W3], [b1, b2, b3]
end

function evaluate_candidate(Ws, bs; n_steps=10)
    net = make_network(Ws, bs)
    net_npi = NV.NetworkNegPosIdx(net)
    input_set = Hyperrectangle(low=[-1.0], high=[1.0])

    ub_dpnfv_0 = dpnfv_upper_bound(net_npi, input_set; max_vars=0)
    ub_dpnfv_10 = dpnfv_upper_bound(net_npi, input_set; max_vars=10)

    net_crown = make_crown_chain(Ws, bs; poly=false)
    net_pcrown = make_crown_chain(Ws, bs; poly=true)
    ub_acrown_0 = acrown_upper_bound(net_crown, input_set; n_steps=0)
    ub_pcrown_0 = pcrown_upper_bound(net_pcrown, input_set; n_steps=0)

    ub_acrown_opt = acrown_upper_bound(net_crown, input_set; n_steps=n_steps)
    ub_pcrown_opt = pcrown_upper_bound(net_pcrown, input_set; n_steps=n_steps)

    return (
        Ws = Ws,
        bs = bs,
        ub_dpnfv_0 = ub_dpnfv_0,
        ub_dpnfv_10 = ub_dpnfv_10,
        ub_acrown_0 = ub_acrown_0,
        ub_pcrown_0 = ub_pcrown_0,
        ub_acrown_opt = ub_acrown_opt,
        ub_pcrown_opt = ub_pcrown_opt,
    )
end

function candidate_matches(metrics; tol=1e-2)
    # Require a clear, non-negligible gain to avoid accepting cases that only
    # differ by floating-point noise. The comparisons are all on upper bounds,
    # so we want the "better" method to be visibly tighter than the baseline.
    dpn = metrics.ub_dpnfv_0 > metrics.ub_dpnfv_10 + tol
    poly_gt_acrown = metrics.ub_pcrown_0 < metrics.ub_acrown_0 - tol
    poly_gt_acrown_opt = metrics.ub_pcrown_opt < metrics.ub_acrown_opt - tol
    poly_improved_by_opt = metrics.ub_pcrown_opt < metrics.ub_pcrown_0 - tol
    return dpn && poly_gt_acrown && poly_gt_acrown_opt && poly_improved_by_opt
end

# Lower score is nicer: small integer values and small-denominator rationals score best.
# This is only a human-readability heuristic, not a mathematical guarantee.
function nice_bound_score(x::N; max_den=8, tol=1e-3) where N<:Number
    # Small integers are preferred.
    r = round(x)
    if abs(x - r) <= tol
        return (1, 0.0, 0)
    end

    # Rational approximation with small denominators is preferred.
    q = rationalize(Float64(x), tol=tol)
    den = denominator(q)
    if den <= max_den
        return (2, abs(float(q) - x), den)
    end

    return (3, abs(x), 10_000)
end

function nice_bound_score(c::NamedTuple; max_den=8, tol=1e-3)
    bounds = [
        c.ub_dpnfv_0,
        c.ub_dpnfv_10,
        c.ub_acrown_0,
        c.ub_pcrown_0,
    ]
    scores = [nice_bound_score(bound; max_den=max_den, tol=tol) for bound in bounds]

    return (
        sum(score[1] for score in scores),
        sum(score[2] for score in scores),
        sum(score[3] for score in scores),
    )
end

function sort_by_nice_bounds(candidates; max_den=8, tol=1e-3)
    sort!(candidates, by=c -> nice_bound_score(c; max_den=max_den, tol=tol))
    return candidates
end

function find_candidates(; n_trials=5000, n_steps=25, rng_seed=1, tol=1e-2, max_width=2, max_den=8)
    rng = MersenneTwister(rng_seed)
    candidates = []

    for trial in 1:n_trials
        Ws, bs = sample_random_candidate(rng, max_width=max_width)
        metrics = evaluate_candidate(Ws, bs; n_steps=n_steps)
        if candidate_matches(metrics; tol=tol)
            push!(candidates, metrics)
        end
    end

    return sort_by_nice_bounds(candidates; max_den=max_den)
end

function print_candidate_summary(c; idx=nothing, n_steps=25)
    if !isnothing(idx)
        println("Candidate #", idx)
    end 

    println("W1 = ", c.Ws[1])
    println("b1 = ", c.bs[1])
    println("W2 = ", c.Ws[2])
    println("b2 = ", c.bs[2])
    println("W3 = ", c.Ws[3])
    println("b3 = ", c.bs[3])
    println("DPNFV(max_vars=0) upper = ", c.ub_dpnfv_0)
    println("DPNFV(max_vars=10) upper = ", c.ub_dpnfv_10)
    println("aCROWN(n_steps=0) upper = ", c.ub_acrown_0)
    println("PolyCROWN(n_steps=0) upper = ", c.ub_pcrown_0)
    println("aCROWN(n_steps=$(n_steps)) upper = ", c.ub_acrown_opt)
    println("PolyCROWN(n_steps=$(n_steps)) upper = ", c.ub_pcrown_opt)
    println("PolyCROWN improvement from n_steps=0 to n_steps=$(n_steps) = ", c.ub_pcrown_0 - c.ub_pcrown_opt)
    println("nice output score = ", nice_bound_score(c))
    println("-" ^ 72)
end

function get_output_polys(net, α1, α2, lbs, ubs, solver, input_set)
    α1 = copy(α1)
    α2 = copy(α2)
    # setup precomputed values
    s = NP.initialize_symbolic_domain(solver, net[1:1], input_set)
    ŝ, _, _, rs, cs, symmetric_factor, unique_idxs, duplicate_idxs = NP.initialize_params_bounds(solver, net, 2, s)

    #println(lbs)
    #println(ubs)
    
    # change params to the optimized monomial coefficients
    net[1].α .= α1
    net[2].α .= α2
    
    s_poly = NP.forward_act_stub(solver.poly_solver, net[1], ŝ, lbs[1], ubs[1], rs, cs, symmetric_factor, unique_idxs, duplicate_idxs)
    s_crown = NV.forward_network(solver.lin_solver, net[2:end], s_poly, lbs[2:end], ubs[2:end])
    
    L, _ = NP.interval_map_common(min.(0, s_crown.Λ), max.(0, s_crown.Λ), s_poly.poly_interval.Low, s_poly.poly_interval.Up, s_crown.λ)
    _, U = NP.interval_map_common(min.(0, s_crown.Γ), max.(0, s_crown.Γ), s_poly.poly_interval.Low, s_poly.poly_interval.Up, s_crown.γ)

    return s_poly, L, U
end

function get_bounding_functions(solver::NP.DPNFV, net, lbs, ubs, input_set)
    s = NP.init_symbolic_interval_fvheur(net, input_set, max_vars=solver.max_vars)
    ŝ = NV.forward_network(solver, net, s)
    L, U = NP.substitute_variables(ŝ)

    f_lower = x -> (L[1,1] * x + L[1,2])
    f_upper = x -> (U[1,1] * x + U[1,2])

    return f_lower, f_upper
end

function get_bounding_functions(solver::NP.aCROWN, net, lbs, ubs, input_set)
    ŝ = NV.forward_network(solver, net, input_set, lbs, ubs)

    f_lower = x -> (ŝ.Λ[1,1] * x + ŝ.λ[1])
    f_upper = x -> (ŝ.Γ[1,1] * x + ŝ.γ[1])

    return f_lower, f_upper
end

function get_bounding_functions(solver::NP.PolyCROWN, net, lbs, ubs, input_set)
    s_poly, L, U = get_output_polys(net, net[1].α, net[2].α, lbs, ubs, solver, input_set)

    f_lower = x -> NP.evaluate(L, x)[1]
    f_upper = x -> NP.evaluate(U, x)[1]

    return f_lower, f_upper
end

function plot_candidate(c; plot_lower=false, loss_fun=NP.upper_bound_loss)
    l, u = -1., 1.
    max_vars = 10

    net_npi = make_network(c.Ws, c.bs) |> NV.NetworkNegPosIdx
    net_acrown = make_crown_chain(c.Ws, c.bs; poly=false)
    net_pcrown = make_crown_chain(c.Ws, c.bs; poly=true)

    input_set = Hyperrectangle(low=[l], high=[u])

    f_dpnfv_lower, f_dpnfv_upper = get_bounding_functions(NP.DPNFV(max_vars=max_vars), net_npi, nothing, nothing, input_set)

    pCROWN = NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true), poly_layers=1, prune_neurons=false)
    params_poly = NP.OptimisationParams(n_steps=1000, print_freq=100, start_lr=0.1, decay=0.98, patience=1000)
    t = @elapsed res, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(pCROWN, net_pcrown, input_set, params=params_poly, loss_fun=loss_fun, print_results=true)
    f_pcrown_lower, f_pcrown_upper = get_bounding_functions(pCROWN, net_pcrown, lbs_pcrown, ubs_pcrown, input_set)

    aCROWN = NP.aCROWN()
    params = NP.OptimisationParams(n_steps=1000, print_freq=100, start_lr=0.1, decay=0.98, patience=1000)
    t = @elapsed res, lbs_acrown, ubs_acrown = NP.optimise_bounds(aCROWN, net_acrown, input_set, params=params, loss_fun=loss_fun)
    f_acrown_lower, f_acrown_upper = get_bounding_functions(aCROWN, net_acrown, lbs_acrown, ubs_acrown, input_set)

    xs = range(l, u; length=200)
    y_nn = net_acrown(reshape(xs, 1, :)) |> vec

    pl = plot(xs, y_nn, label="NN output")
    plot!(xs, f_dpnfv_upper.(xs), label="DPNFV upper bound")
    plot!(xs, f_acrown_upper.(xs), label="aCROWN upper bound")
    plot!(xs, f_pcrown_upper.(xs), label="pCROWN upper bound")

    if plot_lower
        plot!(xs, f_dpnfv_lower.(xs), label="DPNFV lower bound")
        plot!(xs, f_acrown_lower.(xs), label="aCROWN lower bound")
        plot!(xs, f_pcrown_lower.(xs), label="pCROWN lower bound")
    end

    pl 
end

"""
Plot relaxations for a candidate network and a set of verification algorithms.

args:
    c - candidate network
    options - dict of algorithm_name => [param range], where the parameters are max_vars vor DPNFV and the iterations after which the result should be plotted for aCROWN and pCROWN.
"""
function plot_candidate(c, options::Dict; plot_lower=false, loss_fun=NP.upper_bound_loss, linewidth=2, no_legend=false, ylims=nothing, framestyle=:axes)
    l, u = -1., 1. 
    max_vars = 10 

    net_npi = make_network(c.Ws, c.bs) |> NV.NetworkNegPosIdx
    net_acrown = make_crown_chain(c.Ws, c.bs; poly=false)
    net_pcrown = make_crown_chain(c.Ws, c.bs; poly=true)

    input_set = Hyperrectangle(low=[l], high=[u])

    dpnfv_functions = []
    if "DPNFV" in keys(options)
        println("=== DPNFV ===")
        for max_vars in options["DPNFV"]
            f_dpnfv_lower, f_dpnfv_upper = get_bounding_functions(NP.DPNFV(max_vars=max_vars), net_npi, nothing, nothing, input_set)
            push!(dpnfv_functions, (max_vars, f_dpnfv_lower, f_dpnfv_upper))
        end
    end

    acrown_functions = []
    if "aCROWN" in keys(options)
        println("=== aCROWN ===")
        for n_steps in options["aCROWN"]
            aCROWN = NP.aCROWN()
            params = NP.OptimisationParams(n_steps=n_steps, print_freq=100, start_lr=0.1, decay=0.98, patience=1000)
            t = @elapsed res, lbs_acrown, ubs_acrown = NP.optimise_bounds(aCROWN, net_acrown, input_set, params=params, loss_fun=loss_fun)
            f_acrown_lower, f_acrown_upper = get_bounding_functions(aCROWN, net_acrown, lbs_acrown, ubs_acrown, input_set)
            push!(acrown_functions, (n_steps, f_acrown_lower, f_acrown_upper))

            println("lbs: ", lbs_acrown[end])
            println("ubs: ", ubs_acrown[end])
        end
    end

    pcrown_functions = []
    if "pCROWN" in keys(options)
        println("=== pCROWN ===")
        for n_steps in options["pCROWN"]
            pCROWN = NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true), poly_layers=1, prune_neurons=false)
            params_poly = NP.OptimisationParams(n_steps=n_steps, print_freq=100, start_lr=0.1, decay=0.98, patience=1000)
            t = @elapsed res, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(pCROWN, net_pcrown, input_set, params=params_poly, loss_fun=loss_fun)
            f_pcrown_lower, f_pcrown_upper = get_bounding_functions(pCROWN, net_pcrown, lbs_pcrown, ubs_pcrown, input_set)
            push!(pcrown_functions, (n_steps, f_pcrown_lower, f_pcrown_upper))

            println("lbs: ", lbs_pcrown[end])
            println("ubs: ", ubs_pcrown[end])
        end
    end


    xs = range(l, u; length=200)
    y_nn = net_acrown(reshape(xs, 1, :)) |> vec
    if no_legend
        pl = plot(xs, y_nn, label="NN output", color=1, linewidth=linewidth, legend=false, framestyle=framestyle, xlabel="x", ylabel="y")
    else 
        pl = plot(xs, y_nn, label="NN output", color=1, linewidth=linewidth, framestyle=framestyle, xlabel="x", ylabel="y")
    end

    if !isnothing(ylims)
        ylims!(ylims)
    end

    alg_dict = Dict("DPNFV" => dpnfv_functions, "aCROWN" => acrown_functions, "pCROWN" => pcrown_functions)

    for (i, algo) in enumerate(keys(alg_dict))
        len_params = length(alg_dict[algo])
        for (j, (n, fl, fu)) in enumerate(alg_dict[algo])
            param_str = algo == "DPNFV" ? "$n vars" : "$n steps"

            # alpha = i / len_params
            # s.t. last setting is solid, earlier settings are more transparent
            if plot_lower
                plot!(xs, fl.(xs), label=nothing, color=i+1, alpha=j / len_params, linewidth=linewidth)
            end

            plot!(xs, fu.(xs), label="$(algo) bound ($(param_str))", color=i+1, alpha=j / len_params, linewidth=linewidth)
        end
    end

    pl    
end


function get_saved_candidate(idx)
    if idx == 1
        # might be ok
        c = (Ws = [[-2.0; -1.0;;], [0.5 0.5; 1.0 -2.0], [-1.0 1.0]], 
            bs = [[0.5, 0.5], [1.0, 0.0], [1.0]], 
            ub_dpnfv_0 = 0.40625, ub_dpnfv_10 = -0.0625, 
            ub_acrown_0 = 0.40000000000000013, ub_pcrown_0 = 0.3055555555555556,
            ub_acrown_opt = 0.40000000000000013, ub_pcrown_opt = 0.2544843302973525)
    elseif idx == 2
        # don't use that
        c = (Ws = [[-2.0; -0.5;;], [2.0 2.0; 1.0 2.0], [-1.0 2.0]], 
            bs = [[0.5, -0.5], [-0.5, 0.5], [-0.5]], 
            ub_dpnfv_0 = 4.121875, ub_dpnfv_10 = 2.5545454545454542, 
            ub_acrown_0 = 3.0, ub_pcrown_0 = 2.0000000000000004, 
            ub_acrown_opt = 3.0, ub_pcrown_opt = 1.0)
    elseif idx == 3
        # don't use that
        c = (Ws = [[2.0; 0.5;;], [-2.0 -0.5; -1.0 -2.0], [0.5 -0.5]], 
            bs = [[0.5, -0.5], [-0.5, 0.5], [-0.5]], 
            ub_dpnfv_0 = 0.1375, ub_dpnfv_10 = -0.0818181818181819, 
            ub_acrown_0 = 0.75, ub_pcrown_0 = 0.25, 
            ub_acrown_opt = 0.1262191868126451, ub_pcrown_opt = -0.5)
    elseif idx == 4
        c = (Ws = [[-2.0; -2.0;;], [0.5 -1.0; 2.0 0.5], [-2.0 -2.0]], 
            bs = [[0.5, 0.5], [1.0, 1.0], [-0.5]], 
            ub_dpnfv_0 = 1.1581249999999996, ub_dpnfv_10 = 1.1248470279720282, 
            ub_acrown_0 = 3.0, ub_pcrown_0 = 0.125, 
            ub_acrown_opt = -3.561336447698669, ub_pcrown_opt = -4.192802795967198)
    elseif idx == 5
        c = (Ws = [[-1.0; -2.0;;], [0.5 -0.5; -1.0 -2.0], [-2.0 2.0]], 
            bs = [[0.5, 0.5], [1.0, 0.5], [0.5]], 
            ub_dpnfv_0 = 4.375, ub_dpnfv_10 = 3.4375,
            ub_acrown_0 = 5.5, ub_pcrown_0 = 3.9999999999999996, 
            ub_acrown_opt = -0.0028714336221752623, ub_pcrown_opt = -0.2598704991645755)
    elseif idx == 6
        # why is aCROWN better than pCROWN here?
        c = (Ws = [[1.0; 2.0;;], [1.0 -2.0; -1.0 1.0], [-1.0 -2.0]], 
            bs = [[1.0, 0.5], [0.5, 0.5], [-0.5]], 
            ub_dpnfv_0 = -0.09895833333333337, ub_dpnfv_10 = -0.28246934225195097, 
            ub_acrown_0 = 2.0, ub_pcrown_0 = 0.5, 
            ub_acrown_opt = -0.5216754907413461, ub_pcrown_opt = -0.6583304411855313)
    elseif idx == 7
        # good
        c = (Ws = [[-2.0; -2.0;;], [0.5 -1.0; 2.0 0.5], [-2.0 -2.0]], 
            bs = [[0.5, 0.5], [1.0, 1.0], [-0.5]], 
            ub_dpnfv_0 = 1.1581249999999996, ub_dpnfv_10 = 1.1248470279720282, 
            ub_acrown_0 = 3.0, ub_pcrown_0 = 0.125, 
            ub_acrown_opt = -3.561336447698669, ub_pcrown_opt = -4.192802795967198)
    elseif idx == 8
        # good
        c = (Ws = [[-1.0; -2.0;;], [0.5 -0.5; -1.0 -2.0], [-2.0 2.0]], 
            bs = [[0.5, 0.5], [1.0, 0.5], [0.5]], 
            ub_dpnfv_0 = 4.375, ub_dpnfv_10 = 3.4375,
            ub_acrown_0 = 5.5, ub_pcrown_0 = 3.9999999999999996, 
            ub_acrown_opt = -0.0028714336221752623, ub_pcrown_opt = -0.2598704991645755)
    elseif idx == 9
        # good
        c = (Ws = [[-2.0; -2.0;;], [1.0 -2.0; 2.0 -1.0], [1.0 -2.0]], 
            bs = [[1.0, 0.5], [-0.5, 0.5], [0.0]], 
            ub_dpnfv_0 = 2.803571428571429, ub_dpnfv_10 = 2.6000000000000005, 
            ub_acrown_0 = 3.5, ub_pcrown_0 = 3.3333333333333326, 
            ub_acrown_opt = 0.22183746387244208, ub_pcrown_opt = -0.761333617902771)
    elseif idx == 10
        # good
        c = (Ws = [[2.0; 1.0;;], [0.5 -1.0; -1.0 -1.0], [0.5 2.0]], 
            bs = [[0.5, 0.0], [0.5, -1.0], [0.5]], 
            ub_dpnfv_0 = 1.875, ub_dpnfv_10 = 1.6901408450704227, 
            ub_acrown_0 = 1.575, ub_pcrown_0 = 1.375, 
            ub_acrown_opt = 1.0739141365660774, ub_pcrown_opt = 0.9176197226814151)
    elseif idx == 11
        # VERY GOOD CANDIDATE FOR DISSERTATION EXAMPLE
        c = (Ws = [[-1.0; -2.0;;], [0.5 -0.5; -1.0 -2.0], [-2.0 2.0]], 
            bs = [[0.5, 0.5], [1.0, 0.5], [1.5]], 
            ub_dpnfv_0 = 4.375, ub_dpnfv_10 = 3.4375,
            ub_acrown_0 = 5.5, ub_pcrown_0 = 3.9999999999999996, 
            ub_acrown_opt = -0.0028714336221752623, ub_pcrown_opt = -0.2598704991645755)
    elseif idx == 12
        # CANDIDATE WITH SLIGHTLY EASIER NUMBERS THAN 11
        c = (Ws = [[-1.0; -2.0;;], [1. -1; -2.0 -4.0], [-1.0 1.0]], 
            bs = [[0.5, 0.5], [2.0, 1.], [1.5]], 
            ub_dpnfv_0 = 4.375, ub_dpnfv_10 = 3.4375,
            ub_acrown_0 = 5.5, ub_pcrown_0 = 3.9999999999999996, 
            ub_acrown_opt = -0.0028714336221752623, ub_pcrown_opt = -0.2598704991645755)
    elseif idx == 13
        c = (Ws = [[1.0; 2.0;;], [-1. 1; 2.0 4.0], [1.0 -1.0]], 
            bs = [[-0.5, -0.5], [2.0, 1.], [1.5]], 
            ub_dpnfv_0 = 4.375, ub_dpnfv_10 = 3.4375,
            ub_acrown_0 = 5.5, ub_pcrown_0 = 3.9999999999999996, 
            ub_acrown_opt = -0.0028714336221752623, ub_pcrown_opt = -0.2598704991645755)
    else 
        error("No saved candidate for index $idx")
    end

    return c
end

function plot_saved_candidate(idx, options; plot_lower=false, no_legend=false, ylims=nothing)
    c = get_saved_candidate(idx)
    plot_candidate(c, options, plot_lower=plot_lower, no_legend=no_legend, ylims=ylims)  
end

function plot_saved_candidate(idx; plot_lower=false, no_legend=false)
    c = get_saved_candidate(idx)
    plot_candidate(c, plot_lower=plot_lower, no_legend=no_legend)
end

function gif_saved_candidate(idx, solver_name, param_range, gif_filename; plot_lower=false, no_legend=false, fps=15, ylims=nothing)
    anim = @animate for p in param_range
        options = Dict(solver_name => [p])
        plot_saved_candidate(idx, options, plot_lower=plot_lower, no_legend=no_legend, ylims=ylims)
    end

    gif(anim, gif_filename, fps=fps)
end

function main(; n_trials=2000, n_steps=25, rng_seed=1, tol=1e-2, max_width=2, max_den=8)
    matches = find_candidates(n_trials=n_trials, n_steps=n_steps, rng_seed=rng_seed, tol=tol, max_width=max_width, max_den=max_den)
    println("Found ", length(matches), " candidate networks with tolerance ", tol, ".")
    if isempty(matches)
        println("No matches found. Increase n_trials or adjust the candidate pool.")
        return matches
    end

    for (idx, c) in enumerate(matches[1:min(10, length(matches))])
        print_candidate_summary(c; idx=idx)
    end

    return matches
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
