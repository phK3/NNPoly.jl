
# ==========================================================================================
# Presentation & plotting: human-readable bound representations and relaxation plots.
#
# Ported (and adapted) from debug/find_compare_examples.jl. Plotting requires Plots.jl;
# ExampleSearch.jl includes this file in a try/catch, so the core module also loads
# without Plots.
# ==========================================================================================

import Plots

subscript(i) = join(Char(0x2080 + d) for d in reverse!(digits(i)))

"""
Prints a linear function of the form y = A*x + b in human-readable form,
with coefficients as (rationalized) fractions.
"""
function print_linear_function(A, b; indentation=0, io=stdout)
    m, n = size(A)
    for i in 1:m
        line = "\t"^indentation * "y$(subscript(i)) = "
        for j in 1:n
            coeff = A[i, j]
            if coeff != 0
                line *= frac_str(coeff) * "·x$(subscript(j)) + "
            end
        end
        line *= frac_str(b[i])
        println(io, line)
    end
end

"""
Symbolic bound representation via DPNFV restricted to the first `layer` layers.

`preact=true` uses `Id` as the last activation (pre-activation bounds of `layer`),
`preact=false` applies the ReLU. Returns (and prints) the substituted linear
lower/upper bound representations `L, U` in terms of the input.
"""
function get_bound_representation(solver::NP.DPNFV, Ws, bs, lbs, ubs, input_set;
                                  layer=3, preact=true, printing=true, indentation=0, io=stdout)
    last_act = preact ? NV.Id() : NV.ReLU()
    net = make_network(Ws[1:layer], bs[1:layer], last_act=last_act) |> NV.NetworkNegPosIdx

    s = NP.init_symbolic_interval_fvheur(net, input_set, max_vars=solver.max_vars)
    ŝ = NV.forward_network(solver, net, s)
    L, U = NP.substitute_variables(ŝ)

    if printing
        println(io, "Lower bound representation:")
        print_linear_function(L[:, 1:1], L[:, end], indentation=indentation+1, io=io)
        println(io, "Upper bound representation:")
        print_linear_function(U[:, 1:1], U[:, end], indentation=indentation+1, io=io)
    end

    return L, U
end

"""
Per-layer bound representation of the whole network (pre- and post-activation).
"""
function get_bound_representation_net(solver, Ws, bs, lbs, ubs, input_set; io=stdout)
    n_layers = length(Ws)
    for i in 1:n_layers
        println(io, "=== Layer $i ===")
        println(io, "Pre-activation bounds:")
        get_bound_representation(solver, Ws, bs, lbs, ubs, input_set; layer=i, preact=true, io=io)

        if i != n_layers
            # last layer is just Id activation
            println(io, "Post-activation bounds:")
            get_bound_representation(solver, Ws, bs, lbs, ubs, input_set; layer=i, preact=false, io=io)
            println(io)
        end
    end
end

"""
Bound representation for a candidate from the registry.
"""
function print_bound_representation(idx, solver; input_set=nothing, io=stdout)
    c = get_saved_candidate(idx)
    input_set = isnothing(input_set) ? input_set_of(c) : input_set
    get_bound_representation_net(solver, c.Ws, c.bs, nothing, nothing, input_set; io=io)
end

"""
Interval (Box) bound representation restricted to the first `layer` layers.
"""
function get_bound_representation(solver::NV.Box, Ws, bs, lbs, ubs, input_set;
                                  layer=3, preact=true, printing=true, indentation=0, io=stdout)
    last_act = preact ? NV.Id() : NV.ReLU()
    net = make_network(Ws[1:layer], bs[1:layer], last_act=last_act)

    ŝ = NV.forward_network(NV.Box(), net, input_set)

    if printing
        n = length(high(ŝ))
        println(io, "Lower bound representation:")
        print_linear_function(zeros(n, 1), low(ŝ), indentation=indentation+1, io=io)
        println(io, "Upper bound representation:")
        print_linear_function(zeros(n, 1), high(ŝ), indentation=indentation+1, io=io)
    end

    return ŝ
end

"""
Linear bounding functions Λx + λ / Γx + γ of aCROWN (after optimisation), restricted
to the first `layer` layers. Uses the optimised intermediate bounds, so the full
network must be optimised first.
"""
function get_bound_representation(solver::NP.aCROWN, Ws, bs, lbs, ubs, input_set;
                                  layer=3, preact=true, printing=true, io=stdout,
                                  loss_fun=NP.upper_bound_loss, n_steps=1000, print_freq=100)
    # need the full network with the right optimization objective to get the right intermediate bounds
    net_full = make_crown_chain(Ws, bs; poly=false)
    params = NP.OptimisationParams(n_steps=n_steps, print_freq=print_freq, start_lr=0.1,
                                   decay=0.98, patience=1000, verbosity=0)
    _, lbs_acrown, ubs_acrown = NP.optimise_bounds(solver, net_full, input_set, params=params, loss_fun=loss_fun)

    last_act = preact ? NV.Id() : NV.ReLU()
    net = make_crown_chain(Ws[1:layer], bs[1:layer]; poly=false, last_act=last_act)

    ŝ = NV.forward_network(solver, net, input_set, lbs_acrown, ubs_acrown)

    if printing
        println(io, "Lower bound representation:")
        print_linear_function(ŝ.Λ, ŝ.λ, io=io)
        println(io, "Upper bound representation:")
        print_linear_function(ŝ.Γ, ŝ.γ, io=io)
    end

    return ŝ
end

function get_bounding_functions(solver::NP.DPNFV, net, lbs, ubs, input_set)
    s = NP.init_symbolic_interval_fvheur(net, input_set, max_vars=solver.max_vars)
    ŝ = NV.forward_network(solver, net, s)
    L, U = NP.substitute_variables(ŝ)

    f_lower = x -> (L[1, 1] * x + L[1, 2])
    f_upper = x -> (U[1, 1] * x + U[1, 2])

    return f_lower, f_upper
end

function get_bounding_functions(solver::NP.aCROWN, net, lbs, ubs, input_set)
    ŝ = NV.forward_network(solver, net, input_set, lbs, ubs)

    f_lower = x -> (ŝ.Λ[1, 1] * x + ŝ.λ[1])
    f_upper = x -> (ŝ.Γ[1, 1] * x + ŝ.γ[1])

    return f_lower, f_upper
end

"""
Evaluate the optimized PolyCROWN relaxations and return the output polynomials
`L, U` (as NNPoly polynomial objects).
"""
function get_output_polys(net, α1, α2, lbs, ubs, solver, input_set)
    α1 = copy(α1)
    α2 = copy(α2)
    # setup precomputed values
    s = NP.initialize_symbolic_domain(solver, net[1:1], input_set)
    ŝ, _, _, rs, cs, symmetric_factor, unique_idxs, duplicate_idxs =
        NP.initialize_params_bounds(solver, net, 2, s)

    # change params to the optimized monomial coefficients
    net[1].α .= α1
    net[2].α .= α2

    s_poly = NP.forward_act_stub(solver.poly_solver, net[1], ŝ, lbs[1], ubs[1],
                                 rs, cs, symmetric_factor, unique_idxs, duplicate_idxs)
    s_crown = NV.forward_network(solver.lin_solver, net[2:end], s_poly, lbs[2:end], ubs[2:end])

    L, _ = NP.interval_map_common(min.(0, s_crown.Λ), max.(0, s_crown.Λ),
                                  s_poly.poly_interval.Low, s_poly.poly_interval.Up, s_crown.λ)
    _, U = NP.interval_map_common(min.(0, s_crown.Γ), max.(0, s_crown.Γ),
                                  s_poly.poly_interval.Low, s_poly.poly_interval.Up, s_crown.γ)

    return s_poly, L, U
end

function get_bounding_functions(solver::NP.PolyCROWN, net, lbs, ubs, input_set)
    s_poly, L, U = get_output_polys(net, net[1].α, net[2].α, lbs, ubs, solver, input_set)

    f_lower = x -> NP.evaluate(L, x)[1]
    f_upper = x -> NP.evaluate(U, x)[1]

    return f_lower, f_upper
end

"""
    plot_candidate(c; plot_lower=false, loss_fun, n_steps)

Plot the NN output and the DPNFV/aCROWN/PolyCROWN upper (and optionally lower)
bounding functions for a candidate `c` (registry record or `(Ws, bs)` named tuple).
"""
function plot_candidate(c; plot_lower=false, loss_fun=NP.upper_bound_loss, n_steps=1000)
    max_vars = 10
    iset = input_set_of(c)
    l, u = low(iset)[1], high(iset)[1]

    net_npi = make_network(c.Ws, c.bs) |> NV.NetworkNegPosIdx
    net_acrown = make_crown_chain(c.Ws, c.bs; poly=false)
    net_pcrown = make_crown_chain(c.Ws, c.bs; poly=true)

    input_set = Hyperrectangle(low=[l], high=[u])

    f_dpnfv_lower, f_dpnfv_upper =
        get_bounding_functions(NP.DPNFV(max_vars=max_vars), net_npi, nothing, nothing, input_set)

    pCROWN = NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true), poly_layers=1, prune_neurons=false)
    params_poly = NP.OptimisationParams(n_steps=n_steps, print_freq=100, start_lr=0.1,
                                        decay=0.98, patience=1000, verbosity=0)
    _, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(pCROWN, net_pcrown, input_set,
                                                   params=params_poly, loss_fun=loss_fun)
    f_pcrown_lower, f_pcrown_upper = get_bounding_functions(pCROWN, net_pcrown, lbs_pcrown, ubs_pcrown, input_set)

    aCROWN = NP.aCROWN()
    params = NP.OptimisationParams(n_steps=n_steps, print_freq=100, start_lr=0.1,
                                   decay=0.98, patience=1000, verbosity=0)
    _, lbs_acrown, ubs_acrown = NP.optimise_bounds(aCROWN, net_acrown, input_set,
                                                   params=params, loss_fun=loss_fun)
    f_acrown_lower, f_acrown_upper = get_bounding_functions(aCROWN, net_acrown, lbs_acrown, ubs_acrown, input_set)

    xs = range(l, u; length=200)
    y_nn = net_acrown(reshape(xs, 1, :)) |> vec

    pl = Plots.plot(xs, y_nn, label="NN output")
    Plots.plot!(xs, f_dpnfv_upper.(xs), label="DPNFV upper bound")
    Plots.plot!(xs, f_acrown_upper.(xs), label="aCROWN upper bound")
    Plots.plot!(xs, f_pcrown_upper.(xs), label="pCROWN upper bound")

    if plot_lower
        Plots.plot!(xs, f_dpnfv_lower.(xs), label="DPNFV lower bound")
        Plots.plot!(xs, f_acrown_lower.(xs), label="aCROWN lower bound")
        Plots.plot!(xs, f_pcrown_lower.(xs), label="pCROWN lower bound")
    end

    return pl
end

"""
    plot_candidate(c, options; kwargs...)

Plot relaxations for a candidate network and a configurable set of verification
algorithms/parameters.

`options` maps algorithm name => parameter sweeps, e.g.

    Dict("DPNFV"  => Dict("max_vars" => [0, 10]),
         "aCROWN" => Dict("n_steps" => [0, 25, 1000]),
         "pCROWN" => Dict("n_steps" => [0, 25, 1000]))

Keyword arguments: `plot_lower`, `loss_fun`, `linewidth`, `no_legend`, `ylims`,
`framestyle`, `input_bounds` (list of `(l, u)` tuples), `alpha_decay`.
"""
function plot_candidate(c, options::Dict; plot_lower=false, loss_fun=NP.upper_bound_loss,
                        linewidth=2, no_legend=false, ylims=nothing, framestyle=:axes,
                        input_bounds=nothing, alpha_decay=true)
    isnothing(input_bounds) && (input_bounds = [(low(input_set_of(c))[1], high(input_set_of(c))[1])])
    net_npi = make_network(c.Ws, c.bs) |> NV.NetworkNegPosIdx
    net_acrown = make_crown_chain(c.Ws, c.bs; poly=false)
    net_pcrown = make_crown_chain(c.Ws, c.bs; poly=true)

    dpnfv_functions = []
    acrown_functions = []
    pcrown_functions = []

    for (l, u) in input_bounds
        input_set = Hyperrectangle(low=[l], high=[u])

        if "DPNFV" in keys(options)
            for max_vars in options["DPNFV"]["max_vars"]
                solver = "method" in keys(options["DPNFV"]) ?
                    NP.DPNFV(max_vars=max_vars, method=options["DPNFV"]["method"]) :
                    NP.DPNFV(max_vars=max_vars)
                f_dpnfv_lower, f_dpnfv_upper = get_bounding_functions(solver, net_npi, nothing, nothing, input_set)
                push!(dpnfv_functions, (max_vars, f_dpnfv_lower, f_dpnfv_upper, l, u))
            end
        end

        if "aCROWN" in keys(options)
            for n_steps in options["aCROWN"]["n_steps"]
                aCROWN = NP.aCROWN()
                params = NP.OptimisationParams(n_steps=n_steps, print_freq=100, start_lr=0.1,
                                               decay=0.98, patience=1000, verbosity=0)
                _, lbs_acrown, ubs_acrown = NP.optimise_bounds(aCROWN, net_acrown, input_set,
                                                               params=params, loss_fun=loss_fun)
                f_acrown_lower, f_acrown_upper = get_bounding_functions(aCROWN, net_acrown, lbs_acrown, ubs_acrown, input_set)
                push!(acrown_functions, (n_steps, f_acrown_lower, f_acrown_upper, l, u))
            end
        end

        if "pCROWN" in keys(options)
            for n_steps in options["pCROWN"]["n_steps"]
                start_lr = haskey(options["pCROWN"], "start_lr") ? options["pCROWN"]["start_lr"] : 0.1
                pCROWN = NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true), poly_layers=1, prune_neurons=false)
                params_poly = NP.OptimisationParams(n_steps=n_steps, print_freq=100, start_lr=start_lr,
                                                    decay=0.98, patience=1000, verbosity=0)
                _, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(pCROWN, net_pcrown, input_set,
                                                               params=params_poly, loss_fun=loss_fun)
                f_pcrown_lower, f_pcrown_upper = get_bounding_functions(pCROWN, net_pcrown, lbs_pcrown, ubs_pcrown, input_set)
                push!(pcrown_functions, (n_steps, f_pcrown_lower, f_pcrown_upper, l, u))
            end
        end
    end

    l = minimum(first.(input_bounds))
    u = maximum(last.(input_bounds))
    xs = range(l, u; length=200)

    y_nn = net_acrown(reshape(xs, 1, :)) |> vec
    if no_legend
        pl = Plots.plot(xs, y_nn, label="NN output", color=1, linewidth=linewidth, legend=false,
                  framestyle=framestyle, xlabel="x", ylabel="y")
    else
        pl = Plots.plot(xs, y_nn, label="NN output", color=1, linewidth=linewidth,
                  framestyle=framestyle, xlabel="x", ylabel="y")
    end

    if !isnothing(ylims)
        Plots.ylims!(pl, ylims)
    end

    alg_dict = Dict("DPNFV" => dpnfv_functions, "aCROWN" => acrown_functions, "pCROWN" => pcrown_functions)

    for (i, algo) in enumerate(keys(alg_dict))
        len_params = length(alg_dict[algo])
        for (j, (n, fl, fu, l, u)) in enumerate(alg_dict[algo])
            xs_algo = range(l, u, 200)
            param_str = algo == "DPNFV" ? "$n vars" : "$n steps"

            # last setting is solid, earlier settings are more transparent
            alpha = alpha_decay ? j / len_params : 1.0

            if plot_lower
                Plots.plot!(pl, xs_algo, fl.(xs_algo), label=nothing, color=i+1, alpha=alpha, linewidth=linewidth)
            end

            Plots.plot!(pl, xs_algo, fu.(xs_algo), label="$(algo) bound ($(param_str))",
                  color=i+1, alpha=alpha, linewidth=linewidth)
        end
    end

    return pl
end
