
# ==========================================================================================
# Upper-bound evaluation for all ladder rungs
#
# All functions return *concrete upper bounds on the first output* as Float64
# (converted from exact rationals where necessary), plus optional details.
# ==========================================================================================

"""
Default input set for the running example: x ∈ [-1, 1].
"""
default_input_set(low=-1.0, high=1.0) = Hyperrectangle(low=[low], high=[high])

"""
Upper bound via standard interval propagation (rung i).
"""
function box_upper_bound(net, input_set)
    ŝ = NV.forward_network(NV.Box(), net, input_set)
    return Float64(high(ŝ)[1])
end

"""
Upper bound via DPNFV (rungs ii, iii, iii'). Returns the bound and the final
symbolic state (which holds the per-layer concrete bounds).
"""
function dpnfv_upper_bound(net_npi, input_set; max_vars, method=:NeurifyRelax, state=false)
    s = NP.init_symbolic_interval_fvheur(net_npi, input_set, max_vars=max_vars)
    ŝ = NV.forward_network(NP.DPNFV(max_vars=max_vars, method=method), net_npi, s)
    lbs, ubs = NP.bounds(ŝ)
    return state ? (ub=Float64(ubs[end][1]), ŝ=ŝ) : Float64(ubs[end][1])
end

"""
Evaluate all *forward* rungs (i)–(iv) for a candidate network.
"""
function evaluate_forward(Ws, bs, input_set=default_input_set())
    net = make_network(Ws, bs)
    net_npi = NV.NetworkNegPosIdx(net)

    ub_box = box_upper_bound(net, input_set)
    ub_outer0 = dpnfv_upper_bound(net_npi, input_set; max_vars=0, method=:OuterBoundsNeurify)
    ub_sep0   = dpnfv_upper_bound(net_npi, input_set; max_vars=0, method=:NeurifyRelax)
    ub_fv     = dpnfv_upper_bound(net_npi, input_set; max_vars=10, method=:NeurifyRelax)
    ub_fv_dp  = dpnfv_upper_bound(net_npi, input_set; max_vars=10, method=:DeepPolyRelax)
    ub_zono   = Float64(zono_upper_bound(net, input_set)[1])

    return (ub_box=ub_box, ub_outer0=ub_outer0, ub_sep0=ub_sep0,
            ub_fv=ub_fv, ub_fv_dp=ub_fv_dp, ub_zono=ub_zono)
end

_opt_params0() = NP.OptimisationParams(n_steps=0, timeout=30.0, patience=10,
                                       print_freq=1000, verbosity=0)
_opt_params_opt(n_steps) = NP.OptimisationParams(n_steps=n_steps, timeout=30.0, patience=10,
                                                 print_freq=1000, verbosity=0,
                                                 start_lr=0.1, decay=0.98)

const _PCROWN_SOLVER() = NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true), poly_layers=1)

"""
Rungs (v) and (vi): backsubstitution bounds without optimisation (n_steps=0,
i.e. the initialisation forward pass of aCROWN / PolyCROWN).
"""
function evaluate_crown_init(Ws, bs, input_set=default_input_set())
    net_crown = make_crown_chain(Ws, bs; poly=false)
    net_pcrown = make_crown_chain(Ws, bs; poly=true)

    _, _, ubs_a = NP.optimise_bounds(NP.aCROWN(), net_crown, input_set,
                                     params=_opt_params0(), loss_fun=NP.upper_bound_loss)
    _, _, ubs_p = NP.optimise_bounds(_PCROWN_SOLVER(), net_pcrown, input_set,
                                     params=_opt_params0(), loss_fun=NP.upper_bound_loss)
    return (ub_acrown0=Float64(ubs_a[end][1]), ub_pcrown0=Float64(ubs_p[end][1]))
end

"""
Rungs (vii) and (viii): backsubstitution bounds with optimised relaxation parameters.
"""
function evaluate_crown_opt(Ws, bs, input_set=default_input_set(); n_steps=25)
    net_crown = make_crown_chain(Ws, bs; poly=false)
    net_pcrown = make_crown_chain(Ws, bs; poly=true)

    _, _, ubs_a = NP.optimise_bounds(NP.aCROWN(), net_crown, input_set,
                                     params=_opt_params_opt(n_steps), loss_fun=NP.upper_bound_loss)
    _, _, ubs_p = NP.optimise_bounds(_PCROWN_SOLVER(), net_pcrown, input_set,
                                     params=_opt_params_opt(n_steps), loss_fun=NP.upper_bound_loss)
    return (ub_acrown_opt=Float64(ubs_a[end][1]), ub_pcrown_opt=Float64(ubs_p[end][1]))
end

"""
Details of the aCROWN(0) backsubstitution: the linear bounding functions
Λx + λ (lower) and Γx + γ (upper) plus the concrete output bounds.
Used by the niceness analysis and presentation code.
"""
function acrown0_details(Ws, bs, input_set=default_input_set())
    net_crown = make_crown_chain(Ws, bs; poly=false)
    lbs0, ubs0 = NP.initialize_params_bounds(NP.aCROWN(), net_crown, 1, input_set)
    s = NV.forward_network(NP.aCROWN(initialize=true, separate_alpha=false),
                           net_crown, input_set, lbs0, ubs0)
    return (Λ=s.Λ, λ=s.λ, Γ=s.Γ, γ=s.γ, lbs=s.lbs, ubs=s.ubs)
end

"""
Details of a DPNFV run: per-layer concrete pre-activation bounds and the final
symbolic lower/upper bound matrices (in terms of the input only).
"""
function dpnfv_details(net_npi, input_set; max_vars, method=:NeurifyRelax)
    s = NP.init_symbolic_interval_fvheur(net_npi, input_set, max_vars=max_vars)
    ŝ = NV.forward_network(NP.DPNFV(max_vars=max_vars, method=method), net_npi, s)
    lbs, ubs = NP.bounds(ŝ)
    L, U = NP.substitute_variables(ŝ)
    return (ub=Float64(ubs[end][1]),
            lbs_per_layer=[copy(v) for v in s.lbs],
            ubs_per_layer=[copy(v) for v in s.ubs],
            L=L, U=U)
end

"""
Re-evaluate all rungs for a candidate stored in the registry.
"""
function evaluate_candidate(Ws, bs; n_steps=25, input_set=default_input_set())
    m = evaluate_forward(Ws, bs, input_set)
    m = merge(m, evaluate_crown_init(Ws, bs, input_set))
    m = merge(m, evaluate_crown_opt(Ws, bs, input_set; n_steps=n_steps))
    return m
end
