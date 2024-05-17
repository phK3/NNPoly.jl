
@with_kw struct PolyCROWN <: NV.Solver 
    separate_alpha = true
    use_tightened_bounds = true
    prune_neurons = true  # prune fixed inactive ReLUs
    # number of layers for polynomial relaxation
    poly_layers = 1
    # solver for polynomial part
    poly_solver = DiffNNPolySym()
    # solver for linear part
    lin_solver = aCROWN()
end


function PolyCROWN(psolver ; separate_alpha=true, use_tightened_bounds=true, initialize=false, prune_neurons=true, poly_layers=1)
    return PolyCROWN(separate_alpha, use_tightened_bounds, prune_neurons, poly_layers, psolver, 
                     aCROWN(separate_alpha=separate_alpha, use_tightened_bounds=use_tightened_bounds, initialize=initialize))
end


# TODO: move to DiffNNPolySym part
function forward_linear(solver::DiffNNPolySym, L::CROWNLayer, input::DiffPolyInterval)
    if solver.common_generators
        Low, Up = interval_map_common(min.(0, L.weights), max.(0, L.weights), input.poly_interval.Low, input.poly_interval.Up, L.bias)
    else
        Low, Up = interval_map(min.(0, L.weights), max.(0, L.weights), input.poly_interval.Low, input.poly_interval.Up, L.bias)
    end
    return DiffPolyInterval(Low, Up, input.lbs, input.ubs)
end


"""
Perform symbolic bounds propagation using forward propagation of polynomials for the first layers of the network and
linear backsubstitution for the later layers.

args:
    solver - solver to use
    net_poly - network consisting of earlier layers
    net - network consisting of later layers
    input_set - DiffPolyInterval for net_poly
    αs_poly - parameters for polynomial relaxation (if separate_alpha, then even length)
    αs - parameters for linear relaxation (if separate_alpha, then even length)
"""
function NV.forward_network(solver::PolyCROWN, net_poly::NN, net::NN, input_set::DiffPolyInterval{N}; 
                            from_layer=1, lbs, ubs, printing=false) where {NN<:Union{NV.Network, NV.NetworkNegPosIdx}, N<:Number}
    # don't store bounds for polynomial layers in lbs/best_lbs, they are already
    # stored in the DiffPolyInterval
    nₗ = length(net_poly.layers)
    best_lbs = isnothing(lbs) ? Vector{Vector{N}}() : lbs[nₗ+1:end]
    best_ubs = isnothing(ubs) ? Vector{Vector{N}}() : ubs[nₗ+1:end]
    lbs = Vector{Vector{N}}()
    ubs = Vector{Vector{N}}()

    psolver = solver.poly_solver
    lsolver = solver.lin_solver  
    
    sₗ = NV.forward_network(psolver, net_poly, input_set, αs_polyₗ)
    sᵤ = NV.forward_network(psolver, net_poly, input_set, αs_polyᵤ)

    Lₗ = sₗ.poly_interval.Low
    Uₗ = sₗ.poly_interval.Up
    Lᵤ = sᵤ.poly_interval.Low
    Uᵤ = sᵤ.poly_interval.Up

    # define them here, s.t. we can reference them in the return statement
    # the remainder of the code would also work if we didn't define them here at all
    # just somehow define them with the same type as later, it's not important that they
    # have the values of sₗ and sᵤ
    l_poly = SparsePolynomial(copy(Lₗ.G), copy(Lₗ.E), copy(Lₗ.ids))
    u_poly = SparsePolynomial(copy(Uᵤ.G), copy(Uᵤ.E), copy(Uᵤ.ids))
    # careful with from_layer and i!!!
    for (i, l) in enumerate(net.layers[from_layer:end])
        if printing
            println("Layer ", from_layer + i-1)
        end

        nn_part = NN(net.layers[from_layer:i])
        
        Zl = backward_network(lsolver, nn_part, lbs[1:i-1], ubs[1:i-1], input_set, αsₗ[1:i])
        Zu = backward_network(lsolver, nn_part, lbs[1:i-1], ubs[1:i-1], input_set, αsᵤ[1:i], upper=true)
        
        if solver.poly_solver.common_generators
            l_poly = interval_map_common_lower(min.(0, Zl.Λ), max.(0, Zl.Λ), Lₗ, Uₗ, Zl.γ)
            u_poly = interval_map_common_upper(min.(0, Zu.Λ), max.(0, Zu.Λ), Lₗ, Uₗ, Zu.γ)
        else
            l_poly = exact_addition(affine_map(max.(0, Zl.Λ), Lₗ, Zl.γ), linear_map(min.(0, Zl.Λ), Uₗ))
            u_poly = exact_addition(affine_map(max.(0, Zu.Λ), Uᵤ, Zu.γ), linear_map(min.(0, Zu.Λ), Lᵤ))
        end
        
        ll, lu = bounds(l_poly)
        ul, uu = bounds(u_poly)
        
        lbs = vcat(lbs, [ll])
        ubs = vcat(ubs, [uu])
        
        if solver.use_tightened_bounds
            # keep upper approach for better derivatives, need lower approach
            # for continuing with tighter bounds
            ChainRulesCore.ignore_derivatives() do
                if length(best_lbs) < i
                    push!(best_lbs, ll)
                    push!(best_ubs, uu)
                else                    
                    best_lb = max.(lbs[i], best_lbs[i])
                    best_ub = min.(ubs[i], best_ubs[i])
                    best_lbs[i] .= best_lb
                    best_ubs[i] .= best_ub
                    lbs[i] = best_lb
                    ubs[i] = best_ub
                end
            end
        end
    end

    # want also bounds for first part of network
    lbs = [sₗ.lbs[1:end]; best_lbs]
    ubs = [sᵤ.ubs[1:end]; best_ubs]
    return DiffPolyInterval(l_poly, u_poly, lbs, ubs)
end


function initialize_symbolic_domain(solver::PolyCROWN, net::Chain, input::AbstractHyperrectangle)
    return initialize_symbolic_domain(solver.poly_solver, net[1:solver.poly_layers], input)
end


"""
Initialises parameters for the polynomial relaxations for the first layers of the network
and the slopes of the linear relaxations for the later layers of the network.

Also computes the lower and upper bounds generated by the initialisation run.

args:
    net_poly - part of the network for polynomial overapproximation
    net - later layers of the network for linear overapproximation using backsubstitution
    degree - degree of the polynomial relaxation
    input - input set of the property

returns:
    α_poly - vector of initial parameters for the polynomial relaxations
    α - vector of parameters for the linear relaxations
    lbs - vector of vector of concrete lower bounds obtained during initialisation
    ubs - vector of vector of concrete upper bounds obtained during initialisation
"""
function initialize_params(solver::PolyCROWN, net_poly, net, degree::N, input::DiffPolyInterval) where N <: Number
    psolver = solver.poly_solver
    α_poly = initialize_params(psolver, net_poly, degree, input)
    αps = vec2propagation(net_poly, degree, α_poly)
    
    n_neurons = sum(length(l.bias) for l in net.layers)
    α = zeros(n_neurons)
    αs = vec2propagation(net, α)
    # for initialization separate_alpha isn't needed
    isolver = PolyCROWN(psolver, initialize=true, separate_alpha=false)
    
    ŝ = NV.forward_network(isolver, net_poly, net, input, αps, αs);

    α_lin = reduce(vcat, vec.(αs))
    α_poly = solver.separate_alpha ? [α_poly; α_poly] : α_poly
    α_lin = solver.separate_alpha ? [α_lin; α_lin] : α_lin
        
    return α_poly, α_lin, ŝ.lbs, ŝ.ubs
end


function initialize_params(solver::PolyCROWN, net::NV.NetworkNegPosIdx, degree::N, input::DiffPolyInterval; return_bounds=false) where N <: Number
    #for compatibility with vnnlib.jl
    net_poly = NV.NetworkNegPosIdx(net.layers[1:solver.poly_layers])
    net = NV.NetworkNegPosIdx(net.layers[solver.poly_layers+1:end])

    α_poly, α_lin, lbs, ubs = initialize_params(solver, net_poly, net, degree, input)
    α0 = [α_poly; α_lin]

    if return_bounds
        return α0, lbs, ubs
    else
        return α0
    end
end


function initialize_params_bounds(solver::PolyCROWN, net, degree::N, input) where N<:Number
    psolver = solver.poly_solver
    # just copy the solver, but set init=true
    ipsolver = DiffNNPolySym(truncation_terms=psolver.truncation_terms,
                                separate_relaxations=psolver.separate_relaxations,
                                relaxations=psolver.relaxations, splitting_depth=psolver.splitting_depth,
                                init=true, init_method=psolver.init_method, 
                                save_bounds=psolver.save_bounds,
                                common_generators=psolver.common_generators)
    ŝ = forward_linear(ipsolver, net[1], input)

    # don't know the sizes of these arrays beforehand, so just let them empty. Real numbers get pushed during initialization.
    rs = Vector{Int}()
    cs = Vector{Int}()
    symmetric_factor = Vector{Int}()
    unique_idxs = Vector{Int}()
    duplicate_idxs = Vector{Int}()

    # for first layer, bounds from s.Low and s.Up are the same
    l, u = bounds(ŝ.poly_interval.Low)

    s_poly = forward_act_stub(ipsolver, net[1], ŝ, l, u, rs, cs, symmetric_factor, unique_idxs, duplicate_idxs)
    
    lbs_lin, ubs_lin = initialize_params_bounds(solver.lin_solver, net[2:end], 1, s_poly)
    return ŝ, [[l]; lbs_lin], [[u]; ubs_lin], rs, cs, symmetric_factor, unique_idxs, duplicate_idxs
end


function propagate(solver::PolyCROWN, net_poly::NV.NetworkNegPosIdx, 
                    net::NV.NetworkNegPosIdx, input::DiffPolyInterval, α_poly, α; printing=false, 
                    lbs=nothing, ubs=nothing)
    s = NV.forward_network(solver, net_poly, net, input, αps, αs, lbs=lbs, ubs=ubs);

    ll, lu = bounds(s.poly_interval.Low)
    ul, uu = bounds(s.poly_interval.Up)

    printing && println("lbs = ", ll)
    printing && println("ubs = ", uu)

    # for now, minimize range between all outputs
    loss = sum(uu - ll)
    return loss
end


function propagate(solver::PolyCROWN, net::NV.NetworkNegPosIdx, input::DiffPolyInterval, α; printing=false, lbs=nothing, ubs=nothing)
    # dummy method for compatibility with vnnlib.jl
    net_poly = NV.NetworkNegPosIdx(net.layers[1:solver.poly_layers])
    net = NV.NetworkNegPosIdx(net.layers[solver.poly_layers+1:end])

    n_neurons_poly = sum(length(l.bias) for l in net_poly.layers)

    degree = 2
    nₚ = solver.separate_alpha ? 2*2*degree*n_neurons_poly : 2*degree*n_neurons_poly

    αp = α[1:nₚ]
    αl = α[nₚ+1:end]
    return propagate(solver, net_poly, net, input, αp, αl, printing=printing, lbs=lbs, ubs=ubs)
end


function propagate(solver::PolyCROWN, net::Chain, input::DiffPolyInterval, lbs, ubs; printing=false)
    ŝ = forward_linear(solver.poly_solver, net[1], input)

    # for first layer, bounds from s.Low and s.Up are the same
    l, u = bounds(ŝ.Low)

    s_poly = forward_act_stub(solver.poly_solver, net[1], ŝ, l, u)
    s_crown = NV.forward_network(solver.lin_solver, m[2:end], s_poly, lbs, ubs)

    ll, lu = bounds(s_crown.Λ, s_crown.λ, s_poly)
    ul, uu = bounds(s_crown.Γ, s_crown.γ, s_poly)

    printing && println("lbs = ", ll)
    printing && println("ubs = ", uu)

    loss = sum(uu .- ll)
    return loss
end


"""
Propagates a given DiffPolyInterval with precomputed bounds through the 1st non-linear layer (with ReLU activation function).

Don't use this if you want to propagate through a later non-linear layer (even if before were not ReLU-nonlinearities)!!!

The method utilises that propagating the input through the first linear layer always results in the same reachable set 
regardless of α-parameters defining the shape of the relaxation of the activation function.
Therefore, we can precompute that set and also precompute its bounds and only have to propagate that through the ReLU layer.
"""
function forward_act_stub(solver::DiffNNPolySym, L::CROWNLayer{NV.ReLU, MN, BN, AN}, input::DiffPolyInterval, l, u, rs, cs, symmetric_factor, unique_idxs, duplicate_idxs) where {MN,BN,AN}
    s = input.poly_interval
    if solver.init && solver.init_method == :CROWNQuad
        # CROWNQuad initialisation
        cₗ = relax_relu_crown_quad_lower_matrix(l, u)
        cᵤ = relax_relu_crown_quad_upper_matrix(l, u)

        # CROWNQuad is quadratic relaxation, so set first two params
        L.α[:, 1:2, 1] .= cₗ[:,2:3]
        L.α[:, 1:2, 2] .= cᵤ[:,2:3]
    elseif solver.init && solver.init_method == :linear
        # linear CROWN initialisation
        cₗ = NV.relaxed_relu_gradient_lower.(l, u)
        cᵤ = NV.relaxed_relu_gradient.(l, u)
        
        # only need to set the slope, shifting takes care of the rest
        L.α[:,1,1] .= cₗ
        L.α[:,1,2] .= cᵤ
        # need to set quad-part to zero, because is only initialized with similar(...)
        L.α[:,2,1] .= 0
        L.α[:,2,2] .= 0

        # need full monomials to propagate through quad_prop_common
        cₗ = get_lower_polynomial_shift(l, u, 2, L.α[:, :, 1])
        cᵤ = get_upper_polynomial_shift(l, u, 2, L.α[:, :, 2])
    else
        cₗ = get_lower_polynomial_shift(l, u, 2, L.α[:, :, 1])
        cᵤ = get_upper_polynomial_shift(l, u, 2, L.α[:, :, 2])
    end

    L̂, Û = quad_prop_common!(cₗ, cᵤ, s.Low, s.Up, rs, cs, symmetric_factor, unique_idxs, duplicate_idxs, init=solver.init)

    return DiffPolyInterval(L̂, Û, input.lbs, input.ubs)
end


function optimise_bounds(solver::PolyCROWN, net::Chain, input_set::Hyperrectangle; opt=Optimisers.Adam(), params=OptimisationParams(), loss_fun=bounds_loss, print_results=false)
    psolver = solver.poly_solver
    # TODO: implement method for Chain
    s = initialize_symbolic_domain(solver, net[1:solver.poly_layers], input_set)

    # TODO: is there some better way of returning all those precomputed values?
    # bounds before activation in first layer are just interval bounds and don't change
    # with different α parameters, so we can just reuse ŝ (the reachable set after the 1st linear layer),
    # l and u (the bounds after the 1st linear layer) throughout the optimization loop
    ŝ, lbs, ubs, rs, cs, symmetric_factor, unique_idxs, duplicate_idxs = initialize_params_bounds(solver, net, 2, s)

    if solver.prune_neurons
        # TODO: maybe add as callback to optimisation?
        ŝ = select_idxs(ŝ, .~(ubs[1] .<= 0), 1)
        net, lbs, ubs = prune(ZeroPruner(), net, lbs, ubs)
    end

    optfun = m -> begin
        if all(ubs[end] .- lbs[end] .== 0)
            println("Output bounds are exact!")
            return 0.
        end

        s_poly = forward_act_stub(solver.poly_solver, m[1], ŝ, lbs[1], ubs[1], rs, cs, symmetric_factor, unique_idxs, duplicate_idxs)
        s_crown = NV.forward_network(solver.lin_solver, m[2:end], s_poly, lbs[2:end], ubs[2:end])

        ll, lu = bounds(s_crown.Λ, s_crown.λ, s_poly)
        ul, uu = bounds(s_crown.Γ, s_crown.γ, s_poly)

        #loss = sum(uu .- ll)
        #loss = sum(max.(0., uu))  # loss for verifying Ay - b ≤ 0 properties
        return loss_fun(ll, uu)
    end

    res = optimise(optfun, net, opt, params=params)

    print_results && println("lbs = ", lbs[end])
    print_results && println("ubs = ", ubs[end])
    
    return res, lbs, ubs
end


function optimise_bounds(solver::PolyCROWN, net::NV.NetworkNegPosIdx, input_set::Hyperrectangle; opt=nothing,
                        params=OptimisationParams(), print_result=false, poly_layers=1)
    psolver = solver.poly_solver
    
    opt = isnothing(opt) ? OptimiserChain(Adam()) : opt

    net_poly = NV.NetworkNegPosIdx(net.layers[1:poly_layers])
    net = NV.NetworkNegPosIdx(net.layers[poly_layers+1:end])
    s = initialize_symbolic_domain(psolver, net_poly, input_set)

    α_poly, α_lin, lbs0, ubs0 = initialize_params(solver, net_poly, net, 2, s)
    α0 = [α_poly; α_lin]

    nₚ = length(α_poly)
    nₗ = length(α_lin)

    if solver.use_tightened_bounds
        optfun = α -> begin
            αp = α[1:nₚ]
            αl = α[nₚ+1:end]
            # lbs0, ubs0 will be overwritten with tighter bounds during the optimization process
            propagate(solver, net_poly, net, s, αp, αl, lbs=lbs0, ubs=ubs0)
        end
    else
        optfun = α -> begin
            αp = α[1:nₚ]
            αl = α[nₚ+1:end]
            propagate(solver, net_poly, net, s, αp, αl)
        end
    end

    res = optimise(optfun, opt, α0, params=params)

    if print_result
        α = res.x_opt
        αp = α[1:nₚ]
        αl = α[nₚ+1:end]
        propagate(solver, net_poly, net, s, αp, αl, lbs=lbs0, ubs=ubs0, printing=true)
    end

    return res
end