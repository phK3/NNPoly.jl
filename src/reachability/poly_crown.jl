
@with_kw struct PolyCROWN <: NV.Solver 
    separate_alpha = true
    use_tightened_bounds = true
    # number of layers for polynomial relaxation
    poly_layers = 1
    # solver for polynomial part
    poly_solver = DiffNNPolySym()
    # solver for linear part
    lin_solver = aCROWN()
end


function PolyCROWN(psolver ; separate_alpha=true, use_tightened_bounds=true, initialize=false, poly_layers=1)
    return PolyCROWN(separate_alpha, use_tightened_bounds, poly_layers, psolver, 
                     aCROWN(separate_alpha=separate_alpha, use_tightened_bounds=use_tightened_bounds, initialize=initialize))
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
function NV.forward_network(solver::PolyCROWN, net_poly::NN, net::NN, input_set::DiffPolyInterval{N}, αs_poly, αs; 
                            from_layer=1, lbs=nothing, ubs=nothing, printing=false) where {NN<:Union{NV.Network, NV.NetworkNegPosIdx}, N<:Number}
    # don't store bounds for polynomial layers in lbs/best_lbs, they are already
    # stored in the DiffPolyInterval
    nₗ = length(net_poly.layers)
    best_lbs = isnothing(lbs) ? Vector{Vector{N}}() : lbs[nₗ+1:end]
    best_ubs = isnothing(ubs) ? Vector{Vector{N}}() : ubs[nₗ+1:end]
    lbs = Vector{Vector{N}}()
    ubs = Vector{Vector{N}}()

    psolver = solver.poly_solver
    lsolver = solver.lin_solver
    
    if solver.separate_alpha
        half = Int(floor(0.5*length(αs)))
        αsₗ = αs[1:half]
        αsᵤ = αs[half+1:end]
        
        half_poly = Int(floor(0.5*length(αs_poly)))
        αs_polyₗ = αs_poly[1:half_poly]
        αs_polyᵤ = αs_poly[half_poly+1:end]
    else
        αsₗ = αs
        αsᵤ = αs
        αs_polyₗ = αs_poly
        αs_polyᵤ = αs_poly
    end
    
    
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
        
        l_poly = exact_addition(affine_map(max.(0, Zl.Λ), Lₗ, Zl.γ), linear_map(min.(0, Zl.Λ), Uₗ))
        u_poly = exact_addition(affine_map(max.(0, Zu.Λ), Uᵤ, Zu.γ), linear_map(min.(0, Zu.Λ), Lᵤ))
        
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


function initialize_symbolic_domain(solver::PolyCROWN, net::NV.NetworkNegPosIdx, input::AbstractHyperrectangle)
    # for compatibility with vnnlib.jl
    net_poly = NV.NetworkNegPosIdx(net.layers[1:solver.poly_layers])
    return initialize_symbolic_domain(solver.poly_solver, net_poly, input)
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


function propagate(solver::PolyCROWN, net_poly::NV.NetworkNegPosIdx, 
                    net::NV.NetworkNegPosIdx, input::DiffPolyInterval, α_poly, α; printing=false, 
                    lbs=nothing, ubs=nothing)
    if solver.separate_alpha
        @assert length(α) % 2 == 0 "If solver.separate_alpha, then only even lengths of α are allowed."
        @assert length(α_poly) % 2 == 0 "If solver.separate_alpha, then only even lengths of α_poly are allowed"
        half = Int(0.5*length(α))
        half_poly = Int(0.5*length(α_poly))
        αs = [vec2propagation(net, α[1:half]); vec2propagation(net, α[half+1:end])]
        αps = [vec2propagation(net_poly, 2, α_poly[1:half_poly]); vec2propagation(net_poly, 2, α_poly[half_poly+1:end])]
    else
        αs = vec2propagation(net, α)
        αps = vec2propagation(net_poly, 2, α_poly)
    end

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


function optimise_bounds(solver::PolyCROWN, net::NV.NetworkNegPosIdx, input_set::Hyperrectangle; opt=nothing,
                        print_freq=50, n_steps=100, patience=50, timeout=60, print_result=false, poly_layers=1)
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

    α, y_hist, g_hist, d_hist, csims = optimise(optfun, opt, α0, print_freq=print_freq, n_steps=n_steps,
                            patience=patience, timeout=timeout)

    if print_result
        αp = α[1:nₚ]
        αl = α[nₚ+1:end]
        propagate(solver, net_poly, net, s, αp, αl, lbs=lbs0, ubs=ubs0, printing=true)
    end

    return α, y_hist, g_hist, d_hist, csims
end