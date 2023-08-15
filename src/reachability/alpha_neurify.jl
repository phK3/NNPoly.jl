
@with_kw struct AlphaNeurify <: NV.Solver
    initialize = false
    # always use tightest bounds found during optimisation at intermediate neurons
    use_tightened_bounds = true
end


function forward_linear(solver::AlphaNeurify, L::NV.LayerNegPosIdx, input)
    Λ = L.W_neg * input.Γ .+ L.W_pos * input.Λ
    λ = L.W_neg * input.γ .+ L.W_pos * input.λ .+ L.bias
    Γ = L.W_neg * input.Λ .+ L.W_pos * input.Γ
    γ = L.W_neg * input.λ .+ L.W_pos * input.γ .+ L.bias

    return SymbolicIntervalDiff(Λ, λ, Γ, γ, input.lb, input.ub, input.lbs, input.ubs)
end


function forward_act(solver::AlphaNeurify, L::NV.LayerNegPosIdx{NV.ReLU}, input, α)
    ll̂, lû = bounds(input.Λ, input.λ, input.lb, input.ub)
    ul̂, uû = bounds(input.Γ, input.γ, input.lb, input.ub)

    if solver.use_tightened_bounds
        # the input.lbs or input.ubs are constants, we should be able to ignore them for the gradient
        ll_prev = zeros(length(ll̂))
        uu_prev = zeros(length(uû))
        ChainRulesCore.ignore_derivatives() do
            ll_prev = input.lbs[L.index]
            uu_prev = input.ubs[L.index]
        end

        ll = max.(ll_prev, ll̂)
        lu = max.(ll, lû)
        #ll = max.(input.lbs[L.index], ll̂)
        #lu = max.(ll, lû)

        uu = min.(uu_prev, uû)
        ul = min.(ul̂, uu)
        #uu = min.(input.ubs[L.index], uû)
        #ul = min.(ul̂, uû)
    else
        ll = ll̂
        lu = lû
        ul = ul̂
        uu = uû
    end

    if solver.initialize
        aₗ = ChainRulesCore.ignore_derivatives() do
            aₗ = NV.relaxed_relu_gradient_lower.(ll, lu)
            α .= aₗ
            aₗ
        end
    else
        aₗ = [if l >= 0 1. elseif u <= 0 0. else αᵢ end for (l, u, αᵢ) in zip(ll, lu, α)]
    end

    aᵤ = [NV.relaxed_relu_gradient(ulᵢ, uuᵢ) for (ulᵢ, uuᵢ) in zip(ul, uu)]
    # aᵤ = relaxed_relu_gradient.(ul, uu)
    bᵤ = -ul

    Λ = aₗ .* input.Λ
    λ = aₗ .* input.λ

    Γ = aᵤ .* input.Γ
    γ = aᵤ .* (input.γ .+ max.(0, bᵤ))  # should be the same as below
    # γ = aᵤ .* input.γ .+ aᵤ .* max.(0, bᵤ)

    ChainRulesCore.ignore_derivatives() do
        input.lbs[L.index] .= ll
        input.ubs[L.index] .= uu
    end

    return SymbolicIntervalDiff(Λ, λ, Γ, γ, input.lb, input.ub, input.lbs, input.ubs)
end


function forward_act(solver::AlphaNeurify, L::NV.LayerNegPosIdx{NV.Id}, input, α)
    return input
end


"""
Converts vector of parameters to list of vectors of parameters for each layer.

Different method than for polynomials as only the lower bound needs a parameter
instead of both the lower and the upper bound.
"""
function vec2propagation(net, α::AbstractVector)
    layer_sizes = [length(l.bias) for l in net.layers]
    extended_layer_sizes = [0; layer_sizes]
    cls = cumsum(extended_layer_sizes)

    return [α[cls[i]+1:cls[i+1]] for i in 1:length(layer_sizes)]
end


# here αs is a vector! (in contrast to forward_network)
# different bounding than for polynomials
function propagate(solver::AlphaNeurify, net::NV.NetworkNegPosIdx, input, αs; printing=false)
    α = vec2propagation(net, αs)
    s = forward_network(solver, net, input, α)

    ll, lu = bounds(s.Λ, s.λ, s.lb, s.ub)
    ul, uu = bounds(s.Γ, s.γ, s.lb, s.ub)

    printing && println("lbs = ", ll)
    printing && println("ubs = ", uu)

    # for now, minimize range between all outputs
    loss = sum(uu - ll)
    return loss
end


function propagate(solver::AlphaNeurify, net::NV.NetworkNegPosIdx, input::AbstractHyperrectangle, αs; printing=false)
    s = init_symbolic_interval_diff(net, input)
    return propagate(solver, net, s, αs, printing=printing)
end


"""
Initialize the symbolic domain corresponding to the given solver with the respective input set.
"""
function initialize_symbolic_domain(solver::AlphaNeurify, net::NV.NetworkNegPosIdx, input::AbstractHyperrectangle)
    return init_symbolic_interval_diff(net, input)
end


"""
Initialize slopes for lower ReLU relaxation using DeepPoly heuristic.

The degree argument is not relevant as AlphaNeurify only uses linear relaxations.
"""
function initialize_params(solver::AlphaNeurify, net::NV.NetworkNegPosIdx, degree::N, input) where N<:Number
    n_neurons = sum(length(l.bias) for l in net.layers)
    α = zeros(n_neurons)
    αs = vec2propagation(net, α)
    isolver = AlphaNeurify(initialize=true, use_tightened_bounds=solver.use_tightened_bounds)
    ŝ = forward_network(isolver, net, input, αs)
    # convert back to vector
    return reduce(vcat, vec.(αs))
end


function optimise_bounds(solver::AlphaNeurify, net::NV.NetworkNegPosIdx, input_set; opt=nothing,
                         print_freq=50, n_steps=100, patience=50, timeout=60)
    opt = isnothing(opt) ? OptimiserChain(Adam(), Projection(0., 1.)) : opt
    s = init_symbolic_interval_diff(net, input_set)

    α0 = initialize_params(solver, net, 1, s)

    optfun = α -> propagate(solver, net, s, α)

    α, y_hist, g_hist, d_hist, csims = optimise(optfun, opt, α0, print_freq=print_freq, n_steps=n_steps,
                                                patience=patience, timeout=timeout)
end
