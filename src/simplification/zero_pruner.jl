
struct ZeroPruner end

"""
Removes neurons from a ReLU layer that are fixed inactive and thus their output is always zero.

args:
    pruner - the pruner to use
    L - the ReLU layer
    lbs - concrete lower bounds on the input of each ReLU neuron
    ubs - concrete upper bounds on the input of each ReLU neuron
    fixed_inact_prev - mask that is set to true for every neuron that was fixed_inactive in the previous layer
"""
function prune(pruner::ZeroPruner, L::CROWNLayer{NV.ReLU,MN,BN,AN}, lbs, ubs, fixed_inact_prev) where {MN,BN,AN}
    fixed_inactive = ubs .<= 0

    Ŵ = L.weights[.~fixed_inactive,.~fixed_inact_prev]
    b̂ = L.bias[.~fixed_inactive]
    
    # needs to work for both vectors and multidimensional tensors
    mask = Tuple(ifelse(i == 1, .~fixed_inactive, :) for i in 1:ndims(L.α))
    α̂ = L.α[mask...]

    return CROWNLayer(Ŵ, b̂, L.activation, α̂), fixed_inactive, lbs[.~fixed_inactive], ubs[.~fixed_inactive]
end


function prune(pruner::ZeroPruner, L, lbs, ubs, fixed_inact_prev)
    # for non-ReLU layers, we don't know if they are fixed, so return false everywhere
    fixed_inactive = similar(L.bias, Bool)
    fixed_inactive .= false

    Ŵ = L.weights[.~fixed_inactive, .~fixed_inact_prev]
    b̂ = L.bias[.~fixed_inactive]

    # for Id layer, α = [], so just return that again
    α̂ = length(L.α) > 0 ? L.α[mask...] : similar(L.α)

    return CROWNLayer(Ŵ, b̂, L.activation, α̂), fixed_inactive, lbs, ubs
end


function prune_output_layer(pruner::ZeroPruner, L, lbs, ubs, fixed_inact_prev)
    fixed_inactive = similar(L.bias, Bool)
    fixed_inactive .= false

    # for output layer, we only want to remove the connections to pruned input neurons!!!
    # NN outputs are still needed to make statements about the specs.
    Ŵ = L.weights[:, .~fixed_inact_prev]
    
    # for Id layer, α = [], so just return that again
    α̂ = length(L.α) > 0 ? L.α[mask...] : similar(L.α)
    return CROWNLayer(Ŵ, L.bias, L.activation, α̂), fixed_inactive, lbs, ubs
end


function prune(pruner::ZeroPruner, net::Chain, lbs, ubs)
    L_first = net[1]
    m, n = size(L_first.weights)
    fixed_inact_prev = similar(L_first.bias, Bool, n)
    # view all inputs as not-fixed
    fixed_inact_prev .= false

    layers = []
    lbs_new = similar(lbs)
    ubs_new = similar(ubs)
    # don't prune the output layer!!!
    for (i, (L, lb, ub)) in enumerate(zip(net.layers[1:end-1], lbs[1:end-1], ubs[1:end-1]))
        L̂, fixed_inact_prev, lb_new, ub_new = prune(pruner, L, lb, ub, fixed_inact_prev)
        push!(layers, L̂)

        # also throw away bounds for pruned neurons
        lbs_new[i] = lb_new
        ubs_new[i] = ub_new
    end

    L̂, _, lb_new, ub_new = prune_output_layer(pruner, net[end], lbs[end], ubs[end], fixed_inact_prev)

    push!(layers, L̂)
    lbs_new[end] = lb_new
    ubs_new[end] = ub_new
    return Chain(layers...), lbs_new, ubs_new
end