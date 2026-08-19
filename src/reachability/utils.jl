

function forward_layer(solver, L::Union{NV.Layer,NV.LayerNegPosIdx}, input, α)
    ŝ = forward_linear(solver, L, input)
    s = forward_act(solver, L, ŝ, α)
    return s
end


function NV.forward_network(solver, net::NV.AbstractNetwork{N}, input, αs) where N<:Number
    s = input
    for (L, α) in zip(net.layers, αs)
        s = forward_layer(solver, L, s, α)
    end

    return s
end


"""
    `leaky_clamp(x, l, u)`

Clamps x to [l, u] in the forward pass, but acts as the identity function on the backward pass
(like a straight through estimator).

This is benefitial since d/dx clamp(x, l, u) = 0 if x < l or x > u.
But we often would still like to have a signal to change x in some direction.
"""
function leaky_clamp(x, l, u)
    return clamp.(x, l, u)
end

function ChainRulesCore.rrule(::typeof(leaky_clamp), x, l, u)
    y = clamp.(x, l, u)

    function leaky_clamp_pullback(Δy)
        # just return output gradient for x,
        # no tangent for the function itself or the lower/upper constants
        return (NoTangent(), Δy, NoTangent(), NoTangent())
    end
    
    return y, leaky_clamp_pullback
end