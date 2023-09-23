

function forward_layer(solver, L::Union{NV.Layer,NV.LayerNegPosIdx}, input, α)
    ŝ = forward_linear(solver, L, input)
    s = forward_act(solver, L, ŝ, α)
    return s
end


function NV.forward_network(solver, net::Union{NV.Network,NV.NetworkNegPosIdx}, input, αs)
    s = input
    for (L, α) in zip(net.layers, αs)
        s = forward_layer(solver, L, s, α)
    end

    return s
end
