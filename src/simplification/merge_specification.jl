

function merge_spec_output_layer(net::Chain, output_spec::HPolytope)
    layers = [L for L in net.layers]

    weights = net[end].weights
    biases  = net[end].bias

    A, b = tosimplehrep(output_spec)

    Ŵ = A*weights
    b̂ = A*biases .- b

    # last layer has id activation, so α doesn't matter, can just take the old one
    L_end = CROWNLayer(Ŵ, b̂, layers[end].activation, layers[end].α)

    layers[end] = L_end
    return Chain(layers...)
end