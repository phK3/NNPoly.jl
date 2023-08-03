


struct DiffPolyInterval{N<:Number}
    poly_interval::PolyInterval{N}  # NNPoly.PolyInterval
    lbs::Vector{Vector{N}}  # lower bounds for intermediate values
    ubs::Vector{Vector{N}}  # upper bounds for intermediate values
end


function DiffPolyInterval(Low, Up, lbs, ubs)
    return DiffPolyInterval(PolyInterval(Low, Up), lbs, ubs)
end


function DiffPolyInterval(net, input_set)
    layer_sizes = [length(l.bias) for l in net.layers]
    lbs = [fill(-Inf, ls) for ls in layer_sizes]
    ubs = [fill( Inf, ls) for ls in layer_sizes]
    s = init_poly_interval(input_set)
    return DiffPolyInterval(s, lbs, ubs)
end
