


struct DiffPolyInterval{N<:Number,M<:Integer}
    poly_interval::PolyInterval{N,M}  # NNPoly.PolyInterval
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


function DiffPolyInterval(net::Chain, input_set)
    lbs = [similar(L.bias) for L in net.layers]
    ubs = [similar(L.bias) for L in net.layers]

    for i in eachindex(lbs)
        # is there any way to directly set them to ±Inf while also getting  the possibly gpu type right?
        lbs[i] .= -Inf
        ubs[i] .= Inf
    end

    s = init_poly_interval(input_set)
    return DiffPolyInterval(s, lbs, ubs)
end


"""
Calculates concrete bounds for A*s + b for DiffPolyInterval s with common generators.
"""
function bounds(A::AbstractMatrix, b::AbstractVector, s::DiffPolyInterval)
    L, U = interval_map_common(min.(0, A), max.(0, A), s.poly_interval.Low, s.poly_interval.Up, b)
    ll, lu = bounds(L)
    ul, uu = bounds(U)
    return ll, uu
end
