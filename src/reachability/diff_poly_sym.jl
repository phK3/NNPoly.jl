

@with_kw struct DiffNNPolySym <: NV.Solver
    truncation_terms = 50
    separate_relaxations = true
    relaxations = :shift
    splitting_depth = 0
    # set to true in beginning to get first α
    init = false
    init_method = :CROWNQuad
    save_bounds = true
    common_generators = false
end


function forward_linear(solver::DiffNNPolySym, L::NV.LayerNegPosIdx, input::DiffPolyInterval)
    if solver.common_generators
        Low, Up = interval_map_common(L.W_neg, L.W_pos, input.poly_interval.Low, input.poly_interval.Up, L.bias)
    else
        Low, Up = interval_map(L.W_neg, L.W_pos, input.poly_interval.Low, input.poly_interval.Up, L.bias)
    end
    return DiffPolyInterval(Low, Up, input.lbs, input.ubs)
end


function forward_act(solver::DiffNNPolySym, L::NV.LayerNegPosIdx{NV.ReLU}, input::DiffPolyInterval, α)
    sym = input.poly_interval
    n = size(sym.Low.G, 1)
    degrees = 2*ones(Integer, n)

    if solver.common_generators
        s = truncate_desired_common(sym, solver.truncation_terms)
    else
        s = truncate_desired(sym, solver.truncation_terms)
    end

    # take bounds w/o splitting depth for being differentiable
    ll, lu = bounds(s.Low)
    ul, uu = bounds(s.Up)

    if solver.save_bounds
        ChainRulesCore.ignore_derivatives() do
            input.lbs[L.index] .= max.(ll, input.lbs[L.index])
            input.ubs[L.index] .= min.(uu, input.ubs[L.index])
        end
    end

    if solver.init
        if solver.init_method == :Chebyshev
            # Chebyshev initialisation
            resl = relax_relu_chebyshev.(ll, lu, 2*ones(Integer, n))
            resu = relax_relu_chebyshev.(ul, uu, 2*ones(Integer, n))
            α[1,1:2,:] .= vecOfVec2Mat(first.(resl))'[2:3,:]
            α[2,1:2,:] .= vecOfVec2Mat(first.(resu))'[2:3,:]

            cₗ = vecOfVec2Mat(first.(resl))
            cₗ[:,1] .-= last.(resl)
            cᵤ = vecOfVec2Mat(first.(resu))
            cᵤ[:,1] .+= last.(resu)

            cₗ = [c for c in eachrow(cₗ)]
            cᵤ = [c for c in eachrow(cᵤ)]
        elseif solver.init_method == :CROWNQuad
            # CROWNQuad initialisation
            cₗ = relax_relu_crown_quad_lower.(ll, lu)
            cᵤ = relax_relu_crown_quad_upper.(ul, uu)

            # CROWNQuad is quadratic relaxation, so set first two params
            α[1,1:2,:] .= vecOfVec2Mat(cₗ)'[2:3,:]
            α[2,1:2,:] .= vecOfVec2Mat(cᵤ)'[2:3,:]

            cₗ = vecOfVec2Mat(cₗ)
            cᵤ = vecOfVec2Mat(cᵤ)
        else
            throw(ArgumentError("Initialisation method $(solver.init_method) not known!"))
        end
    else
        #cₗ = [ifelse(l >= 0, [0., 1, 0], ifelse(u <= 0, zeros(3), get_lower_polynomial_shift(l, u, 2, a))) for (l, u, a) in zip(ll, lu, eachcol(α[1,:,:]))]
        #cᵤ = [ifelse(l >= 0, [0., 1, 0], ifelse(u <= 0, zeros(3), get_upper_polynomial_shift(l, u, 2, a))) for (l, u, a) in zip(ul, uu, eachcol(α[2,:,:]))]
        #cₗ = get_lower_polynomial_shift.(ll, lu, 2, eachcol(α[1,:,:]))
        #cᵤ = get_upper_polynomial_shift.(ul, uu, 2, eachcol(α[2,:,:]))
        cₗ = get_lower_polynomial_shift(ll, lu, 2, α[1,:,:]')
        cᵤ = get_upper_polynomial_shift(ul, uu, 2, α[2,:,:]')
    end

    #cₗ = vecOfVec2Mat(cₗ)
    #cᵤ = vecOfVec2Mat(cᵤ)

    if solver.common_generators
        L̂, Û = quad_prop_common(cₗ, cᵤ, s.Low, s.Up, ll, lu, ul, uu)
    else
        L̂ = fast_quad_prop(cₗ[:,3], cₗ[:,2], cₗ[:,1], s.Low, ll, lu)
        Û = fast_quad_prop(cᵤ[:,3], cᵤ[:,2], cᵤ[:,1], s.Up, ul, uu)
    end

    return DiffPolyInterval(L̂, Û, input.lbs, input.ubs)
end


function forward_act(solver::DiffNNPolySym, L::NV.LayerNegPosIdx{NV.Id}, input::DiffPolyInterval, α)
    return input
end


## Transform vector of multipliers into one vector for each layer.

function vec2propagation(net, degree::Integer, α::AbstractVector)
    layer_sizes = [length(l.bias) for l in net.layers]
    extended_layer_sizes = [0; layer_sizes]
    cls = cumsum(extended_layer_sizes)

    αs = [reshape(α[2*degree*cls[i]+1:2*degree*cls[i+1]], 2, degree, :) for i in 1:length(layer_sizes)]
    return αs
end


function propagation2vec(net, degree, αs)
    return reduce(vcat, vec.(αs))
end


function initialize_params(net, degree; method=:random)
    layer_sizes = [length(l.bias) for l in net.layers]
    n_neurons = sum(layer_sizes)
    # for each neuron lower and upper relaxation have degree params each
    n_params = 2*degree*n_neurons

    if method == :random
        return randn(n_params)
    elseif method == :zero
        return zeros(n_params)
    elseif method == :test
        return collect(1:n_params)
    else
        throw(ValueError("method $method not known!"))
    end
end


function initialize_params(solver::DiffNNPolySym, net, degree, s::DiffPolyInterval; method=:CROWNQuad)
    # just copy the solver, but set init=true
    dsolver = DiffNNPolySym(truncation_terms=solver.truncation_terms,
                                separate_relaxations=solver.separate_relaxations,
                                relaxations=solver.relaxations, splitting_depth=solver.splitting_depth,
                                init=true, save_bounds=solver.save_bounds,
                                common_generators=solver.common_generators)
    α0 = initialize_params(net, degree, method=:zero)
    αs = vec2propagation(net, degree, α0)

    ŝ = forward_network(dsolver, net, s, αs)

    return propagation2vec(net, degree, αs)
end


"""
Initialize the symbolic domain corresponding to the given solver with the respective input set.
"""
function initialize_symbolic_domain(solver::DiffNNPolySym, net::NV.NetworkNegPosIdx, input::AbstractHyperrectangle)
    return DiffPolyInterval(net, input)(net, input)
end



## Propagation through Network
#  We need separate propagate() method that takes in a vector of α instead of
#  one vector per layer.

# here αs is a vector! (in contrast to forward_network)
function propagate(solver::DiffNNPolySym, net::NV.NetworkNegPosIdx, input, α; printing=false)
    αs = vec2propagation(net, 2, α)
    s = forward_network(solver, net, input, αs)

    ll, lu = bounds(s.poly_interval.Low, solver.splitting_depth)
    ul, uu = bounds(s.poly_interval.Up, solver.splitting_depth)

    printing && println("lbs = ", ll)
    printing && println("ubs = ", uu)

    # for now, minimize range between all outputs
    loss = sum(uu - ll)
    return loss
end


function propagate(solver::DiffNNPolySym, net::NV.NetworkNegPosIdx, input::AbstractHyperrectangle, α; printing=false)
    s = DiffPolyInterval(net, input)
    return propagate(solver, net, s, α, printing=printing)
end


## optimisation

function optimise_bounds(solver::DiffNNPolySym, net::NV.NetworkNegPosIdx, input_set; clip_norm=10., opt=nothing,
                         print_freq=50, n_steps=100, patience=50, timeout=60.)
    opt = isnothing(opt) ? OptimiserChain(ClipNorm(clip_norm), Adam()) : opt
    s = DiffPolyInterval(net, input_set)

    α0 = initialize_params(solver, net, 2, s)

    optfun = α -> propagate(solver, net, s, α)

    # TODO: maybe check num_crossing after optimisation, especially if gradient was 0
    # TODO: maybe store intermediate bounds also for output layer and return min/max of that

    α, y_hist, g_hist, d_hist, csims = optimise(optfun, opt, α0, print_freq=print_freq, n_steps=n_steps,
                                                patience=patience, timeout=timeout)
end
