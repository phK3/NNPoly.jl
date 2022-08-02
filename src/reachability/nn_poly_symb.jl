
# NNPolySym propagates symbolic intervals with polynomial lower and upper bounds
# through the NN.


@with_kw struct NNPolySym <: NV.Solver
    truncation_terms = 50
    relaxations = :separate
    splitting_depth = 2
    bounding = :interval  # alternative is :bab
end


function NV.forward_linear(solver::NNPolySym, L::NV.Layer, input::PolyInterval)
    Low, Up = interval_map(min.(0, L.weights), max.(0, L.weights), input.Low, input.Up, L.bias)
    return PolyInterval(Low, Up)
end


function NV.forward_act(solver::NNPolySym, L::NV.Layer{NV.ReLU}, input::PolyInterval)
    n = size(input.Low.G, 1)
    degrees = 2*ones(Integer, n)

    s = truncate_desired(input, solver.truncation_terms)

    if solver.bounding == :interval
        ll, lu = bounds(s.Low, solver.splitting_depth)
        ul, uu = bounds(s.Up, solver.splitting_depth)
    elseif solver.bounding == :bab
        ll = zeros(n)
        lu = zeros(n)
        ul = zeros(n)
        uu = zeros(n)
        for i in 1:n
            dir = zeros(n)
            dir[i] = 1
            ll[i] = -max_in_dir_bab(-dir, s.Low, max_steps=solver.splitting_depth)
            lu[i] = max_in_dir_bab(dir, s.Low, max_steps=solver.splitting_depth)
            ul[i] = -max_in_dir_bab(-dir, s.Up, max_steps=solver.splitting_depth)
            uu[i] = max_in_dir_bab(dir, s.Up, max_steps=solver.splitting_depth)
        end
    end

    if solver.relaxations == :separate
        res = relax_relu_chebyshev.(ll, lu, degrees)
        cs = vecOfVec2Mat(first.(res))
        ϵs = last.(res)
        L̂ = quadratic_propagation(cs[:,3], cs[:,2], cs[:,1] .- ϵs, s.Low)

        res = relax_relu_chebyshev.(ul, uu, degrees)
        cs = vecOfVec2Mat(first.(res))
        ϵs = last.(res)
        Û = quadratic_propagation(cs[:,3], cs[:,2], cs[:,1] .+ ϵs, s.Up)
    else
        res = relax_relu_chebyshev.(ll, uu, degrees)
        cs = vecOfVec2Mat(first.(res))
        ϵs = last.(res)

        # cheby(x) - ϵ is the lower relaxation, cheby(x) + ϵ is the upper relaxation
        L̂ = quadratic_propagation(cs[:,3], cs[:,2], cs[:,1] .- ϵs, s.Low)
        Û = quadratic_propagation(cs[:,3], cs[:,2], cs[:,1] .+ ϵs, s.Up)
    end

    return PolyInterval(L̂, Û)
end


function NV.forward_act(solver::NNPolySym, L::NV.Layer{NV.Id}, input::PolyInterval)
    return input
end
