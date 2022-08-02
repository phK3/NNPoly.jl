

@with_kw struct NNPolyZono <: NV.Solver
    truncation_order = 50
    approximation = :Chebyshev
    bounds_splits = 0
end


function NV.forward_linear(solver::NNPolyZono, L::NV.Layer, input::SparsePolynomial)
    return affine_map(L.weights, input, L.bias)
end


function NV.forward_act(solver::NNPolyZono, L::NV.Layer{NV.ReLU}, input::SparsePolynomial)
    n = size(input.G, 1)
    # for now only quadratic relaxation
    degrees = 2*ones(Integer, n)

    s = truncate_desired(input, solver.truncation_order)
    lb, ub = bounds(s, solver.bounds_splits)

    if solver.approximation == :Chebyshev
        res = relax_relu_chebyshev.(lb, ub, degrees)
        cs = vecOfVec2Mat(first.(res))
        ϵs = last.(res)
    else
        @assert false string("Chebyshev is the only valid approximation yet!")
    end

    ŝ = quadratic_propagation(cs[:,3], cs[:,2], cs[:,1], s)

    crossing_idxs = findall(ϵs .!= 0)
    n_crossing = length(crossing_idxs)
    n, m = size(ŝ.G)
    n_vars = length(ŝ.ids)
    G = [ŝ.G ϵs .* partial_I(n, crossing_idxs)]
    E = [ŝ.E                           zeros(Integer, n_vars, n_crossing);
         zeros(Integer, n_crossing, m) I                             ]
    ids = [ŝ.ids; maximum(ŝ.ids)+1:maximum(ŝ.ids)+n_crossing]
    return SparsePolynomial(G, E, ids)
end


function NV.forward_act(solver::NNPolyZono, L::NV.Layer{NV.Id}, input::SparsePolynomial)
    return input
end
