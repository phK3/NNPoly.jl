

"""
Creates symbolic relaxation of max(x, y) where x,y can be any SparsePolynomials.

Utilizes insight from OVAL that max(x, y) = max(0, y - x) + x s.t, we can use
linear ReLU relaxation.
Bounds on y - x needed for the relaxation are computed using a BaB procedure.

args:
    x - SparsePolynomial
    y - SparsePolynomial

kwargs:
    printing - (bool) whether to print obtained bounds on y - x
    max_steps - (Integer) maximum number of steps in BaB procedure
    optimality_gap - (Float64) optimality gap for early stopping of BaB procedure
"""
function relax_max_upper(x, y; printing=false, max_steps=30, optimality_gap=1e-4)
    # max(x,y) = max(0, y - x) + x

    # TODO: pull in front of all relaxations as matrix operation!!!
    p_in = subtract(y, x)

    ub =  max_in_dir_bab( [1], p_in, printing=false, max_steps=max_steps, optimality_gap=optimality_gap)
    lb = -max_in_dir_bab([-1], p_in, printing=false, max_steps=max_steps, optimality_gap=optimality_gap)
    printing && println("bounds: ", [lb, ub])

    if lb >= 0
        return y
    elseif ub <= 0
        return x
    else
        λ = ub / (ub - lb)
        β = -ub*lb / (ub - lb)
        return exact_addition(affine_map(λ, p_in, [β]), x)
    end
end


"""
Creates symbolic relaxation of min(x, y) where x,y can be any SparsePolynomials.

We have min(x, y) = -max(-x, -y), so we can use relax_max_upper() to obtain a
symbolic lower bound of min(x, y).

args:
    x - SparsePolynomial
    y - SparsePolynomial

kwargs:
    printing - (bool) whether to print obtained bounds on y - x
    max_steps - (Integer) maximum number of steps in BaB procedure
    optimality_gap - (Float64) optimality gap for early stopping of BaB procedure
"""
function relax_min_lower(x, y; printing=false, max_steps=30, optimality_gap=1e-4)
    return negate(relax_max_upper(negate(x), negate(y), printing=printing, max_steps=max_steps, optimality_gap=optimality_gap))
end


function relax_min_max_non_parallel(x, y; printing=false, max_steps=30, optimality_gap=1e-4)
    l_relax = relax_min_lower(x, y, printing=printing, max_steps=max_steps, optimality_gap=optimality_gap)
    u_relax = relax_max_upper(x, y, printing=printing, max_steps=max_steps, optimality_gap=optimality_gap)
    return l_relax, u_relax
end

"""
Creates parallel symbolic relaxation of [min(x,y), max(x, y)] where x,y can be any SparsePolynomials.

We first compute μ(x, y) = 1/2*(x + y) and calculate bounds for μ(x, y) - x
(which handily also translate into bounds for μ(x, y) - y) and shift μ(x,y) down
and up by the corresponding bounds to obtain a lower and an upper relaxation that
are parallel to each other.

args:
    x - SparsePolynomial
    y - SparsePolynomial

kwargs:
    printing - (bool) whether to print obtained bounds on y - x
    max_steps - (Integer) maximum number of steps in BaB procedure
    optimality_gap - (Float64) optimality gap for early stopping of BaB procedure
"""
function relax_min_max_parallel(x, y; printing=false, max_steps=30, optimality_gap=1e-4)
    p_xy = stack_polys([x, y])
    μ = linear_map([0.5 0.5], p_xy)
    p_diff = linear_map([-0.5 0.5], p_xy)

    u =  max_in_dir_bab( [1], p_diff, printing=false, max_steps=max_steps, optimality_gap=optimality_gap)
    l = -max_in_dir_bab([-1], p_diff, printing=false, max_steps=max_steps, optimality_gap=optimality_gap)
    printing && println("bounds: ", [l, u])

    l_relax = translate(μ, [min(l, -u)])
    u_relax = translate(μ, [max(-l, u)])

    return l_relax, u_relax
end


function relax_min_max(x, y; printing=false, max_steps=30, optimality_gap=1e-4, parallel=false)
    if parallel
        l_relax, u_relax = relax_min_max_parallel(x, y; printing=printing, max_steps=max_steps, optimality_gap=optimality_gap)
    else
        l_relax, u_relax = relax_min_max_non_parallel(x, y; printing=printing, max_steps=max_steps, optimality_gap=optimality_gap)
    end

    return l_relax, u_relax
end        
