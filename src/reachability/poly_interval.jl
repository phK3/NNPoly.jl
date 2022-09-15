
# symbolic interval with polynomial functions as lower and upper relaxation

struct PolyInterval
    Low::SparsePolynomial
    Up::SparsePolynomial
end


function init_poly_interval(h::Hyperrectangle)
    Low = SparsePolynomial(h)
    Up = SparsePolynomial(h)
    return PolyInterval(Low, Up)
end


"""
Interval map overapproximating an affine map Wx + b for x ∈ [L, U].

W⁻ - (matrix) negative weights
W⁺ - (matrix) positive weights
L  - (SparsePolynomial) lower relaxation
U  - (SparsePolynomial) upper relaxation
b  - (vector) bias
"""
function interval_map(W⁻, W⁺, L, U, b)
    Low = exact_addition(linear_map(W⁻, U), linear_map(W⁺, L))
    Up  = exact_addition(linear_map(W⁻, L), linear_map(W⁺, U))
    Low = translate(Low, b)
    Up  = translate(Up, b)
    return Low, Up
end


"""
Calculates concrete lower and upper bounds on the PolyInterval iteratively
splitting the largest generator up to a specified splitting depth.
"""
function bounds(pint::PolyInterval, splitting_depth::Integer)
    ll, lu = bounds(pint.Low, splitting_depth)
    ul, uu = bounds(pint.Up, splitting_depth)
    return ll, uu
end


"""
Truncate the n_gens smallest generators in the sparse polynomial by overapproximating
them with intervals.
"""
function truncate_generators(sp::SparsePolynomial, n_gens::Integer)
    n_gens <= 0 && return sp, sp

    n, m = size(sp.G)
    l2s = vec(sum(sp.G .^2, dims=1))
    idxs = sortperm(l2s)
    all_idxs = 1:m

    truncated = 0
    truncated_idxs = zeros(n_gens)
    ub = zeros(n)
    lb = zeros(n)

    i = 0
    while truncated < n_gens
        i += 1
        idx = idxs[i]
        ei = sp.E[:,idx]

        if sum(ei) == 0
            # can't truncate constants
            continue
        elseif all(iseven.(ei))
            # if negative generator, we can add 0, as even exponent can never be negative
            ub .+= max.(0, sp.G[:,idx])
            # if positive generator, we can add 0
            lb .+= min.(0, sp.G[:,idx])
        else
            ub .+= abs.(sp.G[:,idx])
            lb .-= abs.(sp.G[:,idx])
        end

        truncated += 1
        truncated_idxs[truncated] = idx
    end

    #@show maximum(ub)
    #@show minimum(lb)

    remaining_idxs = setdiff(all_idxs, truncated_idxs)

    G = sp.G[:, remaining_idxs]
    E = sp.E[:, remaining_idxs]
    s = SparsePolynomial(G, E, sp.ids)
    return translate(s, lb), translate(s, ub)
end


"""
Truncate the PolyInterval, s.t. only n_gens generators are left.
"""
function truncate_desired(pint::PolyInterval, n_gens::Integer)
    n, m = size(pint.Low.G)
    n_del = m - n_gens
    L_low, L_up = truncate_generators(pint.Low, n_del)

    n, m = size(pint.Up.G)
    n_del = m - n_gens
    U_low, U_up = truncate_generators(pint.Up, n_del)

    return PolyInterval(L_low, U_up)
end
