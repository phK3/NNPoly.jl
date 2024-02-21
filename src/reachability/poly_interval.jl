
# symbolic interval with polynomial functions as lower and upper relaxation

struct PolyInterval{N<:Number,M<:Integer,T<:Integer}
    Low::SparsePolynomial{N,M,T}
    Up::SparsePolynomial{N,M,T}
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


function interval_map_common_lower(W⁻, W⁺, L::SparsePolynomial, U::SparsePolynomial, b)
    @assert L.E == U.E "Exponent matrices must be equal for common interval map!"
    Gₗ = W⁻ * U.G .+ W⁺ * L.G
    L̂ = SparsePolynomial(Gₗ, L.E, L.ids)
    return translate(L̂, b)
end


function interval_map_common_upper(W⁻, W⁺, L::SparsePolynomial, U::SparsePolynomial, b)
    @assert L.E == U.E "Exponent matrices must be equal for common interval map!"
    Gᵤ = W⁻ * L.G .+ W⁺ * U.G
    Û = SparsePolynomial(Gᵤ, U.E, U.ids)
    return translate(Û, b)
end


function interval_map_common(W⁻, W⁺, L::SparsePolynomial, U::SparsePolynomial, b)
    @assert L.E == U.E "Exponent matrices must be equal for common interval_map!"
    Gₗ = W⁻ * U.G .+ W⁺ * L.G
    Gᵤ = W⁻ * L.G .+ W⁺ * U.G
    L̂ = SparsePolynomial(Gₗ, L.E, L.ids)
    Û = SparsePolynomial(Gᵤ, U.E, U.ids)
    return translate(L̂, b), translate(Û, b)
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

    l2s = vec(sum(sp.G .^2, dims=1))
    idxs = sortperm(l2s)

    E = @view sp.E[:,idxs[1:n_gens]]
    if 0 in sum(E, dims=1)
        # the constant term is included, but its truncation has no effect
        # so truncate one more term
        n_gens += 1
    end

    G = sp.G[:, idxs[1:n_gens]]  # slightly faster with @view but only for large matrices?
    E = sp.E[:, idxs[1:n_gens]]
    lb, ub = bounds(SparsePolynomial(G, E, sp.ids))

    spt = SparsePolynomial(sp.G[:, idxs[n_gens+1:end]], sp.E[:, idxs[n_gens+1:end]], sp.ids)
    # TODO: most of the time is actually spent doing the translate operations!!!
    spl = translate(spt, lb)
    spu = translate(spt, ub)

    return spl, spu
end


"""
Truncate the **same** n_gens smallest generators across two sparse polynomial by overapproximating
them with intervals.
"""
function truncate_common(lp::SparsePolynomial, up::SparsePolynomial, n_gens::Integer)
    n_gens <= 0 && return lp, up

    @assert lp.E == up.E "Exponent matrices must be equal for common truncation!"

    l2s = vec(sum(lp.G .^2 .+ up.G .^2, dims=1))
    idxs = sortperm(l2s)

    E = @view lp.E[:,idxs[1:n_gens]]
    if 0 in sum(E, dims=1)
        n_gens += 1
    end

    E = lp.E[:, idxs[1:n_gens]]
    Gₗ = lp.G[:,idxs[1:n_gens]]
    Gᵤ = up.G[:,idxs[1:n_gens]]

    ll, lu = bounds(SparsePolynomial(Gₗ, E, lp.ids))
    ul, uu = bounds(SparsePolynomial(Gᵤ, E, up.ids))

    lp_trunc = SparsePolynomial(lp.G[:,idxs[n_gens+1:end]], lp.E[:,idxs[n_gens+1:end]], lp.ids)
    up_trunc = SparsePolynomial(up.G[:,idxs[n_gens+1:end]], up.E[:,idxs[n_gens+1:end]], up.ids)

    l̂p = translate(lp_trunc, ll)
    ûp = translate(up_trunc, uu)

    return l̂p, ûp
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


"""
Truncate the PolyInterval, s.t. only n_gens generators are left.
"""
function truncate_desired_common(pint::PolyInterval, n_gens::Integer)
    n, m = size(pint.Low.G)
    n_del = m - n_gens

    L, U = truncate_common(pint.Low, pint.Up, n_del)

    return PolyInterval(L, U)
end



# TODO: include in array interface?
function select_idxs(s::PolyInterval, idxs)
    L̂ = SparsePolynomial(s.Low.G[idxs, :], s.Low.E, s.Low.ids)
    Û = SparsePolynomial(s.Up.G[idxs, :], s.Up.E, s.Up.ids)
    return PolyInterval(L̂, Û)
end
