

"""
Use interval overapproximation to get lower and upper relaxation of truncated polynomial.

Overapproximates the monomial at index idx of the exponent matrix
by an interval and uses it to generate lower and upper relaxations.
The resulting relaxations are parallel (just as for TaylorModels)!

args:
    sp - (SparsePolynomial) the polynomial to truncate
    idx - (Integer) the index of the mononial to overapproximate
"""
function truncate_interval(sp::SparsePolynomial, idx)
    ei = sp.E[:,idx]
    if all(iseven.(ei))
        # all even monomials are always positive
        # get min/max of 0*G or 1*G
        l_relax = min.(0, sp.G[:,idx])
        u_relax = max.(0, sp.G[:,idx])
    else
        # get min/max of -1*G or 1*G
        l_relax = min.(-sp.G[:,idx], sp.G[:,idx])
        u_relax = max.(-sp.G[:,idx], sp.G[:,idx])
    end

    all_idxs = 1:size(sp.G, 2)
    remaining_idxs = setdiff(all_idxs, idx)

    spₗ = SparsePolynomial(sp.G[:, remaining_idxs], sp.E[:, remaining_idxs], sp.ids)
    spₗ = translate(spₗ, l_relax)
    spᵤ = SparsePolynomial(sp.G[:, remaining_idxs], sp.E[:, remaining_idxs], sp.ids)
    spᵤ = translate(spᵤ, u_relax)

    return spₗ, spᵤ
end


"""
Use symbolic overapproximation via chebyshev relaxations to get lower and upper
relaxation of truncated polynomial.

Overapproximates the monomial at index idx of the exponent matrix
symbolically and uses it to generate lower and upper relaxations.
The resulting relaxations may be non-parallel!

args:
    sp - (SparsePolynomial) the polynomial to truncate
    idx - (Integer) the index of the mononial to overapproximate
"""
function truncate_symbolic(sp::SparsePolynomial, idx)
    # truncate generator with idx symbolically
    l_relax, u_relax = relax_monomial(sp.E[:, idx], sp.ids)

    all_idxs = 1:size(sp.G, 2)
    remaining_idxs = setdiff(all_idxs, idx)

    # for lower bound, multiply positive coeffs with l_relax, negative coeffs with u_relax
    Ĝ = max.(0, sp.G[:, idx]) .* l_relax.G .+ min.(0, sp.G[:, idx]) .* u_relax.G
    Gₗ = [sp.G[:, remaining_idxs] Ĝ]
    Eₗ = [sp.E[:, remaining_idxs] l_relax.E]

    # for upper bound, multiply positive coeffs with u_relax, negative coeffs with l_relax
    Ĝ = max.(0, sp.G[:, idx]) .* u_relax.G .+ min.(0, sp.G[:, idx]) .* l_relax.G
    Gᵤ = [sp.G[:, remaining_idxs] Ĝ]
    Eᵤ = [sp.E[:, remaining_idxs] u_relax.E]

    spₗ = compact(SparsePolynomial(Gₗ, Eₗ, sp.ids))
    spᵤ = compact(SparsePolynomial(Gᵤ, Eᵤ, sp.ids))
    return spₗ, spᵤ
end


"""
Iteratively truncate the smallest generator symbolically until only a desired
number of generators is left.

The process will terminate, as symbolic truncation always reduces the degree of
the monomials involved and if only a constant monomial is left, we can't truncate
any further.
TODO: efficiency of this approach might be limited.

args:
    sp - (SparsePolynomial) the polynomial to truncate
    n - (Integer) the number of desired generators

kwargs:
    lower - (bool) if we want the lower relaxation to be calculated (default: true)
    upper - (bool) if we want the upper relaxation to be calculated (default: true)
"""
function truncate_symbolic_desired(sp::SparsePolynomial, n::Integer; lower=true, upper=true)
    # lower - keep searching for lower bound, upper - keep searching for upper bound
    # n is number of desired generators
    if size(sp.G, 2) <= n
        # we don't need to truncate, if we already have less than n generators
        return sp, sp
    end

    l2s = vec(sum(sp.G .^ 2, dims=1))
    idx = argmin(l2s)

    if sum(sp.E[:,idx]) == 0
        # constant entry is smallest -> but can't truncate constants
        p = sortperm(l2s)
        idx = p[2] # take second-smallest generator
    end

    spₗ, spᵤ = truncate_symbolic(sp, idx)

    # @show size(spₗ.G, 2)
    if size(spₗ.G, 2) > n && lower
        # now only interested in lower bound
        spₗ, sp_lu = truncate_symbolic_desired(spₗ, n; upper=false)
    end

    if size(spᵤ.G, 2) > n && upper
        # only interested in upper bound
        sp_ul, spᵤ = truncate_symbolic_desired(spᵤ, n; lower=false)
    end

    return spₗ, spᵤ
end
