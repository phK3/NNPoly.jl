
# Sparse represenation of multivariate and multioutput polynomials with
# variables xᵢ ∈ [-1, 1].
# Implemented after the Sparse Polynomial Zonotope paper:
#   Niklas Kochdumper, Matthias Althoff, Sparse Polynomial Zonotopes: A Novel
#       Set Representation for Reachability Analysis
#       https://mediatum.ub.tum.de/doc/1591469/ijh936tu65rc82gdlzx53oe5f.PolynomialZonotopes_Journal.pdf
#
# Changes mostly due to making the code differentiable by Zygote.

struct SparsePolynomial{N<:Number,M<:Integer,T<:Integer,GM<:AbstractArray{N},EM<:AbstractArray{M},VI<:AbstractArray{T}}
    G::GM  # generator matrix
    E::EM  # exponent matrix  (default is UInt16)
    ids::VI  # vector holding variable ids
end


function SparsePolynomial(h::Hyperrectangle{N}, exponent_dtype=UInt16, id_dtype=UInt16) where N <: Number
    n = dim(h)
    unfixed_mask = (h.radius .!= 0)
    G = [h.center I(n)[:, unfixed_mask] .* h.radius]
    E = [zeros(exponent_dtype, n) I(n)[:, unfixed_mask]]
    sp = SparsePolynomial(G, E, Vector{id_dtype}(1:n))
    return sp
end


zono_order(sp::SparsePolynomial) = size(sp.G, 2) / size(sp.G, 1)


## Utility functions


"""
Creates a SparsePolynomial with only the given monomial term.

args:
    exponents - (vector) exponents of the individual variables

kwargs:
    ids - (vector) ids of variables of monomial

"""
function make_monomial(exponents; ids=nothing)
    ids = isnothing(ids) ? [1] : ids
    G = vecOfVec2Mat([[1.]])
    E = vecOfVec2Mat([exponents])
    return SparsePolynomial(G, E, ids)
end


function getMonomialBasis(sp::SparsePolynomial)
    @polyvar x[sp.ids]
    return prod(x.^(sp.E), dims=1)
end


"""
return monomial coefficients for univariate one-dimensional polynomials
order is constant, linear, quadratic, ...
"""
function get_monomial_coefficients(sp::SparsePolynomial)
    # sort exponents and return corresponding coefficients
    @assert length(sp.ids) == 1 "only univariate polynomials are supported!\nsp=$sp"
    @assert size(sp.G, 1) == 1 "only one-dimensional polynomials are supported\nsp=$sp!"
    # want all monomials until highest order and also the constant term
    G = zeros(1, maximum(sp.E) + 1)
    p = sortperm(collect(eachcol(sp.E)))
    Ĝ = sp.G[:, p]
    G[:, vec(sp.E[:, p]) .+ 1] .= Ĝ
    return G[1,:]
end


"""
return all monomial coefficients for univariate one-dimensional polynomials up to
order n_order.

Order is constant, linear, quadratic, ...
"""
function get_monomial_coefficients(sp::SparsePolynomial, n_order::Integer)
    # sort exponents and return corresponding coefficients
    @assert length(sp.ids) == 1 "only univariate polynomials are supported!\nsp=$sp"
    @assert size(sp.G, 1) == 1 "only one-dimensional polynomials are supported!\nsp=$sp"
    # want all monomials until n_order and also the constant term
    G = zeros(1, n_order + 1)

    for (i, e) in enumerate(sp.E)
        # need e+1, since the constant term is at the first index, not zero
        if e <= n_order
            G[:, e+1] .= sp.G[:, i]
        end
    end

    return G[1,:]
end


function sparsePoly2DynamicPoly(sp::SparsePolynomial)
    basis = getMonomialBasis(sp)
    return sp.G * basis'
end


"""
Removes duplicate monomial entries by summing up monomial coefficients for
    monomials with same exponents.
"""
function compact(sp::SparsePolynomial)
    unique_idxs, duplicate_idxs = compact_idxs(sp)
    return compact(sp, unique_idxs, duplicate_idxs, remove_zeros=false)
end


"""
Calculate indices of coefficients of unique monomials and vector mapping duplicate monomials to 
an occurence of the same monomial in a compacted version of the polyonmial.

args:
    sp - the SparsePolynomial to be compacted

returns:
    unique_idxs - vector s.t. sp.E[:,unique_idxs] contains every monomial occuring in sp exactly once
    duplicate_idxs - vector s.t. duplicate_idxs[i] = duplicate_idxs[j] = k, iff. the i-th and k-th coefficients
                     are for the same monomial and this monomial has index k in the compacted representation. 
"""
function compact_idxs(sp::SparsePolynomial)
    d = @ignore_derivatives duplicates(eachcol(sp.E))
    unique_idxs = @ignore_derivatives first.(values(d))

    duplicate_idxs = @ignore_derivatives begin
        idxs = zeros(Int, size(sp.E, 2))
        for (i, v_list) in enumerate(collect(values(d)))
            for v in v_list
                idxs[v] = i
            end
        end
        idxs
    end

    return unique_idxs, duplicate_idxs
end


"""
Removes duplicate monomial coefficients by summing up coefficients for monomials with the same exponents.

If indices mapping duplicate coefficients to a single coefficient are already computed, these coefficients
can be directly passed.

args:
    sp - SparsePolynomial to be compacted
    unique_idxs - vector of indices, s.t. sp.E[:,unique_idxs] contains each monomial coefficient exactly once
    duplicate_idxs - vector of indices mapping each monomial in sp to a position in the compact representation of sp.E[:,unique_idxs]
            i.e. if idxs[i] = j, then monomial i will be mapped to position j in the resulting polynomial.
            (it will have idxs[i] = idxs[k] = j, if i,k are duplicates)

kwargs:
    remove_zeros - whether to remove monomials with all-zero coefficients
"""
function compact(sp::SparsePolynomial, unique_idxs, duplicate_idxs; remove_zeros=false)
    Ê = sp.E[:, unique_idxs]
    Ĝ = Flux.NNlib.scatter(+, sp.G, duplicate_idxs)

    if remove_zeros
        non_zeros = vec(sum(abs.(Ĝ), dims=1) .!= 0)
        Ê = Ê[:, non_zeros]
        Ĝ = Ĝ[:, non_zeros]

        if size(Ĝ, 2) == 0
            # all terms were zero
            # -> create constant zero SparsePolynomial
            n_dim = size(Ĝ, 1)
            Ĝ = zeros(n_dim, 1)
            Ê = zeros(M, length(sp.ids), 1)
        end
    end

    return SparsePolynomial(Ĝ, Ê, sp.ids)
end


"""
Adds ids of new variables, that have no terms associated with them, to a polynomial.

args:
    sp - (SparsePolynomial) the polynomial
    new_ids - (Vector{Integer}) the ids of the new variables
"""
function expand_ids(sp::SparsePolynomial, new_ids)
    n_ids, n_gens = size(sp.E)

    # keep ids sorted
    ids = sort(sp.ids ∪ new_ids)
    Ê = zeros(M, length(ids), n_gens)

    for (i, id) in enumerate(ids)
        if id in sp.ids
            Ê[i, :] .= vec(sp.E[sp.ids .== id, :])
        end
        # other ids were not in sp and are thus zero
    end

    return SparsePolynomial(sp.G, Ê, ids)
end


"""
Returns the starting point (not necessarily the center) of the polynomial.
"""
function get_center(sp::SparsePolynomial)
    # TODO: if monomials are sorted and compacted, could do this in constant
    # time instead of iterating through the monomials
    n, m = size(sp.G)
    c = zeros(n)
    for (j, ej) in enumerate(eachcol(sp.E))
        if sum(ej) == 0
            c += sp.G[:,j]
        end
    end

    return c
end


"""
Returns -f(x) for input f(x)
"""
function negate(sp::SparsePolynomial)
    return SparsePolynomial(.-sp.G, sp.E, sp.ids)
end


## Polynomial Arithmetic


"""
Evaluate the polynomial at x.
"""
function evaluate(sp::SparsePolynomial, x)
    return sp.G * prod(x.^sp.E, dims=1)'
end


"""
Component-wise addition of two polynomials.
Both polynomials have to depend on the same variables
(i.e. the variable identifiers have to be equal)
"""
function exact_addition(sp1::SparsePolynomial, sp2::SparsePolynomial)
    @assert sp1.ids == sp2.ids
    p̂ = SparsePolynomial([sp1.G sp2.G], [sp1.E sp2.E], sp1.ids)
    # compact summarizes terms with common exponent (can this be made more efficient?)
    return compact(p̂)
end


"""
Multiply each dimension with the same scalar or multiply each dimension with a
specific scalar for that dimension.
"""
function multiply(a::Union{N, VN}, sp::SparsePolynomial) where {N <: Number, VN<:Vector{N}}
    # @assert length(a) == 1 || length(a) == size(sp.G, 1) "|a| = $(length(a)), but size(G) = $(size(sp.G))"
    Ĝ = a .* sp.G
    return SparsePolynomial(Ĝ, sp.E, sp.ids)
end


"""
Component-wise substraction sp1 - sp2 of two polynomials
Both polynomials have to depend on the same variables
(i.e. the variable identifiers have to be equal)
"""
function subtract(sp1::SparsePolynomial, sp2::SparsePolynomial)
    return exact_addition(sp1, negate(sp2))
end


"""
Shifts the starting point of the polynomial by a vector v.
"""
function translate(sp::SparsePolynomial{N,M,T,GM,EM,VI}, v::AbstractVector) where {N,M,T,GM,EM,VI}
    const_idx = @ignore_derivatives findfirst(x -> sum(x) == 0, eachcol(sp.E))

    if isnothing(const_idx)
        Ĝ = [v sp.G]
        Ê = [zeros(M, size(sp.E, 1)) sp.E]
    else
        other_idxs = @ignore_derivatives 1:size(sp.G, 2) .!= const_idx
        Ĝ = [sp.G[:,const_idx] .+ v sp.G[:,other_idxs]]
        Ê = @ignore_derivatives [sp.E[:,const_idx] sp.E[:,other_idxs]]
    end

    return SparsePolynomial(Ĝ, Ê, sp.ids)
end


function linear_map(A, sp::SparsePolynomial)
    return SparsePolynomial(A*sp.G, sp.E, sp.ids)
end


function affine_map(A, sp::SparsePolynomial, b)
    # TODO: maybe store center as its own field, or just keep generators sorted?
    p_lin = linear_map(A, sp)
    return translate(p_lin, b)
end


"""
Computes one quadratic map (depending on all of the input dimensions) for each output dimension.

quadratic_map(Qs, S) = {x | xᵢ = s'Qᵢs, s ∈ S}
one quadratic map Qᵢ (n × n) for every output dimension
"""
function quadratic_map(Qs, sp::SparsePolynomial)
    n, m = size(sp.G)
    k = length(Qs)

    Ĝ = reduce(hcat, [vec(sp.G'*Qᵢ'*sp.G) for Qᵢ in Qs])'
    Ê = @ignore_derivatives repeat(sp.E, 1, m) .+ repeat(sp.E, inner=(1,m))

    return compact(SparsePolynomial(Ĝ, Ê, sp.ids))
end


"""
Computes a one-dimensional quadratic map for each input dimension.
I.e. yᵢ =  qᵢxᵢ² for each input dimension i
"""
function quadratic_map_1d(qs, sp::SparsePolynomial)
    n, m = size(sp.G)
    k = length(qs)

    Ĝ = qs .* repeat(sp.G, inner=(1, m)) .* repeat(sp.G, 1, m)
    # multiplying with 1 vector is just repeating
    # Ê = [E₁ E₂ ... Eₘ], Eⱼ = E + E[:,j] * ones(Integer, m)'
    Ê = @ignore_derivatives repeat(sp.E, inner=(1, m)) .+ repeat(sp.E, 1, m)

    return compact(SparsePolynomial(Ĝ, Ê, sp.ids))
end


"""
Computes a one-dimensional quadratic function for each input dimension.
I.e. yᵢ = aᵢxᵢ² + bᵢxᵢ + cᵢ for each input dimension i
"""
function quadratic_propagation(a, b, c, sp::SparsePolynomial)
    n, m = size(sp.G)

    p_quad = quadratic_map_1d(a, sp)
    G_lin = b .* sp.G
    p_affine = SparsePolynomial([c G_lin], [zeros(M, length(sp.ids)) sp.E], sp.ids)
    # TODO: how much computation can we save by knowing that both came from the same polynomial?
    return exact_addition(p_quad, p_affine)
end


"""
Computes the convex hull of two polynomials.
Careful: Dependencies between two polynomials are discarded to compute the convex hull!
"""
function convex_hull(sp1::SparsePolynomial, sp2::SparsePolynomial)
    # TODO: can we find a way to include the dependencies? Just use exact addition?
    G = 0.5 .* [sp1.G sp1.G sp2.G -sp2.G]

    n1, m1 = size(sp1.E)
    n2, m2 = size(sp2.E)
    z12 = zeros(M, n1, m2)
    z21 = zeros(M, n2, m1)
    z_11 = zeros(M, 1, m1)
    z_12 = zeros(M, 1, m2)
    o_11 = ones(M, 1, m1)
    o_12 = ones(M, 1, m2)
    E = [sp1.E sp1.E z12   z12;
         z21   z21   sp2.E sp2.E;
         z_11  o_11  z_12  o_12]

    ids = vec(1:(n1 + n2 + 1))
    return SparsePolynomial(G, E, ids)
end


## Splitting


"""
Substitute every occurrence of ϵᵢⁿ by (αδ + β)ⁿ
The new variable δ will take the variable id of ϵᵢ
"""
function substitute_binomial(sp::SparsePolynomial, eps_i, α, β)
    eps_i = findall(sp.ids .== eps_i)[1]
    e_max = maximum(sp.E[eps_i,:])

    G = copy(sp.G)
    E = copy(sp.E)

    # replace every occurence of ϵⁿ by (αδ + β)ⁿ
    for e_cur in 1:e_max
        idxs = findall(sp.E[eps_i,:] .== e_cur)
        for i in idxs
            Ĝₑ = α^e_cur * [binomial(e_cur, k) * (β/α)^(e_cur - k) for k in 0:e_cur]' .* sp.G[:,i]
            Êₑ = repeat(sp.E[:,i], 1, e_cur+1)
            Êₑ[eps_i,:] .= 0:e_cur

            G[:,i] .= Ĝₑ[:,1]
            E[:,i] .= Êₑ[:,1]
            G = [G Ĝₑ[:,2:end]]
            E = [E Êₑ[:,2:end]]
        end
    end

    return compact(SparsePolynomial(G, E, sp.ids))
end


"""
Normalize variable ϵᵢ ∈ [l, u] to [-1, 1]
"""
function normalize_variable(sp::SparsePolynomial, eps_i, l, u)
    α = 0.5 * (u - l)
    β = 0.5 * (u + l)
    return substitute_binomial(sp, eps_i, α, β)
end


"""
Rescale normalized variable ϵᵢ ∈ [-1, 1] to [l, u]
"""
# rescale normalized variable to interval [l, u]
function rescale_variable(sp::SparsePolynomial, eps_i, l, u)
    α = 1 / (0.5*(u - l))
    β = -(u + l) / (u - l)
    return substitute_binomial(sp, eps_i, α, β)
end


"""
Bisects the range of a given variable in a polynomial.
Since the new variables' range is no longer normalized to [-1, 1], the polynomial
needs to be renormalized.
"""
function split_error_term(sp::SparsePolynomial, eps_i)
    # we need the row-index corresponding to the identifier
    eps_i = findall(sp.ids .== eps_i)[1]  # since ids are unique there can only be one entry for the identifier
    e_max = maximum(sp.E[eps_i,:])

    G₁ = copy(sp.G)
    G₂ = copy(sp.G)
    E₁ = copy(sp.E)
    E₂ = copy(sp.E)

    for e_cur in 1:e_max
        idxs = findall(sp.E[eps_i,:] .== e_cur)

        # TODO: use normalize_variable here!
        for i in idxs
            # replace ϵⁿ by (1/2 + 1/2 ϵ₁)ⁿ for ϵ ∈ [0, 1] and normalized ϵ₁
            Ĝₑ = 1/2^e_cur * [binomial(e_cur, k) for k in 0:e_cur]' .* sp.G[:,i]
            Êₑ = repeat(sp.E[:,i], 1, e_cur+1)  # we get e_cur+1 new terms, one for each exonent 0,...,e_cur
            Êₑ[eps_i,:] .= 0:e_cur

            # reuse current entry, s.t. we can append the remainder to the end, without having to delete the old entry
            G₁[:,i] .= Ĝₑ[:,1]
            E₁[:,i] .= Êₑ[:,1]
            G₁ = [G₁ Ĝₑ[:,2:end]]
            E₁ = [E₁ Êₑ[:,2:end]]

            # replace ϵⁿ by (-1/2 + 1/2 ϵ₂)ⁿ for ϵ ∈ [-1, 0] and normalized ϵ₂
            Ĝₑ = 1/2^e_cur * [iseven(e_cur - k) ? binomial(e_cur, k) : -binomial(e_cur, k) for k in 0:e_cur]' .* sp.G[:,i]

            # reuse current entry, s.t. we can append the remainder to the end, without having to delete the old entry
            G₂[:,i] .= Ĝₑ[:,1]
            E₂[:,i] .= Êₑ[:,1]
            G₂ = [G₂ Ĝₑ[:,2:end]]
            E₂ = [E₂ Êₑ[:,2:end]]
        end
    end

    # summarize duplicate terms
    return compact(SparsePolynomial(G₁, E₁, sp.ids)), compact(SparsePolynomial(G₂, E₂, sp.ids))
end


"""
Splits the variable with the highest exponent in the generator with the largest L₂ norm
"""
function split_longest_generator(sp::SparsePolynomial)
    # e.g. x²y⁴ has order 2+4=6
    e_order = sum(sp.E, dims=1)
    # constant terms can't be split
    non_const_mask = (e_order .!= 0)'

    if sum(non_const_mask) == 0
        # TODO: if used in BaB, maybe pull out, s.t. no unnecessary splitting
        # only constant terms -> can't split anything
        return sp
    end

    Ĝ = sp.G[:, non_const_mask]
    Ê = sp.E[:, non_const_mask]
    # choose longest generator in ||.||₂ sense
    g_idx = argmax(vec(sum(Ĝ.^2, dims=1)))  # need vec as sum returns matrix
    e_idx = argmax(Ê[:, g_idx]) # split ϵ with largest exponent in longest generator
    ϵᵢ = sp.ids[e_idx]

    return split_error_term(sp, ϵᵢ)
end


"""
Iteratively splits the variable with the highest exponent in the generator with the
largest L₂ norm until a certain splitting depth is reached.
"""
function split_longest_generator_iterative(sp::SparsePolynomial, splitting_depth)
    polys = [sp]

    for d in 1:splitting_depth
        polys2 = []
        for poly in polys
            poly1, poly2 = split_longest_generator(poly)
            push!(polys2, poly1)
            push!(polys2, poly2)
        end
        polys = polys2
    end

    return polys
end


## Computing bounds


"""
Computes lower bound for monomial x₁^e₁ ⋅ x₂^e₂ ...
without a scalar factor.
Values of the variables are assumed to be within [-1, 1]
"""
function monomial_lb(e)
    # if exponents for all variables are 0, we have a constant
    if sum(e) == 0
        lb = 1.
    # if all exponents are even, result will be ≧ 0
    elseif all(iseven.(e))
        lb = 0.
    # otherwise lower bound will be -1
    else
        lb = -1.
    end

    return lb
end


"""
Computes vector of lower bounds for Matrix of exponents.
"""
function monomial_lbs(E::M) where M<:AbstractMatrix{<:Integer}
    one_odd = .~reduce(&, iseven.(E), dims=1)
    # sum is only zero for a column if all rows are zero
    consts  = sum(E, dims=1) .== 0

    # all_even cols have lb=0 (so we don't need to add anything for them)
    # const columns have lb=1 (so we need to add 1 for them)
    # all other cols have lb=-1 (so we subtract 1 for them)
    lbs = .- one_odd .+ consts
    return vec(lbs)
end


"""
Computes component-wise bounds on monomials xᵢⁿxⱼᵐxₖᵒ... of a sparse polynomial
without a scalar factor.
Values of the variables are assumed to be within [-1, 1].
"""
function exponent_bounds(sp::SparsePolynomial)
    n, m = size(sp.E)
    #lbs = [monomial_lb(ej) for ej in eachcol(sp.E)]
    lbs = @ignore_derivatives monomial_lbs(sp.E)
    ubs = ones(m)  # ub is just always 1 under our assumptions

    return lbs, ubs
end


"""
Computes interval bounds for each component of a sparse polynomial.
Variables are assumed to be in range [-1, 1].
"""
function bounds(sp::SparsePolynomial{N,M,T,GM,EM,VI}) where {N,M,T,GM,EM,VI}
    lbs, ubs = exponent_bounds(sp)

    G⁻ = min.(zero(N), sp.G)
    G⁺ = max.(zero(N), sp.G)
    lb = G⁻ * ubs .+ G⁺ * lbs
    ub = G⁻ * lbs .+ G⁺ * ubs

    return lb, ub
end


"""
Computes interval bounds for each component of a sparse polynomial by splitting
the polynomial iteratively along the longest generator up to a certain depth.

Splitting is done once to the specified depth, then bounds for all components
are calculated.
"""
function bounds(sp::SparsePolynomial, splitting_depth::Integer)
    n = size(sp.G, 1)  # dimensions
    polys = split_longest_generator_iterative(sp, splitting_depth)

    lbs = fill(Inf, n)
    ubs = fill(-Inf, n)
    for p in polys
        l, u = bounds(p)
        lbs = min.(lbs, l)
        ubs = max.(ubs, u)
    end

    return lbs, ubs
end


"""
Computes an interval bound for the maximum of a sparse polynomial in direction d.
Variables are assumed to be in range [-1, 1]
"""
function max_in_dir(d, sp::SparsePolynomial)
    p = linear_map(d', sp)
    lb, ub = bounds(p)
    return ub
end


"""
Computes an interval bound for the maximum of a sparse polynomial in direction d
after iteratively splitting the largest generator up to a certain splitting depth.
"""
function max_in_dir(d, sp::SparsePolynomial, splitting_depth)
    # TODO: make BaB instead of blindly splitting everything
    p = linear_map(d', sp)
    polys = split_longest_generator_iterative(p, splitting_depth)
    lb = Inf
    ub = -Inf
    for poly in polys
        lbᵢ, ubᵢ = bounds(poly)
        lb = min(lb, lbᵢ[1])  # TODO: why does bounds return vectors?
        ub = max(ub, ubᵢ[1])
    end

    return ub
end


## Zonotope Overapproximation


"""
Overapproximates a sparse polynomial by a zonotope with the same number of generators.

Return value always has datatype Float.
"""
function zono_overapprox(sp::SparsePolynomial)
    all_idxs = 1:size(sp.E, 2)
    constant_idxs = []
    even_idxs = []
    for (j, ej) in enumerate(eachcol(sp.E))
        if sum(ej) == 0
            push!(constant_idxs, j)
        elseif all(iseven.(ej))
            push!(even_idxs, j)
        end
    end

    other_idxs = setdiff(all_idxs, constant_idxs)
    other_idxs = setdiff(other_idxs, even_idxs)
    c = sum(sp.G[:,constant_idxs], dims=2) .+ 0.5 .* sum(sp.G[:, even_idxs], dims=2)
    G = [0.5 .* sp.G[:, even_idxs] sp.G[:, other_idxs]]

    # TODO: change to something more generic than always Float
    # need vec(c) as it is of Matrix type
    # return vec(c), G
    # return Zonotope(vec(c), G)  but only if datatypes are equal, throws error with Float, Integer
    return Zonotope(Float64.(vec(c)), Float64.(G))
end


## Truncation


"""
Remove n generators and overapproximate them by an order-1 zonotope (basically
an interval for each dimension, represented as error-terms).
"""
function truncation_zono_reduction(sp::SparsePolynomial, n::Integer)
    rows, cols = size(sp.G)
    if rows >= n
        # we need one error term per dimension, so it doesn't help us, if we reduce less than rows
        return sp
    end

    l2s = vec(sum(sp.G .^ 2, dims=1))
    idxs = sortperm(l2s)[1:n]  # get n smallest terms
    all_idxs = 1:size(sp.G, 2)
    remaining_idxs = setdiff(all_idxs, idxs)

    G = sp.G[:, idxs]
    E = sp.E[:, idxs]
    poly = SparsePolynomial(G, E, sp.ids)
    z = zono_overapprox(poly)
    z = reduce_order(z, 1)
    Gz = z.generators

    G = sp.G[:, remaining_idxs]
    E = sp.E[:, remaining_idxs]
    en, em = size(E)

    G = [G Gz]
    E = [E zeros(en, rows); zeros(rows, em) I(rows)]
    max_id = maximum(sp.ids)
    ids = [sp.ids; max_id+1:max_id+rows]
    poly = SparsePolynomial(G, E, ids)
    return translate(poly, z.center)
end


"""
Truncate generators, s.t. a specified zonotope order is reached.
"""
function truncate_order(sp::SparsePolynomial, order::Integer)
    n, m = size(sp.G)
    desired_generators = n * order
    diff_generators = m - desired_generators + n  # +n as we need to introduce new error terms for each dimension

    if desired_generators >= m
        return sp
    end
    return truncation_zono_reduction(sp, diff_generators)
end


"""
Truncate generators, s.t. a specified number of generators is reached.
"""
function truncate_desired(sp::SparsePolynomial, desired_generators::Integer)
    n, m = size(sp.G)
    diff_generators = m - desired_generators + n

    if desired_generators >= m
        return sp
    end
    return truncation_zono_reduction(sp, diff_generators)
end


## Stacking two polynomials on top of each other

"""
Creates a SparsePolynomial that represents a vertical stacking of the given polynomials.
(the dimensions of the first polynomial are the first dimensions of the output polynomial,
the dimensions of the second polynomial are the second dimensions of the output polynomial, ...)

args:
    polys - (Vector of SparsePolynomial) polynomials to be stacked
"""
function stack_polys(polys::AbstractVector{<:SparsePolynomial})
    # TODO: really slow implementation
    if length(polys) == 2
        p1 = polys[1]
        p2 = polys[2]
        n1, m1 = size(p1.G)
        n2, m2 = size(p2.G)
        G = [p1.G zeros(n1, m2); zeros(n2, m1) p2.G]

        # is this necessary?
        ids = sort(p1.ids ∪ p2.ids)
        E = zeros(M, length(ids), size(G, 2))

        for (i, id) in enumerate(ids)
            if id in p1.ids
                E[i, 1:m1] .= vec(p1.E[p1.ids .== id, :])
            end

            # ids can occur in both polynomials
            if id in p2.ids
                E[i, m1+1:end] .= vec(p2.E[p2.ids .== id, :])
            end
        end

        return compact(SparsePolynomial(G, E, ids))
    else
        # can we split in half instead of only appending one element at a time
        p1 = polys[1]
        p2 = stack_polys(polys[2:end])
        return stack_polys([p1, p2])
    end
end


## Plotting

"""
Plots a sparse polynomial by plotting zonotope-overapproximations refined by
iteratively splitting the largest generators up to a certain splitting depth.
"""
@recipe function plot_polynomial(sp::SparsePolynomial; splitting_depth=10)
    # TODO: only split polys that are on boundary of mp
    polys = split_longest_generator_iterative(sp, splitting_depth)
    zonos = zono_overapprox.(polys)

    c_series = get(plotattributes, :seriescolor, nothing)
    c_line = get(plotattributes, :linecolor, nothing)
    if !isnothing(c_line)
        linecolor --> c_line
    elseif !isnothing(c_series)
        linecolor --> c_series
    end

    seriestype --> :shape
    @series _plot_list(zonos)
end
