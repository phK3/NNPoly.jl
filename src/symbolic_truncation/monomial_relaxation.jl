

"""
Creates a SparsePolynomial that represents a vertical stacking of the given polynomials.
(the dimensions of the first polynomial are the first dimensions of the output polynomial,
the dimensions of the second polynomial are the second dimensions of the output polynomial, ...)

args:
    polys - (Vector of SparsePolynomial) polynomials to be stacked
"""
function stack_polys(polys)
    # TODO: really slow implementation
    if length(polys) == 2
        p1 = polys[1]
        p2 = polys[2]
        n1, m1 = size(p1.G)
        n2, m2 = size(p2.G)
        G = [p1.G zeros(n1, m2); zeros(n2, m1) p2.G]

        # is this necessary?
        ids = sort(p1.ids ∪ p2.ids)
        E = zeros(Integer, length(ids), size(G, 2))

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
        p1 = polys[1]
        p2 = stack_polys(polys[2:end])
        return stack_polys([p1, p2])
    end
end


"""
Creates symbolic relaxation of a multivariate monomial described by its exponents
and the associated variables.

Utilizes the fact that for any x, y
    xy ≥ min(xₗyₗ, xₗyᵤ, xᵤyₗ, xᵤyᵤ) and
    xy ≤ min(xₗyₗ, xₗyᵤ, xᵤyₗ, xᵤyᵤ)
and then recursively uses relaxations for min and max.

args:
    es - (Vector{Integer}) exponents of the variables associated with the ids
    ids - (Vector{Integer}) ids of the variables in the monomial
"""
function relax_monomial(es, ids)
    non_zeros = es .!= 0
    zero_ids = ids[.~non_zeros]
    es = es[non_zeros]
    ids = ids[non_zeros]

    if length(es) == 0
        # only constants
        l_min = make_monomial([0])
        u_max = make_monomial([0])
    elseif length(es) == 1
        l_min, u_max = relax_monomial_cheby(es[1], ids[1])
    else
        n = length(es)
        # TODO: this assumes consecutive variables ids !
        l1, u1 = relax_monomial(es[1:floor(Integer, 0.5*n)], ids[1:floor(Integer, 0.5*n)])
        l2, u2 = relax_monomial(es[floor(Integer, 0.5*n)+1:end], ids[floor(Integer, 0.5*n)+1:end])

        polys = stack_polys([l1, u1, l2, u2])

        #@show polys
        l1l2 = zeros(4, 4)
        l1l2[1, 3] = 0.5
        l1l2[3, 1] = 0.5
        b1 = quadratic_map([l1l2], polys)

        l1u2 = zeros(4, 4)
        l1u2[1, 4] = 0.5
        l1u2[4, 1] = 0.5
        b2 = quadratic_map([l1u2], polys)

        u1l2 = zeros(4, 4)
        u1l2[2, 3] = 0.5
        u1l2[3, 2] = 0.5
        b3 = quadratic_map([u1l2], polys)

        u1u2 = zeros(4, 4)
        u1u2[2, 4] = 0.5
        u1u2[4, 2] = 0.5
        b4 = quadratic_map([u1u2], polys)

        l_min1 = relax_min_lower(b1, b2, printing=false)
        l_min2 = relax_min_lower(b3, b4, printing=false)
        l_min  = relax_min_lower(l_min1, l_min2, printing=false)

        u_max1 = relax_max_upper(b1, b2, printing=false)
        u_max2 = relax_max_upper(b3, b4, printing=false)
        u_max  = relax_max_upper(u_max1, u_max2, printing=false)
    end

    l_min = expand_ids(l_min, zero_ids)
    u_max = expand_ids(u_max, zero_ids)
    return l_min, u_max
end
