
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

kwargs:
    parallel - (bool) whether to use parallel relaxations for the monomials
"""
function relax_monomial(es, ids; parallel=false)
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

        #l_min1 = relax_min_lower(b1, b2, printing=false)
        #l_min2 = relax_min_lower(b3, b4, printing=false)
        #l_min  = relax_min_lower(l_min1, l_min2, printing=false)

        #u_max1 = relax_max_upper(b1, b2, printing=false)
        #u_max2 = relax_max_upper(b3, b4, printing=false)
        #u_max  = relax_max_upper(u_max1, u_max2, printing=false)

        l_min1, u_max1 = relax_min_max(b1, b2, printing=false, parallel=parallel)
        l_min2, u_max2 = relax_min_max(b3, b4, printing=false, parallel=parallel)
        if parallel
            # need to ensure that l_min, u_max are parallel!
            l_min, _ = relax_min_max(l_min1, l_min2, printing=false, parallel=parallel)
            _, u_max = relax_min_max(u_max1, u_max2, printing=false, parallel=parallel)
            l_min, u_max = relax_min_max(l_min, u_max, printing=false, parallel=parallel)
        else
            l_min  = relax_min_lower(l_min1, l_min2, printing=false)
            u_max  = relax_max_upper(u_max1, u_max2, printing=false)
        end

    end

    l_min = expand_ids(l_min, zero_ids)
    u_max = expand_ids(u_max, zero_ids)
    return l_min, u_max
end
