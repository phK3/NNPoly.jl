

# only raise matrix elements to power if exponent is non-zero
function non_zero_power(m::AbstractVecOrMat, e::AbstractVector)
    mask = e .!= 0
    return (@view m[:,mask]) .^ e[mask]'
end

function non_zero_mul(m::AbstractVecOrMat, e::AbstractVector)
    mask = e .!= 0
    return (@view m[:,mask]) .* e[mask]'
end


# accumulate product of non-zero powers inplace
function prod_non_zero_power!(prod_cum::AbstractVector, G::AbstractMatrix, e::AbstractVector{<:Integer})
    first_prod = true
    for (i, eᵢ) in enumerate(e)
        if eᵢ != 0
            if first_prod
                prod_cum .= (@view G[:,i]) .^ eᵢ
                first_prod = false
            else
                prod_cum .*= (@view G[:,i]) .^ eᵢ
            end
        end
    end
end


function sum_non_zero_mul!(sum_cum::AbstractVector, E::AbstractMatrix{<:Integer}, e::AbstractVector{<:Integer})
    first_sum = true
    for (i, eᵢ) in enumerate(e)
        if eᵢ != 0
            if first_sum
                sum_cum .= (@view E[:,i]) .* eᵢ
                first_sum = false
            else
                sum_cum .+= (@view E[:,i]) .* eᵢ
            end
        end
    end
end


# raise sparse polynomial to power n
function pow(sp::SparsePolynomial{N,M}, n::Integer) where {N,M}
    r, c = size(sp.G)
    n_terms = binomial(n + c - 1, c-1)

    Ĝ = zeros(r, n_terms)
    Ê = zeros(M, size(sp.E, 1), n_terms)

    prod_cum = zeros(r)
    sum_cum = zeros(size(sp.E, 1))
    for (i, e) in zip(1:n_terms, multiexponents(c, n))
        prod_non_zero_power!(prod_cum, sp.G, e)
        sum_non_zero_mul!(sum_cum, sp.E, e)
        prod_cum .*= multinomial(e...)
        Ĝ[:,i] .= prod_cum
        Ê[:,i] .= sum_cum
    end

    return SparsePolynomial(Ĝ, Ê, sp.ids)
end


"""
Raise SparsePolynomial to power n, but only calculated for dimensions with crossing ReLUs - other
dimensions are 0.
"""
function pow(sp::SparsePolynomial{N,M}, n::Integer, l, u) where {N,M}
    r, c = size(sp.G)
    n_terms = binomial(n + c - 1, c-1)

    # only need to calculate power for crossing ReLUs
    mask = (l .< 0) .& (u .> 0)
    Ĝ = zeros(r, n_terms)
    Ê = zeros(M, size(sp.E, 1), n_terms)

    prod_cum = zeros(sum(mask))
    sum_cum = zeros(size(sp.E, 1))
    for (i, e) in zip(1:n_terms, multiexponents(c, n))
        prod_non_zero_power!(prod_cum, @view(sp.G[mask,:]), e)
        sum_non_zero_mul!(sum_cum, sp.E, e)
        prod_cum .*= multinomial(e...)
        Ĝ[mask,i] .= prod_cum
        Ê[:,i] .= sum_cum
    end

    return SparsePolynomial(Ĝ, Ê, sp.ids)
end


function ChainRulesCore.rrule(::typeof(pow), sp::SparsePolynomial, n::Integer)

    ŝp = pow(sp, n)

    function pow_pullback(Δsp)
        ΔG = Δsp.G
        r, c = size(sp.G)
        Ĝ = zeros(r, c)

        prod_cum = zeros(r)
        sum_cum = zeros(size(sp.E, 1))
        for (i, e) in enumerate(multiexponents(c, n))
            for j in 1:size(sp.G, 2)
                if e[j] == 0
                    continue
                end

                α = e[j]
                e[j] -= 1
                prod!(prod_cum, non_zero_power(sp.G, e))
                #gⱼ = α .* prod(non_zero_power(sp.G, e), dims=2)
                e[j] += 1

                Ĝ[:,j] .+= multinomial(e...) .* (@view ΔG[:,i]) .* α .* prod_cum
            end
        end

        # return tangent types for **all** arguments of the rrule
        # so for ::typeof(power_n_loop), sp, n
        return NoTangent(), Tangent{SparsePolynomial}(G=Ĝ, E=NoTangent(), ids=NoTangent()), NoTangent()
    end

    return ŝp, pow_pullback
end


function ChainRulesCore.rrule(::typeof(pow), sp::SparsePolynomial, n::Integer, l, u)

    ŝp = pow(sp, n, l, u)

    function pow_pullback(Δsp)
        ΔG = Δsp.G
        r, c = size(sp.G)
        Ĝ = zeros(r, c)

        prod_cum = zeros(r)
        sum_cum = zeros(size(sp.E, 1))
        for (i, e) in enumerate(multiexponents(c, n))
            for j in 1:size(sp.G, 2)
                if e[j] == 0
                    continue
                end

                α = e[j]
                e[j] -= 1
                prod!(prod_cum, non_zero_power(sp.G, e))
                e[j] += 1

                Ĝ[:,j] .+= multinomial(e...) .* (@view ΔG[:,i]) .* α .* prod_cum
            end
        end

        # return tangent types for **all** arguments of the rrule
        # so for ::typeof(power_n_loop), sp, n, l, u  (l, u only used in comparison -> Zero gradient)
        return NoTangent(), Tangent{SparsePolynomial}(G=Ĝ, E=NoTangent(), ids=NoTangent()), NoTangent(), ZeroTangent(), ZeroTangent()
    end

    return ŝp, pow_pullback
end


"""
Custom quad prop with bounds on the ReLU inputs, s.t. we don't calculate
quadratic map for fixed ReLUs
"""
function fast_quad_prop(a, b, c, sp::SparsePolynomial{N,M}, l, u) where {N,M}
    n, m = size(sp.G)

    G_lin = b .* sp.G
    p = SparsePolynomial([c G_lin], [zeros(M, length(sp.ids)) sp.E], sp.ids)

    if sum((l .< 0) .& (u .> 0)) != 0
        # only if there are crossing ReLUs
        p_quad = pow(sp, 2, l, u)
        p_quad = multiply(a, p_quad)
        p = exact_addition(p_quad, p)
    end

    return p
end


"""
Propagates lower and upper bounds through different lower and upper relaxations,
but ensures that both share the same exponent matrix.

This is achieved by don't truncating them individually and sending them through
a merged relaxation (although still non-parallel and separate).
"""
function quad_prop_common(cₗ, cᵤ, spl, spu, ll, lu, ul, uu)
    n = size(spl.G, 1)
    sp_both = SparsePolynomial([spl.G; spu.G], spl.E, spl.ids)
    ls = [ll; ul]
    us = [lu; uu]
    cs = [cₗ; cᵤ]

    ŝp = fast_quad_prop(cs[:,3], cs[:,2], cs[:,1], sp_both, ls, us)

    L̂ = SparsePolynomial(ŝp.G[1:n,:], ŝp.E, ŝp.ids)
    Û = SparsePolynomial(ŝp.G[n+1:end,:], ŝp.E, ŝp.ids)

    return L̂, Û
end
