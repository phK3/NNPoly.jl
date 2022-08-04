


function chebyshev_polynomial(n)
    if n == 0
        return SparsePolynomial(vecOfVec2Mat([[1]]), vecOfVec2Mat([[0]]), [1])
    elseif n == 1
        return SparsePolynomial(vecOfVec2Mat([[1]]), vecOfVec2Mat([[1]]), [1])
    else
        T1 = chebyshev_polynomial(n-1)
        T2 = chebyshev_polynomial(n-2)

        # T[n](x) = 2*x*T[n-1](x) - T[n-2](x)
        # 2*x*T[n-1](x) same as multiplying T[n-1] by 2 and increasing its exponents by 1
        Ĝ = 2 .* T1.G
        Ê = T1.E .+ 1
        T̂ = SparsePolynomial([Ĝ -T2.G], [Ê T2.E], T1.ids)
        return T̂
    end
end


function chebyshev_coefficients(f, degree::Integer)
    N = degree + 1
    return [2/N * sum([f(cos(π*(k + 0.5) / N)) * cos(π*j*(k + 0.5) / N) for k =0:N-1]) for j = 0:N-1]
end


function all_chebys(n)
    G = zeros(n+1, n+1)
    E = vecOfVec2Mat(collect(0:n))'

    # T[0](x) and T[1](x)
    G[1,1] = 1
    G[2,2] = 1
    for i in 3:n+1
        # increase exponent by one -> shift to right
        G[i,2:end] .= 2 .* G[i-1,1:end-1]
        G[i,:] .-= G[i-2,:]
    end

    return SparsePolynomial(G, E, [1])
end


# assumes that function input ranges in [-1, 1]
function chebyshev_approximation(f, degree::Integer)
    c = chebyshev_coefficients(f, degree)
    chebys = all_chebys(degree)
    return translate(linear_map(c', chebys), -0.5 * [c[1]])
end


# function input in [l, u]
function chebyshev_approximation(f, degree::Integer, l, u)
    f̂(y) = f(0.5*(u - l)*y + 0.5*(u + l))
    cheby = chebyshev_approximation(f̂, degree)
    return rescale_variable(cheby, 1, l, u)
end


"""
Chebyshev approximation of odd function.

Chebyshev approximations of odd functions have only odd monomials with non-zero
coefficients. Therefore, if we want a third-order approximation of odd f(x),
we can calculate the fourth-order chebyshev approximation and use its sampling
points, as we know the quartic term will be zero.

f - the function to approximate (must be an odd function)
degree - the degree of the approximation (which can be one higher than for usual
    chebyshev approximation due to the function being odd)
"""
function cheby_odd(f, degree)
    cheby = NP.chebyshev_approximation(f, degree)
    odd_mask = vec(isodd.(cheby.E))
    G = cheby.G[:, odd_mask]
    E = cheby.E[:, odd_mask]
    return SparsePolynomial(G, E, cheby.ids)
end


"""
Chebyshev approximation of even function.

Chebyshev approximations of even functions have only even monomials with non-zero
coefficients. Therefore, if we want a fourth-order approximation of even f(x),
we can calculate the fifth-order chebyshev approximation and use its sampling
points, as we know the quintic term will be zero.

f - the function to approximate (must be an even function)
degree - the degree of the approximation (which can be one higher than for usual
    chebyshev approximation due to the function being even)
"""
function cheby_even(f, degree)
    cheby = NP.chebyshev_approximation(f, degree)
    even_mask = vec(iseven.(cheby.E))
    G = cheby.G[:, even_mask]
    E = cheby.E[:, even_mask]
    return SparsePolynomial(G, E, cheby.ids)
end


"""
Returns the monomial coefficients cᵢ of the Chebyshev approximation and the
maximum absolute approximation error ϵ,
s.t. ReLU(x) ∈ c₁ + c₂*x + c₃*x^2 + ... +  ± ϵ for x ∈ [lb, ub]
"""
function relax_relu_chebyshev(lb, ub, degree::Integer)
    if ub <= 0
        return zeros(degree+1), 0
    elseif lb >= 0
        cs = zeros(degree+1)
        cs[2] = 1
        return cs, 0
    else
        # f(x) = -x
        linfun = SparsePolynomial(vecOfVec2Mat([[-1]]), vecOfVec2Mat([[1]]), [1])
        relu_cheby = chebyshev_approximation(x -> max(0, x), degree, lb, ub)

        err_l0, err_u0 = calculate_extrema(relu_cheby, lb, 0)  # error for approximation over inactive region
        err_l1, err_u1 = calculate_extrema(exact_addition(relu_cheby, linfun), 0, ub)  # error for approximation over active region
        ϵₗ = min(err_l0, err_l1)
        ϵᵤ = max(err_u0, err_u1)
        ϵ = ϵᵤ - ϵₗ

        cs = get_monomial_coefficients(relu_cheby)
        cs[1] -= 0.5 * (ϵᵤ + ϵₗ)
        return cs, 0.5*ϵ  # half as we return the center function
    end
end
