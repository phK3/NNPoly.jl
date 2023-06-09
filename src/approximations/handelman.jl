

"""
Get coefficient vector for expansion of (x - l)^n₁ * (u - x)^n₂

Coefficients are sorted in order of increasing degree:
c₁ + c₂x + c₃x² + ... + cₙxⁿ with n = max_degree

optional: make coefficient vector have entries up to max_degree (has to be larger than n₁ + n₂)
"""
function multi_binomial(l, u, n₁, n₂; max_degree=-1)
    max_degree = max(max_degree, n₁ + n₂)
    cs = zeros(max_degree + 1)
    for k₁ in 0:n₁
        for k₂ in 0:n₂
            c = binomial(n₁, k₁)*binomial(n₂, k₂) * (-l)^(n₁ - k₁) * u^(n₂ - k₂)
            cs[k₁ + k₂ + 1] += k₂ % 2 == 0 ? c : -c
        end
    end
    return cs
end


"""
Get multi-binomial factor for x^k₁*x^k₂ in expansion of (x - l)^n₁ * (u - x)^n₂

Note the multi-binomial theorem for this instance:
(x - l)^n₁ * (u - x)^n₂ = (x - l)^n₁ * (-x + u)^n₂
    = ∑ ∑ binom(n₁, k₁)*binom(n₂, k₂) * x^k₁ * (-l)^(n₁ - k₁) * (-x)^n₂ * u^(n₂ - k₂)

With this function, we calculate one summation term.
"""
function single_multi_binomial(l, u, n₁, n₂, k₁, k₂)
    return binomial(n₁, k₁)*binomial(n₂, k₂)*(-1)^k₂ * (-l)^(n₁ - k₁) * u^(n₂ - k₂)
end


"""
Get coefficient for x^n in expansion of (x - l)^n₁ * (u - x)^n₂
"""
function elemwise_multi_binomial(l, u, n, n₁, n₂)
    val = 0
    for k₁ = 0:n₁
        for k₂ = 0:n₂
            if k₁ + k₂ == n
                val += single_multi_binomial(l, u, n₁, n₂, k₁, k₂)
            end
        end
    end
    return val
end


"""
Get pairs of exponents of x^n₁*x^n₂ s.t. the maximal degree is degree.
We first set a value for n₁ and then increment n₂ until n₁ + n₂ = degree, then
we increment n₁, ...

Example: (for degree = 2)
    n₁  n₂
    0   0
    0   1
    0   2
    1   0
    1   1
    2   0
"""
function get_exponent_pairs(degree)
    return [(n₁, n₂) for n₁ in 0:degree for n₂ in 0:degree - n₁]
end

# the function is not differentiable
@non_differentiable get_exponent_pairs(degree)


"""
Get matrix corresponding to comparison of coefficients with respect to
expansion of Handelman, s.t. polynomial p(x) is positive over x ∈ [l, u].

p(x) = ∑ sᵢ (x-l)ⁿ(u-x)ᵐ, where n = 0,..,d and m = 0,...,d-n
"""
function get_handelman_coefficients(l, u, degree)
    #idxs = Tuple{Int, Int}[]
    #ChainRulesCore.ignore_derivatives() do
        # Zygote doesn't like this list comprehension
    #    idxs = [(n₁, n₂) for n₁ in 0:degree for n₂ in 0:degree - n₁]
    #end
    idxs = get_exponent_pairs(degree)
    A = map(x -> elemwise_multi_binomial(l, u, x[1], x[2], x[3]), [(d,i,j) for d in 0:degree, (i,j) in idxs])
    return A
end


# old handelman function, but isn't differentiable by Zygote
#function get_handelman_coefficients(l, u, degree)
#    A = zeros(Int(0.5*(degree+1)*(degree+2)), degree + 1)
#    i = 1
#    for n₁ in 0:degree
#        for n₂ in 0:degree - n₁
#            A[i,:] .= multi_binomial(l, u, n₁, n₂, max_degree=degree)
#            i += 1
#        end
#    end
#
#    return Matrix(A')
#end


## ReLU relaxations

"""
Calculate upper polynomial ReLU relaxation using Handelman's Positivstellensatz.

For y = ReLU(x) with x ∈ [lb, ub], we overapproximate
y ≤ u(x) with a polynomial of the specified degree.

cl(x) ≥ 0 for x ∈ [l, 0]
cu(x) - x ≥ 0 for x ∈ [0, u]
u(x) ≥ max(cl(x), cu(x))

args:
    lb - concrete lower bound of the ReLU's input
    ub - concrete upper bound of the ReLU's input
    degree - degree of the relaxation polynomial
    s - Handelman multipliers for the polynomial approximation in [l, 0]
    t - Handelman multipliers for the polynomial approximation in [0, u]

returns:
    upper relaxation - ([degree+1]) with c₁ + c₂x + c₃x^2 + ... ≥ ReLU(x)
"""
function get_upper_polynomial(lb, ub, degree, s, t)
    if lb >= 0
        # differentiable way to get unit vector
        e₂ = (1:degree+1 .== 2)
        return e₂
    elseif ub <= 0
        return zeros(degree + 1)
    end

    Aₗ = get_handelman_coefficients(lb, 0, degree)
    Aᵤ = get_handelman_coefficients(0, ub, degree)

    e₂ = (1:degree + 1 .== 2)
    cl = Aₗ*s.^2
    cu = Aᵤ*t.^2 .+ e₂
    l, u = calculate_extrema(cu .- cl, lb, ub)

    λ = NV.relaxed_relu_gradient(l, u)
    β = -l

    e₁ = (1:degree + 1 .== 1)   # basis vector [1,0,0,...]
    cs = λ .* (cu .- cl) .+ e₁ .* λ*max(0, β)

    return cs .+ cl
end


"""
Calculate lower polynomial ReLU relaxation using Handelman's Positivstellensatz.

For y = ReLU(x) with x ∈ [lb, ub], we underapproximate
y ≥ l(x) with a polynomial of the specified degree.

-cl(x) ≥ 0 for x ∈ [l, u]
-cu(x) + x ≥ 0 for x ∈ [l, u]
l(x) ≤ max(cl(x), cu(x))

args:
    lb - concrete lower bound of the ReLU's input
    ub - concrete upper bound of the ReLU's input
    degree - degree of the relaxation polynomial
    s - Handelman multipliers for the polynomial approximation ≤ 0
    t - Handelman multipliers for the polynomial approximation ≤ x

returns:
    lower relaxation - ([degree+1]) with c₁ + c₂x + c₃x^2 + ... ≤ ReLU(x)
"""
function get_lower_polynomial(lb, ub, degree, s, t)
    if lb >= 0
        # differentiable way to get unit vector
        e₂ = (1:degree+1 .== 2)
        return e₂
    elseif ub <= 0
        return zeros(degree + 1)
    end

    Aₗ = get_handelman_coefficients(lb, ub, degree)
    Aᵤ = get_handelman_coefficients(lb, ub, degree)

    e₂ = (1:degree + 1 .== 2)
    cl = -Aₗ*s.^2
    cu = -Aᵤ*t.^2 .+ e₂

    l, u = calculate_extrema(cu .- cl, lb, ub)
    # maybe add λ ∈ [0, 1] to the optimization variables
    λ = NV.relaxed_relu_gradient(l, u)  # is it better not to choose 0 or 1
    cs = λ .* (cu .- cl)

    return cs .+ cl
end


"""
Calculate lower polynomial ReLU relaxation using Handelman's Positivstellensatz.

For y = ReLU(x) with x ∈ [lb, ub], we underapproximate
y ≥ l(x) with a polynomial of the specified degree.

-cl(x) ≥ 0 for x ∈ [l, 0]
-cu(x) + x ≥ 0 for x ∈ [0, u]
l(x) ≤ min(cl(x), cu(x))

Comparison to relaxation via maximum: If cl(x) and cu(x) are not equal, there
will be greater overapproximation due to the overapproximation of min!

args:
    lb - concrete lower bound of the ReLU's input
    ub - concrete upper bound of the ReLU's input
    degree - degree of the relaxation polynomial
    s - Handelman multipliers for the polynomial approximation ≤ 0
    t - Handelman multipliers for the polynomial approximation ≤ x

returns:
    lower relaxation - ([degree+1]) with c₁ + c₂x + c₃x^2 + ... ≤ ReLU(x)
"""
function get_lower_polynomial_min(lb, ub, degree, s, t)
    Aₗ = NP.get_handelman_coefficients(lb, 0, degree)
    Aᵤ = NP.get_handelman_coefficients(0, ub, degree)

    e₂ = (1:degree + 1 .== 2)
    cl = -Aₗ*s.^2
    cu = -Aᵤ*t.^2 .+ e₂

    l, u = NP.calculate_extrema(cl .- cu, lb, ub)  # swap order of subtraction for -(cu - cl)

    λ = NV.relaxed_relu_gradient(l, u)
    β = -l

    e₁ = (1:degree + 1 .== 1)   # basis vector [1,0,0,...]
    cs = λ .* (cl .- cu) .+ e₁ .* λ*max(0, β)

    return cl .- cs
end
