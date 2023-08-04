

"""
find real roots of a*x² + b*x + c
"""
function quadratic_roots(a, b, c)
    discriminant = b^2 - 4*a*c
    if discriminant == 0
        return [-0.5*b / a]
    elseif discriminant < 0
        return []
    else
        x₁ = -0.5*(b + sqrt(discriminant)) / a
        x₂ = -0.5*(b - sqrt(discriminant)) / a
        return [x₁, x₂]
    end
end


"""
find real roots to the cubic problem ax³+bx²+cx+d = 0 using Cardano's formula
"""
function cardano_roots(a,b,c,d; only_real_roots=false)
    cbr₁ = (-0.5 - 0.5*sqrt(3)im)  # cube root of 1
                                   # do we need it for finding all solutions?
    q = (3*a*c - b^2) / (9*a^2)
    r = (9*a*b*c - 27*a^2*d - 2*b^3) / (54*a^3)

    discriminant = q^3 + r^2

    if discriminant <= 0
        # in this case all roots are real numbers
        # cbrt is not defined for complex arguments
        s = (Complex(r) + sqrt(Complex(discriminant)))^(1/3)
        t = (Complex(r) - sqrt(Complex(discriminant)))^(1/3)
    else
        # there is only one real root
        s = cbrt(r + sqrt(q^3 + r^2))
        t = cbrt(r - sqrt(q^3 + r^2))
    end

    x1 = s + t - (b / (3*a))
    x2 = -0.5*(s + t) - (b / (3*a)) + 0.5*(s - t)*sqrt(3)im
    x3 = -0.5*(s + t) - (b / (3*a)) - 0.5*(s - t)*sqrt(3)im

    if only_real_roots
        if discriminant <= 0
            @assert imag(x1) == 0 && imag(x2) == 0 && imag(x3) == 0
            return [real(x) for x in [x1, x2, x3]]
        else
            return [real(x) for x in [x1, x2, x3] if abs(imag(x)) < 1e-14]
        end
    end

    return [x1, x2, x3]
end


"""
Calculates the critical points (roots of the derivative) of the univariate polynomial p.

Can handle one-dimensional univariate polynomials up to degree 4.

returns
    x_opt ([Number]) - positions of extrema of p
    y_opt ([Number]) - values of p at x_opt
"""
function calculate_critical_points(sp::SparsePolynomial)
    @assert size(sp.G, 1) == 1  string("only one-dimsional polynomials!")
    @assert length(sp.ids) == 1 string("only univariate polynomials!")
    cs = get_monomial_coefficients(sp)

    return calculate_critical_points(cs)
end


"""
Calculates the critical points (roots of the derivative) of the univariate polynomial
p(x) = c₁ + c₂x + c₃x² + c₄x³ + ...

Can handle one-dimensional univariate polynomials up to degree 4.

returns
    x_opt ([Number]) - positions of extrema of p
    y_opt ([Number]) - values of p at x_opt
"""
function calculate_critical_points(cs::AbstractVector{<:T}) where T
    # one coefficient -> constant, 2 coeffs -> linear, 3 coeffs -> quadratic, ...

    # if terms for last x^n are zero, then it is only n-k degree polynomial!
    idx = @ignore_derivatives findlast(x -> x .!= 0, cs)
    # idx can be nothing, if all entries are 0
    # ĉs = isnothing(idx) ? zeros(T, 1) : cs[1:idx]
    idx = isnothing(idx) ? 1 : idx

    if idx == 1 #length(cs) == 1
        # !!! be careful, optimal value is not only at x_opt
        # if used in conjunction with calculate_extrema(p, x, lb, ub) that doesn't
        # matter, as the p(lb) and p(ub) have the same value
        x_opt = [0]
    elseif idx == 2 #length(cs) == 2
        # if x has non-zero coefficient, then it is unbounded
        x_opt = []
    elseif idx == 3 # length(cs) == 3
        # Julia is 1-indexed, so p(x) = cs[1] + cs[2]*x + cs[3]*x²
        # if cs[3] == 0 and cs[2] != 0, then it is unbounded
        # x_opt = cs[3] == 0 ? [] : [-0.5*(cs[2] / cs[3])]
        x_opt = [-0.5*(cs[2] / cs[3])]
    elseif idx == 4 #length(cs) == 4
        # cs[4]*x³ + cs[3]*x² + cs[2]*x + cs[1]
        x_opt = quadratic_roots(3*cs[4], 2*cs[3], cs[2])
    elseif idx == 5 # length(cs) == 5
        x_opt = cardano_roots(4*cs[5], 3*cs[4], 2*cs[3], cs[2], only_real_roots=true)
    else
        throw(ArgumentError("Polynomial has degree larger than 4! p = $p"))
    end

    # Zygote can't handle broadcast with empty input p.([])
    length(x_opt) == 0 && return Vector{T}(), Vector{T}()

    p = x -> cs' * [x^i for i in 0:length(cs) - 1]
    y_opt = p.(x_opt)
    return x_opt, y_opt
end


"""
Calculates the extrema of the one-dimensional univariate polynomial p(x) over x ∈ [lb, ub].
Returns the minimum and maximum value of p(x) over the domain.
"""
function calculate_extrema(sp::SparsePolynomial, lb, ub)
    cs = get_monomial_coefficients(sp)
    return calculate_extrema(cs, lb, ub)
end


"""
Calculates the extrema of the one-dimensional univariate polynomial
p(x) = c₁ + c₂x + c₃x² + c₄x³ + ...
over x ∈ [lb, ub].

Returns the minimum and maximum value of p(x) over the domain.
"""
function calculate_extrema(cs, lb, ub)
    x_opt, y_opt = calculate_critical_points(cs)

    p = x -> cs' * [x^i for i in 0:length(cs) - 1]
    yₗ = p(lb)
    yᵤ = p(ub)

    # lower alternative is differentiable by Zygote
    # ys = [y_opt[i] for i in 1:length(y_opt) if lb < x_opt[i] && x_opt[i] < ub]
    ys = y_opt[(lb .< x_opt) .& (x_opt .< ub)]
    ys = [ys; yₗ; yᵤ]

    return minimum(ys), maximum(ys)
end


## Implicit Differentiation for Polynomial optimisation


"""
Calculates x_min, the minimizer of a polynomial p(x) = c₁ + c₂x + c₃x² + ...
s.t. p(x_min) = y_min, the minimum on interval [l, u].
"""
function poly_minimizer(cs::AbstractVector, l::N, u::N) where N<:Number
    x_opt, y_opt = calculate_critical_points(cs)

    p = x -> cs' * [x^i for i in 0:length(cs) - 1]
    yₗ = p(l)
    yᵤ = p(u)

    proj_mask = (l .< x_opt) .& (x_opt .< u)
    xs = [x_opt[proj_mask]; l; u]
    ys = [y_opt[proj_mask]; yₗ; yᵤ]

    i = argmin(ys)
    return return xs[i]
end


function forward_poly_minimizer(C, l, u)
    # forward method for ImplicitDifferentiation
    x = poly_minimizer.(eachrow(C), l, u)
    z = 0  # additional constraints info not needed
    return x, z
end

forward_poly_minimizer(x::ComponentArray) = forward_poly_minimizer(x.C, x.l, x.u)


function conditions_poly_minimizer(C, l, u, x, z)
    # conditions for ImplicitDifferentiation
    # if x really is the minimizer, it is the fixed point of projected gradient descent.
    n = size(C, 2) - 1
    # coeffs of derivative
    dC = C[:,2:end] .* (1:n)'
    x_powers = reduce(hcat, [x.^k for k in 0:n-1])
    ∇ₓp = sum(dC .* x_powers, dims=2)

    η = 0.01
    return x .- clamp.(x .- η * ∇ₓp, l, u)
end

conditions_poly_minimizer(y::ComponentArray, x, z) = conditions_poly_minimizer(y.C, y.l, y.u, x, z)

implicit_poly_min = ImplicitFunction(forward_poly_minimizer, conditions_poly_minimizer)


"""
Calculates the minimum values yᵢ_min of polynomials pᵢ(x) = C[i,1] + C[i,2]x + C[i,3]x² + ...
over the intervals [lᵢ, uᵢ].

args:
    C - Matrix of polynomial coefficients
    l - vector of lower bounds
    u - vector of upper bounds

returns:
    y_min - vector of minimum values
"""
function poly_minimum(C::AbstractMatrix, l, u)
    args = comp_vec_clu(C, l, u)
    x_opt = (first ∘ implicit_poly_min)(args)
    x_powers = reduce(hcat, [x_opt.^k for k in 0:size(C,2)-1])
    y_opt = sum(C .* x_powers, dims=2)
    return y_opt
end

"""
Calculates the maximum values yᵢ_max of polynomials pᵢ(x) = C[i,1] + C[i,2]x + C[i,3]x² + ...
over the intervals [lᵢ, uᵢ].

args:
    C - Matrix of polynomial coefficients
    l - vector of lower bounds
    u - vector of upper bounds

returns:
    y_max - vector of maximum values
"""
poly_maximum(C, l, u) = .-poly_minimum(.-C, l, u)


"""
Computes an upper bound for the maximum of a sparse polynomial in direction d
by branch and bound on the largest generator up to a certain number of steps.
"""
function max_in_dir_bab(d, sp::SparsePolynomial; max_steps=10, optimality_gap=1e-3, tol=1e-6, printing=false)
    p = linear_map(d', sp)

    # as all variables are normalized to [-1, 1], the center is the vector of all zeros
    center = zeros(length(p.ids))

    queue = PriorityQueue(Base.Order.Reverse)

    lb, ub = bounds(p)
    ub = ub[1]
    lb = evaluate(p, center)[1]
    enqueue!(queue, p, ub)

    for i in 1:max_steps

        if ub - lb <= optimality_gap
            printing && println("Found optimal value ∈ ", [lb, ub])
            return ub
        end

        poly, ub = peek(queue)
        dequeue!(queue)
        printing && println(i, ": max_x p(x) ∈ ", [lb, ub])

        p1, p2 = split_longest_generator(poly)
        lb1, ub1 = bounds(p1)
        lb2, ub2 = bounds(p2)
        ub1 = ub1[1]
        ub2 = ub2[1]

        # are bounds monotonically increasing, if we split the largest generator?
        # @assert (ub1 <= val + tol) && (ub2 <= val + tol) string("Bounds of splits should be tighter than parent's bounds! ($val, $ub1, $ub2)")
        # ub12 = max(ub1, ub2)

        lb1 = evaluate(p1, center)[1]
        lb2 = evaluate(p2, center)[1]
        lb = max(lb, lb1, lb2)

        # don't add pruned nodes to the queue
        ub1 > lb && enqueue!(queue, p1, ub1)
        ub2 > lb && enqueue!(queue, p2, ub2)
    end

    return ub
end
