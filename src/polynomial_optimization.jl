

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

    # one coefficient -> constant, 2 coeffs -> linear, 3 coeffs -> quadratic, ...
    if length(cs) == 1
        # !!! be careful, optimal value is not only at x_opt
        # if used in conjunction with calculate_extrema(p, x, lb, ub) that doesn't
        # matter, as the p(lb) and p(ub) have the same value
        x_opt = [0]
    elseif length(cs) == 2
        # if x has non-zero coefficient, then it is unbounded
        x_opt = []
    elseif length(cs) == 3
        # Julia is 1-indexed, so p(x) = cs[1] + cs[2]*x + cs[3]*x²
        x_opt = [-0.5*(cs[2] / cs[3])]
    elseif length(cs) == 4
        # cs[4]*x³ + cs[3]*x² + cs[2]*x + cs[1]
        x_opt = quadratic_roots(3*cs[4], 2*cs[3], cs[2])
    elseif length(cs) == 5
        x_opt = cardano_roots(4*cs[5], 3*cs[4], 2*cs[3], cs[2], only_real_roots=true)
    else
        throw(ArgumentError("Polynomial has degree larger than 4! p = $p"))
    end

    y_opt = [evaluate(sp, xᵢ)[1] for xᵢ in x_opt]
    return x_opt, y_opt
end


"""
Calculates the extrema of the one-dimensional univariate polynomial p(x) over x ∈ [lb, ub].
Returns the minimum and maximum value of p(x) over the domain.
"""
function calculate_extrema(sp::SparsePolynomial, lb, ub)
    x_opt, y_opt = calculate_critical_points(sp)
    yₗ = evaluate(sp, lb)[1]
    yᵤ = evaluate(sp, ub)[1]

    ys = [y_opt[i] for i in 1:length(y_opt) if lb < x_opt[i] && x_opt[i] < ub]
    ys = [ys; yₗ; yᵤ]

    return minimum(ys), maximum(ys)
end
