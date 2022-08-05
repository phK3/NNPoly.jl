

"""
Calculates chebyshev relaxation of a univariate monomial.

First calculates the chebyshev approximation c(x) of the monomial x^n, then
bounds the difference c(x) - x^n and x^n - c(x) via BaB to get symbolic polynomial lower
and upper relaxations.

args:
    exponent - (Integer) the exponent of the single variable in the monomial
    id - (Integer) the id of the single variable in the monomial

kwargs:
    degree - (Integer) the degree of the approximation of the monomial (defaults to exponent - 2)
    max_steps - (Integer) maximum number of steps for BaB procedure
    optimality_gap - (Float64) optimality gap for early stopping of BaB procedure
"""
function relax_monomial_cheby(exponent, id; degree=nothing, max_steps=30, optimality_gap=1e-4)
    if exponent == 1
        lcheby = negate(make_monomial([0], ids=[id]))
        ucheby = make_monomial([0], ids=[id])
        return lcheby, ucheby
    elseif exponent == 2
        lcheby = linear_map(0, make_monomial([0], ids=[id]))
        ucheby = make_monomial([0], ids=[id])
        return lcheby, ucheby
    end
            
    degree = isnothing(degree) ? exponent - 2 : degree

    if iseven(exponent)
        relax_deg = iseven(degree) ? degree + 1 : degree
        cheby = cheby_even(x -> x^exponent, relax_deg)
    else
        relax_deg = isodd(degree) ? degree + 1 : degree
        cheby = cheby_odd(x -> x^exponent, relax_deg)
    end

    f = make_monomial([exponent])
    diff = subtract(f, cheby)
    ub =  max_in_dir_bab( [1], diff, printing=false, max_steps=max_steps, optimality_gap=optimality_gap)
    lb = -max_in_dir_bab([-1], diff, printing=false, max_steps=max_steps, optimality_gap=optimality_gap)

    lcheby = translate(cheby, [lb])
    ucheby = translate(cheby, [ub])
    lcheby.ids[1] = id
    ucheby.ids[1] = id

    return lcheby, ucheby
end
