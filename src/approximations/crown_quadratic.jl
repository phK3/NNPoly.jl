

"""
Returns the monomial coefficients cᵢ of the quadratic lower relaxation of a
ReLU neuron given concrete lower and upper bounds on its input.

Relaxation is computed as in the original CROWN paper.

returns:
    c - ([3]) with c = [c₁, c₂, c₃] and ReLu(x) ≥ c₁ + c₂x + c₃x²
"""
function relax_relu_crown_quad_lower(l, u)
    if u <= 0
        cs = [0, 0, 0]
    elseif l >= 0
        cs = [0, 1, 0]
    elseif abs(l) >= u
        cs = zeros(3)
    elseif u >= 2*abs(l)
        cs = [0, 1, 0]
    else
        a = u / (u^2 - l*u)
        b = -a * l
        c = 0
        cs = [c, b, a]
    end

    return cs
end


"""
Returns the monomial coefficients cᵢ of the quadratic upper relaxation of a
ReLU neuron given concrete lower and upper bounds on its input.

Relaxation is computed as in the original CROWN paper

returns:
    c - ([3]) with c = [c₁, c₂, c₃] and ReLu(x) ≤ c₁ + c₂x + c₃x²
"""
function relax_relu_crown_quad_upper(l, u)
    if u <= 0
        return [0, 0, 0]
    elseif l >= 0
        return [0, 1, 0]
    elseif abs(l) > u
        bias = l/(l - u) + 2*l/(u - l)
        k = 2/u - (l + u)/(l*u)
        θ = -bias / k
    else
        bias = l/(l - u) + 2*u/(u - l)
        k = 2/l - (l + u)/(l*u)
        θ = (1 - bias) / k
    end

    a = 1/(u - l) + θ/(u*l)
    b = ((l^2*u)/(l - u) - θ*(l + u)) / (u*l)
    c = θ

    # don't want quadratic terms with large coefficients!
    if a > 10
        a = 0
        b = u / (u - l)
        c = -l*b
    end

    return [c, b, a]
end
