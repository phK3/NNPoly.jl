

"""
Get lower polynomial relaxation l(x) = c₀ + c₁x + ... + cₙxⁿ of ReLU(x)
given coefficients c₁,...,cₙ.

The relaxation is obtained by calculating the maximum deviation between the
polynomial l̂(x) = c₁x + ... + cₙxⁿ and ReLU(x) over the input interval and then
shifting l̂(x) down to get a valid lower relaxation.

args:
    lb - lower bound on the ReLU input
    ub - upper bound on the ReLU input
    degree - degree of the polyomial relaxation
    c - given coefficients of the polynomial (constant coefficient is not needed)

returns:
    coefficients of the polynomial lower relaxation
"""
function get_lower_polynomial_shift(lb::N, ub::N, degree::Integer, c::AbstractVector{<:N}) where N <: Number
    if ub <= 0
        return zeros(degree + 1)
    elseif lb >= 0
        return (1:degree + 1 .== 2)
    end

    # only need x, x², x³, ... coefficient, bias gets adjusted by shift anyways
    ĉ = [0.; c]
    e₂ = (1:degree + 1 .== 2)

    # want p(x) ≤ ReLU(x) --> u = max p(x) - ReLU(x)
    # --> p(x) - ReLU(x) ≤ u --> p(x) - u ≤ ReLU(x)
    lₗ, uₗ = calculate_extrema(ĉ, lb, 0)  # p(x) - 0
    lᵤ, uᵤ = calculate_extrema(ĉ .- e₂, 0, ub) # p(x) - x

    u = max(uₗ, uᵤ)
    e₁ = (1:degree + 1 .== 1)
    return ĉ .- u .* e₁
end


"""
Get upper polynomial relaxation u(x) = c₀ + c₁x + ... + cₙxⁿ of ReLU(x)
given coefficients c₁,...,cₙ.

The relaxation is obtained by calculating the maximum deviation between the
polynomial û(x) = c₁x + ... + cₙxⁿ and ReLU(x) over the input interval and then
shifting û(x) up to get a valid lower relaxation.

args:
    lb - lower bound on the ReLU input
    ub - upper bound on the ReLU input
    degree - degree of the polyomial relaxation
    c - given coefficients of the polynomial (constant coefficient is not needed)

returns:
    coefficients of the polynomial upper relaxation
"""
function get_upper_polynomial_shift(lb::N, ub::N, degree::Integer, c::AbstractVector{<:N}) where N <: Number
    if ub <= 0
        return zeros(degree + 1)
    elseif lb >= 0
        return (1:degree + 1 .== 2)
    end

    # only need x, x², x³, ... coefficient, bias gets adjusted by shift anyways
    ĉ = [0.; c]
    e₂ = (1:degree + 1 .== 2)

    # want p(x) ≥ ReLU(x) --> l = min p(x) - ReLU(x)
    # --> p(x) - ReLU(x) ≥ l --> p(x) - l ≥ ReLU(x)
    lₗ, uₗ = calculate_extrema(ĉ, lb, 0)  # p(x) - 0
    lᵤ, uᵤ = calculate_extrema(ĉ .- e₂, 0, ub) # p(x) - x

    l = min(lₗ, lᵤ)
    e₁ = (1:degree + 1 .== 1)
    return ĉ .- l .* e₁
end