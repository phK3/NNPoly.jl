
# forward propagation of linear symbolic intervals for comparison to polynomials.

"""
Linear symbolic interval
"""
struct SymbolicIntervalDiff
    Λ # matrix for symbolic lower bound
    λ # vector for constant part of symbolic lower bond
    Γ # matrix for symbolic upper bound
    γ # vector for constant part of symbolic upper bound
    lb # lower bound of input domain
    ub # upper bound of input domain
    lbs # lower bounds for intermediate neurons
    ubs # upper bounds for intermediate neurons
end


function init_symbolic_interval_diff(net::NV.NetworkNegPosIdx, input::AbstractHyperrectangle)
    n = dim(input)

    Λ = Matrix(I(n))
    λ = zeros(n)
    Γ = Matrix(I(n))
    γ = zeros(n)

    layer_sizes = [length(l.bias) for l in net.layers]
    lbs = [fill(-Inf, ls) for ls in layer_sizes]
    ubs = [fill( Inf, ls) for ls in layer_sizes]

    return SymbolicIntervalDiff(Λ, λ, Γ, γ, low(input), high(input), lbs, ubs)
end


"""
calculates element-wise lower and upper bounds of a function A*x + b for x ∈ [lb, ub]
"""
function bounds(A::AbstractMatrix, b::AbstractVector, lb::AbstractVector, ub::AbstractVector)
    A⁻ = min.(0, A)
    A⁺ = max.(0, A)

    low = A⁻ * ub .+ A⁺ * lb .+ b
    up  = A⁻ * lb .+ A⁺ * ub .+ b

    return low, up
end


function print_info(s::SymbolicIntervalDiff)
    n = size(s.Λ, 1)

    println("symbolic lower bound")
    for i in 1:n
        println("\t", [s.Λ[i,:]; s.λ[i]])
    end

    println("symbolic upper bound")
    for i in 1:n
        println("\t", [s.Γ[i,:]; s.γ[i]])
    end
end
