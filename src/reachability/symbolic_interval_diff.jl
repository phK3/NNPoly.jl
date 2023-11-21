
# forward propagation of linear symbolic intervals for comparison to polynomials.

"""
Linear symbolic interval
"""
struct SymbolicIntervalDiff{N<:Number}
    Λ::Matrix{N} # matrix for symbolic lower bound
    λ::Vector{N} # vector for constant part of symbolic lower bond
    Γ::Matrix{N} # matrix for symbolic upper bound
    γ::Vector{N} # vector for constant part of symbolic upper bound
    lb::Vector{N} # lower bound of input domain
    ub::Vector{N} # upper bound of input domain
    lbs::Vector{Vector{N}} # lower bounds for intermediate neurons
    ubs::Vector{Vector{N}} # upper bounds for intermediate neurons
end


function init_symbolic_interval_diff(net::NV.NetworkNegPosIdx, input::AbstractHyperrectangle{<:N}) where N<:Number
    n = dim(input)

    Λ = Matrix{N}(I(n))
    λ = zeros(N, n)
    Γ = Matrix{N}(I(n))
    γ = zeros(N, n)

    layer_sizes = [length(l.bias) for l in net.layers]
    lbs = [fill(-Inf, ls) for ls in layer_sizes]
    ubs = [fill( Inf, ls) for ls in layer_sizes]

    return SymbolicIntervalDiff(Λ, λ, Γ, γ, low(input), high(input), lbs, ubs)
end


"""
calculates element-wise lower and upper bounds of a function A*x + b for x ∈ [lb, ub]
"""
#function bounds(A::AbstractMatrix, b::AbstractVector, lb::AbstractVector, ub::AbstractVector)
function bounds(A::Matrix{N}, b::Vector{N}, lb::Vector{N}, ub::Vector{N}) where N<:Number
    A⁻ = min.(zero(eltype(A)), A)
    A⁺ = max.(zero(eltype(A)), A)

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
