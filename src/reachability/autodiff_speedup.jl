
# Zygote can't correctly differentiate through NV.relaxed_relu_gradient.(l, u) (yielding all zero gradients)
# differentiating through [NV.relaxed_relu_gradient(lᵢ, uᵢ) for lᵢ, uᵢ in zip(l, u)] works, but is highly inefficient
# so we defined a vectorized version
function relaxed_relu_gradient_vectorized(l::Vector{N}, u::Vector{N}) where N<:Number
    return NV.relaxed_relu_gradient.(l, u)
end


function ChainRulesCore.rrule(::typeof(relaxed_relu_gradient_vectorized), l::Vector{N}, u::Vector{N}) where N<:Number
    λ = relaxed_relu_gradient_vectorized(l, u)
    
    function relaxed_relu_gradient_vectorized_pullback(Δλ)
        fixed = u .< 0 .|| l .>= 0
        # because function works element-wise, jacobians are always diagonal matrices
        # so we can use hadamard product instead for efficient computation
        # Δl = ∂λ/∂l .* ∂a/∂λ  (where a is the last output, Δλ = ∂a/∂λ)
        # similar for Δu
        Δl = @thunk(ifelse.(fixed, zero(N), u ./ (u .- l).^2) .* Δλ)
        Δu = @thunk(ifelse.(fixed, zero(N), one(N) ./ (u .- l) - u ./ (u .- l).^2) .* Δλ)
        
        return NoTangent(), Δl, Δu
    end
    
    return λ, relaxed_relu_gradient_vectorized_pullback
end