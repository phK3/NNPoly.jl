
# ==========================================================================================
# Exact-rational support for the forward methods.
#
# `NV.relaxed_relu_gradient` returns Float64 literals (0.0 / 1.0) for non-crossing ReLUs,
# which would promote exact Rational{BigInt} symbolic bounds to Float64. This narrower
# method keeps rational propagation exact. Adding a *more specific* method is safe:
# dispatch prefers it for Rational inputs, and NV's generic `(::Real, ::Real)` method
# still serves everything else.
# ==========================================================================================

NV.relaxed_relu_gradient(l::Rational, u::Rational) = u <= 0 ? zero(u) : (l >= 0 ? one(u) : u / (u - l))

"""
    rationalize_network(Ws, bs) -> (Ws_rat, bs_rat)

Convert network weights/biases to `Rational{BigInt}` (exact for the small
alphabet used by the search).
"""
function rationalize_network(Ws, bs)
    return [Rational{BigInt}.(W) for W in Ws], [Rational{BigInt}.(b) for b in bs]
end

"""
    rational_input(low, high) -> Hyperrectangle{Rational{BigInt}}

Input set with exact rational endpoints.
"""
function rational_input(low, high)
    return Hyperrectangle(low=[Rational{BigInt}(low)], high=[Rational{BigInt}(high)])
end
