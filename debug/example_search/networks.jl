
# ==========================================================================================
# Network construction, sampling and nice-ification
# ==========================================================================================

# Default alphabets (same as the old debug/find_compare_examples.jl)
const DEFAULT_WEIGHTS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
const DEFAULT_BIASES  = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Widened alphabets. All values are dyadic rationals (k/2^n), so the exact
# Rational{BigInt} niceness analysis keeps small denominators.
const WIDE_WEIGHTS = [-8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5,
                       0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
const WIDE_BIASES  = [-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
                       0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
const WIDE_B3      = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

"""
Build an `NV.Network` with ReLU hidden layers and (by default) an `Id` output layer.
Generic in the number type (works for Float64 and Rational{BigInt}).
"""
function make_network(Ws, bs; last_act=NV.Id())
    N = eltype(bs[1])
    layers = NV.Layer{<:NV.ActivationFunction, N}[]
    for i in 1:length(Ws)-1
        push!(layers, NV.Layer(Ws[i], bs[i], NV.ReLU()))
    end
    push!(layers, NV.Layer(Ws[end], bs[end], last_act))
    return NV.Network(layers)
end

make_network_npi(Ws, bs; last_act=NV.Id()) = NV.NetworkNegPosIdx(make_network(Ws, bs; last_act=last_act))

"""
Build a `Flux.Chain` of `NP.CROWNLayer`s for aCROWN / PolyCROWN.

With `poly=true` the first ReLU layer gets a 2x2 α tensor (quolynomial
relaxation parameters) as used by PolyCROWN.
"""
function make_crown_chain(Ws, bs; poly=false, last_act=NV.Id())
    layers = Any[]
    for i in 1:length(Ws)-1
        if i == 1 && poly
            α = similar(bs[i], length(bs[i]), 2, 2)
        else
            α = similar(bs[i], length(bs[i]), 1)
        end
        push!(layers, NP.CROWNLayer(Ws[i], bs[i], NV.ReLU(), α))
    end
    push!(layers, NP.CROWNLayer(Ws[end], bs[end], last_act, similar(bs[end], 0)))
    return Chain(layers...)
end

"""
Sample a random tiny candidate network: single input, two hidden ReLU layers
(widths `h1`, `h2`), single output. Weights/biases are drawn from the given
alphabets.

If `structured_W1=true`, the first-layer weights are drawn from a smaller pool
with repetition, which promotes error-term cancellation in the second layer
(the structural pattern that makes the dependency effect visible).

If `nice_bias=true`, the first layer is sampled so that only the *first* neuron
is crossing-capable and all others are one-sided (active or inactive over the
whole input range). Layer-2 quantities then inherit only a *single* λ
denominator from layer 1, which drastically reduces the denominators of the
layer-3 quantities (the dominant source of niceness violations).

`b3_pool` is the alphabet for the (single) output bias.
"""
function sample_candidate(rng::AbstractRNG; h1=2, h2=2,
                          weights=DEFAULT_WEIGHTS, biases=DEFAULT_BIASES,
                          b3_pool=[-0.5, 0.0, 0.5],
                          structured_W1=false, nice_bias=false)
    W1_pool = structured_W1 ? [w for w in weights if abs(w) >= 1.0] : weights

    W1 = Float64.(rand(rng, W1_pool, h1, 1))
    b1 = Float64.(rand(rng, biases, h1))

    if nice_bias
        # neurons 2..h1: one-sided over the input range, i.e. b - |w| >= 0 (always active)
        # or b + |w| <= 0 (always inactive)
        for j in 2:h1
            for _ in 1:100
                w = Float64(rand(rng, weights))
                b = Float64(rand(rng, biases))
                if b - abs(w) >= 0 || b + abs(w) <= 0
                    W1[j, 1] = w
                    b1[j] = b
                    break
                end
            end
        end
    end

    W2 = Float64.(rand(rng, weights, h2, h1))
    b2 = Float64.(rand(rng, biases, h2))

    W3 = Float64.(rand(rng, weights, 1, h2))
    b3_out = Float64.(rand(rng, b3_pool, 1))

    return [W1, W2, W3], [b1, b2, b3_out]
end

"""
    rescale_layer(Ws, bs, i, c)

Scale the pre-activations of hidden layer `i` (1-based) by `c > 0`: multiply
`Wᵢ, bᵢ` by `c` and `Wᵢ₊₁` by `1/c`. Since ReLU is positively homogeneous, the
network function — and therefore *all* verification bounds — is unchanged, but
centers/generators/constants of the intermediate representations get rescaled.
Useful to clear denominators of ugly constants (note: chord slopes λ = u/(u-l)
are invariant under this rescaling).
"""
function rescale_layer(Ws, bs, i, c)
    @assert 1 <= i <= length(Ws) - 1 "can only rescale hidden layers"
    Ws_new = [j == i ? c .* Ws[j] : (j == i + 1 ? Ws[j] ./ c : copy(Ws[j])) for j in eachindex(Ws)]
    bs_new = [j == i ? c .* bs[j] : copy(bs[j]) for j in eachindex(bs)]
    return Ws_new, bs_new
end

"""
    rescale_variants(Ws, bs; factors=(2.0, 4.0, 0.5, 0.25))

All single-layer rescalings of a candidate (identity included).
"""
function rescale_variants(Ws, bs; factors=(2.0, 4.0, 0.5, 0.25))
    variants = [(copy(Ws), copy(bs), :identity)]
    for i in 1:length(Ws)-1, c in factors
        push!(variants, (rescale_layer(Ws, bs, i, c)..., Symbol("layer", i, "_x", c)))
    end
    return variants
end
