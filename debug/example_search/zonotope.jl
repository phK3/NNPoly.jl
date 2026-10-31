
# ==========================================================================================
# Zonotope propagation (rung iv).
#
# This is a transparent, number-type-generic re-implementation of the Theorem 3.1
# (Singh et al. 2018, "AI2"/DeepZ) ReLU relaxation that `NV.Ai2z()` (Ai2{Zonotope})
# performs via `overapproximate(Rectification(Z), Zonotope)`. We re-implement it to
# (a) stay exact in Rational{BigInt} arithmetic and
# (b) expose the intermediate quantities (interval bounds, slopes λ, error-generator
#     magnitudes μ) that a human needs for hand calculation.
#
# Convention: zonotope Z = ⟨c, G⟩ = {c + G*ϵ : ϵ ∈ [-1, 1]^m} (LazySets convention).
#
# ReLU relaxation for a crossing neuron with interval bounds [l, u], l < 0 < u:
#     relu(z) ≤ λ * (z - l),   λ = u / (u - l)
# and relu(z) = λ*(z - l) - e with error e ∈ [0, λ*(-l)]. In centered zonotope form:
#     c ← λ*c + μ,  G ← λ*G,  new generator column with entry μ = -λ*l/2.
# ==========================================================================================

"""
Interval (box) bounds of a zonotope, exact: lo = c - Σ|G|, hi = c + Σ|G|.
"""
function interval_bounds(Z::Zonotope)
    c = center(Z)
    G = genmat(Z)
    lo = c - vec(sum(abs.(G), dims=2))
    hi = c + vec(sum(abs.(G), dims=2))
    return lo, hi
end

"""
Per-neuron data of the Theorem-3.1 ReLU relaxation:
λ (chord slope), μ (new error-generator magnitude) and the crossing mask.
For non-crossing neurons: λ = 1 (active) or 0 (inactive), μ = 0.
"""
function relu_relaxation_data(lo, hi)
    λ = similar(lo)
    μ = similar(lo)
    crossing = falses(length(lo))
    for i in eachindex(lo)
        if lo[i] >= 0
            λ[i] = one(lo[i])
            μ[i] = zero(lo[i])
        elseif hi[i] <= 0
            λ[i] = zero(lo[i])
            μ[i] = zero(lo[i])
        else
            crossing[i] = true
            λ[i] = hi[i] / (hi[i] - lo[i])
            μ[i] = -λ[i] * lo[i] / 2
        end
    end
    return λ, μ, crossing
end

"""
Apply the Theorem-3.1 relaxation to a zonotope, returning the rectified zonotope.
"""
function forward_zono_relu(Z::Zonotope)
    lo, hi = interval_bounds(Z)
    λ, μ, crossing = relu_relaxation_data(lo, hi)

    c = center(Z)
    G = genmat(Z)
    n, m = size(G)

    c_new = λ .* c + μ .* crossing
    G_new = λ .* G
    if any(crossing)
        # one independent error generator per crossing neuron (row v, magnitude μ[v]);
        # merging them into a single column would couple the errors and under-approximate
        E = zeros(eltype(G), n, count(crossing))
        for (j, v) in enumerate(findall(crossing))
            E[v, j] = μ[v]
        end
        G_new = hcat(G_new, E)
    end

    return Zonotope(c_new, G_new), (lo=lo, hi=hi, λ=λ, μ=μ, crossing=crossing)
end

"""
Propagate `input_set` through `net` with the zonotope domain.

Returns `(Zout, trace)` where `trace[i]` holds the pre-activation zonotope, its
interval bounds, the ReLU relaxation data and the post-activation zonotope of
layer `i`. `T` selects the arithmetic (`Float64`, or `Rational{BigInt}` for
exact fractions).
"""
function zono_propagate(net::NV.Network, input_set::AbstractHyperrectangle, ::Type{T}=Float64) where T
    Z = input_zonotope(input_set, T)
    trace = []
    for (i, L) in enumerate(net.layers)
        pre = LazySets.affine_map(T.(L.weights), Z, T.(L.bias))
        if L.activation isa NV.ReLU
            post, data = forward_zono_relu(pre)
        else
            post = pre
            data = (lo=interval_bounds(pre)[1], hi=interval_bounds(pre)[2],
                    λ=ones(T, NV.n_nodes(L)), μ=zeros(T, NV.n_nodes(L)),
                    crossing=falses(NV.n_nodes(L)))
        end
        push!(trace, (layer=i, pre=pre, data..., post=post))
        Z = post
    end
    return Z, trace
end

"""
Convert a hyperrectangular input set to a zonotope in arithmetic `T`.
"""
function input_zonotope(input_set::AbstractHyperrectangle, ::Type{T}=Float64) where T
    lo = T.(low(input_set))
    hi = T.(high(input_set))
    c = (lo + hi) ./ 2
    G = Matrix(Diagonal((hi - lo) ./ 2))
    return Zonotope(c, G)
end

"""
Exact (for the zonotope) upper bound of the output zonotope: c + Σ|G| per output.
"""
zono_upper_bound(Z::Zonotope) = center(Z) + vec(sum(abs.(genmat(Z)), dims=2))

zono_upper_bound(net::NV.Network, input_set::AbstractHyperrectangle, ::Type{T}=Float64) where T =
    zono_upper_bound(zono_propagate(net, input_set, T)[1])

"""
Sanity check: our manual Theorem-3.1 propagation must agree with `NV.Ai2z()`
(the library's implementation via LazySets) up to floating-point noise.
"""
function validate_zono(net::NV.Network, input_set::AbstractHyperrectangle)
    ub_manual = zono_upper_bound(net, input_set)
    Z = NV.forward_network(NV.Ai2z(), net, convert(Zonotope, input_set))
    ub_a2z = center(Z) + vec(sum(abs.(genmat(Z)), dims=2))
    return maximum(abs.(ub_manual - ub_a2z))
end

# ==========================================================================================
# Pretty printing for hand calculation
# ==========================================================================================

"""
Format a number as a fraction string (exact for Rationals, rationalized for floats).
"""
frac_str(x::Rational) = string(x)
frac_str(x::Real) = string(rationalize(Float64(x), tol=1e-12))
frac_str(x::AbstractVector) = "[" * join(frac_str.(x), ", ") * "]"

function print_zonotope(io, Z::Zonotope; indentation=0)
    pre = "\t"^indentation
    c = frac_str(vec(center(Z)))
    G = genmat(Z)
    print(io, pre, "⟨", c, " | ")
    if size(G, 2) == 0
        print(io, "∅")
    else
        for j in 1:size(G, 2)
            print(io, frac_str(G[:, j]), (j < size(G, 2) ? ", " : ""))
        end
    end
    println(io, "⟩")
end

"""
Pretty-print the zonotope propagation trace of `net` on `input_set`:
per layer the pre-activation zonotope, interval bounds, the crossing ReLUs with
their chord slopes λ and error-generator magnitudes μ, and the post-activation
zonotope.
"""
function print_zono_trace(net::NV.Network, input_set::AbstractHyperrectangle, ::Type{T}=Rational{BigInt};
                          io=stdout) where T
    _, trace = zono_propagate(net, input_set, T)
    println(io, "Input set: ", frac_str(low(input_set)), " .. ", frac_str(high(input_set)),
            "  (ϵ ∈ [-1, 1])")
    for t in trace
        println(io, "=== Layer ", t.layer, " (", t.pre, ") ===")
        print(io, "\tpre-activation  : "); print_zonotope(io, t.pre, indentation=0)
        println(io, "\tinterval bounds : lo = ", frac_str(t.lo), ", hi = ", frac_str(t.hi))
        if any(t.crossing)
            println(io, "\tcrossing ReLUs  : ", findall(t.crossing))
            println(io, "\t  λ (chord slope)   = ", frac_str(t.λ))
            println(io, "\t  μ (new generator) = ", frac_str(t.μ))
        else
            println(io, "\tno crossing ReLUs (λ = ", frac_str(t.λ), ")")
        end
        print(io, "\tpost-activation : "); print_zonotope(io, t.post)
    end
    return trace
end
