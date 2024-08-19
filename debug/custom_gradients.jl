
using NNPoly, Zygote, ChainRulesCore
const NP = NNPoly


G = [1. 4 5; -3. 2 1]
E = [0 1 2; 0 2 1]
sp = NP.SparsePolynomial(G, E, 1:2)


testfun = (G, E) -> begin
    sp = NP.SparsePolynomial(G, E, 1:size(E, 1))
    lb, ub = NP.bounds(sp)
    return lb
end


function ChainRulesCore.rrule(::typeof(NP.bounds), sp::NP.SparsePolynomial{N,M,T,GM,EM,VI}) where {N,M,T,GM,EM,VI}
    lbs, ubs = NP.exponent_bounds(sp)

    #G⁻ = min.(zero(N), sp.G)
    #G⁺ = max.(zero(N), sp.G)
    #lb = G⁻ * ubs .+ G⁺ * lbs
    #ub = G⁻ * lbs .+ G⁺ * ubs
    # about 2x as fast, since we save a matrix vector product each time
    lb = vec(sum(sp.G .* ifelse.(sp.G .> 0, lbs', ubs'), dims=2))
    ub = vec(sum(sp.G .* ifelse.(sp.G .> 0, ubs', lbs'), dims=2))

    function bounds_pullback(Δt)
        Δlb, Δub = Δt

        Δsp = @thunk(begin
            #ΔGl = Δlb .* ifelse.(sp.G .> 0, lbs', ubs')
            #ΔGu = Δub .* ifelse.(sp.G .> 0, ubs', lbs')
            # kind of ugly, but when I write it all in one fused . notation line, it needs less memory
            Tangent{NP.SparsePolynomial}(G=Δlb .* ifelse.(sp.G .> 0, lbs', ubs') .+ Δub .* ifelse.(sp.G .> 0, ubs', lbs'), E=NoTangent(), ids=NoTangent())
        end)

        return NoTangent(), Δsp
    end

    return (lb, ub), bounds_pullback
end


function translate_new(sp::NP.SparsePolynomial{N,M,T,GM,EM,VI}, v::AbstractVector) where {N,M,T,GM,EM,VI}
    const_idx = findfirst(x -> sum(x) == 0, eachcol(sp.E))

    if isnothing(const_idx)
        Ĝ = [v sp.G]
        Ê = [zeros(M, size(sp.E, 1)) sp.E]
        ŝp = NP.SparsePolynomial(Ĝ, Ê, sp.ids)
    else
        Ĝ = copy(sp.G) #zeros(N, size(sp.G))
        Ĝ[:,const_idx] .+= v
        ŝp = NP.SparsePolynomial(Ĝ, sp.E, sp.ids)
    end

    return ŝp
end


function ChainRulesCore.rrule(::typeof(NP.translate), sp::NP.SparsePolynomial{N,M,T,GM,EM,VI}, v::AbstractVector) where {N,M,T,GM,EM,VI}
    const_idx = findfirst(x -> sum(x) == 0, eachcol(sp.E))

    if isnothing(const_idx)
        Ĝ = [v sp.G]
        Ê = [zeros(M, size(sp.E, 1)) sp.E]
        ŝp = NP.SparsePolynomial(Ĝ, Ê, sp.ids)
    else
        Ĝ = copy(sp.G) #zeros(N, size(sp.G))
        Ĝ[:,const_idx] .+= v
        ŝp = NP.SparsePolynomial(Ĝ, sp.E, sp.ids)
    end


    function translate_pullback(Δŝp)
        ΔĜ = Δŝp.G

        if isnothing(const_idx)
            # derivative is just the identity for the non-constant part and zero for the constant part (that's just the vector v)
            # other_idxs = 1:size(sp.G, 2) .!= const_idx
            Δsp = Tangent{NP.SparsePolynomial}(G=ΔĜ[:,2:end], E=NoTangent(), ids=NoTangent())
            Δv  = ΔĜ[:,1]
        else
            # we just added some values to the constant column, so just identity for all elements of G
            Δsp = Tangent{NP.SparsePolynomial}(G=ΔĜ, E=NoTangent(), ids=NoTangent())
            # derivative is just identity at column where we added the vector, zero elsewhere
            Δv  = ΔĜ[:,const_idx]
        end

        return NoTangent(), Δsp, Δv
    end

    return ŝp, translate_pullback
end