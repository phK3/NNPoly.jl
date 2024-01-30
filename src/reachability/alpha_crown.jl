
@with_kw struct aCROWN <: NV.Solver 
    # use two different α values for the same neuron (one for computation of lower
    # bound and one for computation of upper bound)
    separate_alpha::Bool = true
    # always use tightest bounds found during whole optimisation process
    use_tightened_bounds::Bool = true
    # set initial α
    initialize::Bool = false
end


"""
Linear bounding function Λx + γ representing either a linear lower or upper bound.
"""
struct SymbolicBound{N<:Number, AN<:AbstractArray{<:N, 2}, BN<:AbstractArray{<:N, 1}}
    # B(x) ≤≥ Λx + γ
    Λ::AN
    γ::BN
end


"""
Perform backsubstitution for a linear layer given linear bounds in terms of the outputs of the linear layer.

Let y = Wx + b be the output of the linear layer and z ≤ Λy + γ.
Then backsubstitution performs z ≤ Λ (Wx + b) + γ

args:
    solver - solver to use 
    L - the current linear layer
    input - the linear bounding function in terms of the output of the linear layer
"""
function backward_linear(solver::aCROWN, L::Union{NV.Layer,NV.LayerNegPosIdx,CROWNLayer}, input::SymbolicBound)
    Λ = input.Λ * L.weights
    γ = input.Λ * L.bias .+ input.γ
    return SymbolicBound(Λ, γ)
end


function initialize_slopes!(solver::aCROWN, lbs::VN, ubs::VN, α::VN) where {N<:Number,VN<:AbstractVector{N}}
    aₗ = NV.relaxed_relu_gradient_lower.(lbs, ubs)
    α .= aₗ
    return aₗ
end

"""
Perform backsubstitution for a ReLU layer given linear bounds in terms of the output of the ReLU layer.

Let y = ReLU(x) and z ≤ Λy + γ
Then backsubstitution performs z ≤ Λ⁺ relu_up(x) + Λ⁻ relu_lo(x) + γ (for upper bound on z)

args:
    solver - solver to use
    L - the current ReLU layer
    input - the linear bounding function in terms of the output of the ReLU layer
    lbs - vector of concrete lower bounds on the ReLU inputs 
    ubs - vector of concrete upper bounds on the ReLU inputs
    α - slopes of linear lower relaxations for the ReLU neurons 

kwargs:
    upper - (defaults to false) wether to compute an upper or lower bound
"""
function backward_act(solver::aCROWN, L::CROWNLayer{NV.ReLU, MN, BN, AN}, input::SymbolicBound{N}, lbs, ubs; upper=false) where {MN, BN, AN, N<:Number}
    flip = upper ? -one(N) : one(N)  # Λ⁺ and Λ⁻ are flipped for upper bound vs lower bound
    Λ⁺ = max.(flip * input.Λ, zero(N))
    Λ⁻ = min.(flip * input.Λ, zero(N))

    aₗ = if solver.initialize
        @ignore_derivatives L.α .= NV.relaxed_relu_gradient_lower.(lbs, ubs)
    else
        crossing = @ignore_derivatives (lbs .< 0) .& (ubs .> 0)
        fixed_active = @ignore_derivatives lbs .>= 0

        # need to clamp α value, since we can't use projection for whole optimisation values, when we
        # polynomially relax the first layer
        # aₗ = crossing .* clamp.(α, 0, 1) .+ fixed_active
        crossing .* clamp.(L.α, zero(N), one(N)) .+ fixed_active
    end

    aᵤ = relaxed_relu_gradient_vectorized(lbs, ubs)
    bᵤ = aᵤ .* max.(.-lbs, zero(eltype(lbs)))    

    Λ = flip * (Λ⁻ .* aᵤ' .+ Λ⁺ .* aₗ')
    γ = flip * (Λ⁻ * bᵤ) .+ input.γ

    return SymbolicBound(Λ, γ)
end


function backward_act(solver::aCROWN, L::Union{NV.Layer{NV.Id},NV.LayerNegPosIdx{NV.Id},CROWNLayer{NV.Id, MN, BN, AN}}, input::SymbolicBound, lbs, ubs; upper=false) where {MN,BN,AN}
    return input
end


"""
Performs one backward substitution pass from the last layer down to the specified layer of a neural network.

args:
    solver - the solver to use
    net - the network to perform the backsubstitution pass for 
    lbs - vector of vectors of concrete lower bounds for neurons in each layer (need bounds up to last layer - 1) 
    ubs - vector of vectors of concrete upper bounds for neurons in each layer (need bounds up to last layer - 1)
    input - input set for the verification

kwargs:
    upper - (defaults to false) whether to compute an upper or a lower bounding function of the outputs
    down_to_layer - (defaults to 1) perform backsubstitution from the last layer down to this layer
"""
function backward_network(solver, net, lbs, ubs, input; upper=false, down_to_layer=1)
    # assumes that last layer is linear!
    #Z = SymbolicBound(I, 0.)
    #Z = backward_linear(solver, net.layers[end], Z)
    Z = SymbolicBound(net.layers[end].weights, net.layers[end].bias)
    for i in reverse(down_to_layer:length(net.layers)-1)
        layer = net.layers[i]

        Ẑ = backward_act(solver, layer, Z, lbs[i], ubs[i], upper=upper)
        Z = backward_linear(solver, layer, Ẑ)
    end

    return Z
end


# @ignore_derivatives creates a closure, which led to some variables being inferred as Core.Box leading to type instability.
# therefore, we factor out the update_bounds! function and add @non_differentiable ourselves.
function update_bounds!(lbs, ubs, lbs_cur, ubs_cur, ll, uu, i)
    if length(lbs) < i
        push!(lbs, ll)
        push!(ubs, uu)
    else
        lbs[i] .= max.(lbs[i], lbs_cur[i])
        ubs[i] .= min.(ubs[i], ubs_cur[i])
        lbs_cur[i] .= lbs[i]
        ubs_cur[i] .= ubs[i]
    end  
end

@non_differentiable update_bounds!(lbs, ubs, lbs_cur, ubs_cur, ll, uu, i)


"""
Perform symbolic bound propagation using backsubstitution method for the whole network 
(i.e. execute L backsubstitution passes for the L layers of the network).

args:
    solver - solver to use
    net - network to perform bound propagation for
    input_set - Hyperrectangle of input property
    αs - vector of vectors with slopes for linear relaxation for each layer

kwargs:
    from_layer - (defaults to 1) from which layer to start backsubstitution
    lbs - (defaults to nothing) possible to use precomputed bounds
    ubs - (defaults to nothing) possible to use precomputed bounds
    printing - (defaults to false) print progress (i.e. which layer currently gets backsubstituted)

returns:
    SymbolicIntervalDiff bounding the output of the network
"""
function NV.forward_network(solver::aCROWN, net::Chain, input_set::Hyperrectangle{N}, lbs::LT, ubs::LT;
                            from_layer=1, printing=false) where {N<:Number, LT}
    lbs_cur = LT()
    ubs_cur = LT()

    # define them here, s.t. we can reference them in the return statement
    # the remainder of the code would also work if we didn't define them here at all
    Zl = SymbolicBound(net.layers[end].weights, net.layers[end].bias)
    Zu = SymbolicBound(net.layers[end].weights, net.layers[end].bias)
    # careful with from_layer and i!!!
    for (i::Int, l) in enumerate(net.layers[from_layer:end])
        if printing
            println("Layer ", from_layer + i - 1)
        end

        # If we just use Chain(...) we can directly index over that
        #nn_part = NN(net.layers[from_layer:i])

        Zl = backward_network(solver, net[from_layer:i], lbs_cur[1:i-1], ubs_cur[1:i-1], input_set)
        Zu = backward_network(solver, net[from_layer:i], lbs_cur[1:i-1], ubs_cur[1:i-1], input_set, upper=true)

        ll, lu = bounds(Zl.Λ, Zl.γ, low(input_set), high(input_set))
        ul, uu = bounds(Zu.Λ, Zu.γ, low(input_set), high(input_set))

        lbs_cur = vcat(lbs_cur, [ll])
        ubs_cur = vcat(ubs_cur, [uu])

        if solver.use_tightened_bounds
            # keep upper approach for better derivatives, need lower approach
            # for continuing with tighter bounds (update_bounds! is marked as non-differentiable)
            update_bounds!(lbs, ubs, lbs_cur, ubs_cur, ll, uu, i)
        end
    end

    return SymbolicIntervalDiff(Zl.Λ, Zl.γ, Zu.Λ, Zu.γ, low(input_set), high(input_set), lbs_cur::LT, ubs_cur::LT)
end


function initialize_params_bounds(solver::aCROWN, net, degree::N, input::Hyperrectangle) where N <: Number
    lbs = [similar(L.bias) for L in net.layers]
    ubs = [similar(L.bias) for L in net.layers]

    for i in eachindex(lbs)
        # is there any way to directly set them to ±Inf while also getting  the possibly gpu type right?
        lbs[i] .= -Inf
        ubs[i] .= Inf
    end

    isolver = aCROWN(initialize=true, separate_alpha=false)
    ŝ = NV.forward_network(isolver, net, input, lbs, ubs)

    return ŝ.lbs, ŝ.ubs
end

"""
Initialize parameters for slopes of lower ReLU relaxations.

args:
    solver - solver to use
    net - network to verify
    degree - not used here, since this is only for linear relaxation
    input - input set for verification

returns:
    vector of initial slopes
"""
function initialize_params(solver::aCROWN, net, degree::N, input::Hyperrectangle; return_bounds=false) where N <: Number
    lbs, ubs = initialize_params_bounds(solver, net, degree, input)
    
    if return_bounds
        return lbs, ubs
    end
end


"""
Initialises symbolic domain for αCROWN.
But since αCROWN's input is just a Hyperrectangle this just does nothing.
"""
function initialize_symbolic_domain(solver::aCROWN, net::NV.NetworkNegPosIdx, input::AbstractHyperrectangle)
    return input
end


"""
Perform symbolic bounds propagation via backsubstitution for the given network and return
the corresponding loss value.

We use the sum of the width of the concrete output bounds as loss.

kwargs:
    printing - (defaults to false) whether to print the concrete output bounds
    lbs - (defaults to nothing) possible to use precomputed bounds
    ubs - (defaults to nothing) possible to use precomputed bounds
"""
function propagate(solver::aCROWN, net, input::Hyperrectangle, lbs, ubs; printing=false)   
    s = NV.forward_network(solver, net, input, lbs, ubs)
    
    ll, lu = bounds(s.Λ, s.λ, s.lb, s.ub)
    ul, uu = bounds(s.Γ, s.γ, s.lb, s.ub)

    printing && println("lbs = ", ll)
    printing && println("ubs = ", uu)

    # for now, minimize range between all outputs
    loss = sum(uu - ll)
    return loss
end


# both Flux reexports Adam and OptimserChain colliding with Optimisers itself -> need to prefix with either Flux or Optimisers (which one doesn't matter)
function optimise_bounds(solver::aCROWN, net, input_set::Hyperrectangle; opt=Optimisers.OptimiserChain(Optimisers.Adam(), Projection(0., 1.)),
                         params::OptimisationParams=OptimisationParams(), print_result=false)
    lbs0, ubs0 = initialize_params_bounds(solver, net, 1, input_set)

    optfun = net -> propagate(solver, net, input_set, lbs0, ubs0)

    res = optimise(optfun, net, opt, params=params)

    print_result && propagate(solver, net, input_set, lbs0, ubs0, printing=true)

    return res
end
