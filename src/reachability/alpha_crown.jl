
@with_kw struct aCROWN <: NV.Solver 
    # use two different α values for the same neuron (one for computation of lower
    # bound and one for computation of upper bound)
    separate_alpha = true
    # always use tightest bounds found during whole optimisation process
    use_tightened_bounds = true
    # set initial α
    initialize = false
end


"""
Linear bounding function Λx + γ representing either a linear lower or upper bound.
"""
struct SymbolicBound
    # B(x) ≤≥ Λx + γ
    Λ::Union{Matrix{Float64}, UniformScaling{Bool}}  # UniformScaling for I
    γ::Union{Vector{Float64}, Float64}  # Float64 for 0.
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
function backward_linear(solver::aCROWN, L::Union{NV.Layer,NV.LayerNegPosIdx}, input::SymbolicBound)
    Λ = input.Λ * L.weights
    γ = input.Λ * L.bias .+ input.γ
    return SymbolicBound(Λ, γ)
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
function backward_act(solver::aCROWN, L::Union{NV.Layer{NV.ReLU},NV.LayerNegPosIdx{NV.ReLU}}, input::SymbolicBound, lbs, ubs, α; upper=false)
    flip = upper ? -1. : 1.  # Λ⁺ and Λ⁻ are flipped for upper bound vs lower bound
    Λ⁺ = max.(flip * input.Λ, 0)
    Λ⁻ = min.(flip * input.Λ, 0)

    if solver.initialize
        aₗ = ChainRulesCore.ignore_derivatives() do
            aₗ = NV.relaxed_relu_gradient_lower.(lbs, ubs)
            α .= aₗ
            aₗ
        end
    else
        crossing = @ignore_derivatives (lbs .< 0) .& (ubs .> 0)
        fixed_active = @ignore_derivatives lbs .>= 0

        # need to clamp α value, since we can't use projection for whole optimisation values, when we
        # polynomially relax the first layer
        aₗ = crossing .* clamp.(α, 0, 1) .+ fixed_active
    end

    # maybe also transform to matrix operation w/o list comprehension
    # but beware of divide by zero!!!
    aᵤ = [NV.relaxed_relu_gradient(lᵢ, uᵢ) for (lᵢ, uᵢ) in zip(lbs, ubs)]
    bᵤ = aᵤ .* max.(-lbs, 0)    

    Λ = flip * (Λ⁻ .* aᵤ' + Λ⁺ .* aₗ')
    γ = flip * (Λ⁻ * bᵤ) .+ input.γ

    return SymbolicBound(Λ, γ)
end


function backward_act(solver::aCROWN, L::Union{NV.Layer{NV.Id},NV.LayerNegPosIdx{NV.Id}}, input::SymbolicBound, lbs, ubs; upper=false)
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
    αs - vector of vectors with slopes for the linear relaxations of the activation functions for each layer

kwargs:
    upper - (defaults to false) whether to compute an upper or a lower bounding function of the outputs
    down_to_layer - (defaults to 1) perform backsubstitution from the last layer down to this layer
"""
function backward_network(solver, net, lbs, ubs, input, αs; upper=false, down_to_layer=1)
    # assumes that last layer is linear!
    Z = SymbolicBound(I, 0.)
    Z = backward_linear(solver, net.layers[end], Z)
    for i in reverse(down_to_layer:length(net.layers)-1)
        layer = net.layers[i]

        Ẑ = backward_act(solver, layer, Z, lbs[i], ubs[i], αs[i], upper=upper)
        Z = backward_linear(solver, layer, Ẑ)
    end

    return Z
end


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
function NV.forward_network(solver::aCROWN, net::NN, input_set::Hyperrectangle, αs; 
                            from_layer=1, lbs=nothing, ubs=nothing, printing=false) where NN<:Union{NV.Network, NV.NetworkNegPosIdx}
    best_lbs = isnothing(lbs) ? [] : lbs
    best_ubs = isnothing(ubs) ? [] : ubs
    lbs = []
    ubs = []
    
    if solver.separate_alpha
        half = Int(floor(0.5*length(αs)))
        αsₗ = αs[1:half]
        αsᵤ = αs[half+1:end]
    else
        αsₗ = αs
        αsᵤ = αs
    end

    # define them here, s.t. we can reference them in the return statement
    # the remainder of the code would also work if we didn't define them here at all
    Zl = SymbolicBound(I, 0.)
    Zu = SymbolicBound(I, 0.)
    # careful with from_layer and i!!!
    for (i, l) in enumerate(net.layers[from_layer:end])
        if printing
            println("Layer ", from_layer + i-1)
        end

        nn_part = NN(net.layers[from_layer:i])
        
        Zl = backward_network(solver, nn_part, lbs[1:i-1], ubs[1:i-1], input_set, αsₗ[1:i])
        Zu = backward_network(solver, nn_part, lbs[1:i-1], ubs[1:i-1], input_set, αsᵤ[1:i], upper=true)

        ll, lu = bounds(Zl.Λ, Zl.γ, low(input_set), high(input_set))
        ul, uu = bounds(Zu.Λ, Zu.γ, low(input_set), high(input_set))
        
        lbs = vcat(lbs, [ll])
        ubs = vcat(ubs, [uu])
        
        if solver.use_tightened_bounds
            # keep upper approach for better derivatives, need lower approach
            # for continuing with tighter bounds
            ChainRulesCore.ignore_derivatives() do
                if length(best_lbs) < i
                    push!(best_lbs, ll)
                    push!(best_ubs, uu)
                else
                    best_lb = max.(lbs[i], best_lbs[i])
                    best_ub = min.(ubs[i], best_ubs[i])
                    best_lbs[i] .= best_lb
                    best_ubs[i] .= best_ub
                    lbs[i] = best_lb
                    ubs[i] = best_ub
                end
            end
        end
    end

    return SymbolicIntervalDiff(Zl.Λ, Zl.γ, Zu.Λ, Zu.γ, low(input_set), high(input_set), lbs, ubs)
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
    n_neurons = sum(length(l.bias) for l in net.layers)
    α = zeros(n_neurons)
    αs = vec2propagation(net, α)
    # for initialization separate_alpha isn't needed
    isolver = aCROWN(initialize=true, separate_alpha=false)
    ŝ = NV.forward_network(isolver, net, input, αs)

    α0 = reduce(vcat, vec.(αs))
    if solver.separate_alpha
        α0 = [α0; α0]
    end
    
    if return_bounds
        return α0, ŝ.lbs, ŝ.ubs
    else
        return α0
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
function propagate(solver::aCROWN, net::NN, input::Hyperrectangle, α; printing=false, lbs=nothing, ubs=nothing) where NN<:Union{NV.Network, NV.NetworkNegPosIdx}
    if solver.separate_alpha
        @assert length(α) % 2 == 0 "If solver.separate_alpha, then only even lengths of α are allowed."
        half = Int(0.5*length(α))
        αs = [vec2propagation(net, α[1:half]); vec2propagation(net, α[half+1:end])]
    else
        αs = vec2propagation(net, α)
    end
    
    s = NV.forward_network(solver, net, input, αs, lbs=lbs, ubs=ubs)
    
    ll, lu = bounds(s.Λ, s.λ, s.lb, s.ub)
    ul, uu = bounds(s.Γ, s.γ, s.lb, s.ub)

    printing && println("lbs = ", ll)
    printing && println("ubs = ", uu)

    # for now, minimize range between all outputs
    loss = sum(uu - ll)
    return loss
end


function optimise_bounds(solver::aCROWN, net::NN, input_set::Hyperrectangle; opt=nothing,
                         params=OptimisationParams(), print_result=false) where NN<:Union{NV.Network, NV.NetworkNegPosIdx}
    opt = isnothing(opt) ? OptimiserChain(Adam(), Projection(0., 1.)) : opt

    α0, lbs0, ubs0 = initialize_params(solver, net, 1, input_set, return_bounds=true)

    if solver.use_tightened_bounds
        optfun = α -> propagate(solver, net, input_set, α, lbs=lbs0, ubs=ubs0)
    else
        optfun = α -> propagate(solver, net, input_set, α)
    end


    res = optimise(optfun, opt, α0, params=params)

    print_result && propagate(solver, net, input_set, res.x_opt, lbs=lbs0, ubs=ubs0, printing=true)

    return res
end
