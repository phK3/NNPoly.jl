
@with_kw struct MIPEncoder <: NV.Solver
    optimizer
end


"""
Encodes a hyperrectangular input set by linear box constraints.

args:
    solver - the encoder to use
    model - a JuMP model to add the constraints to
    input_set - hyperrectangular input set

returns:
    x_in - the constraint variables describing the input set
"""
function encode_input(solver, model, input_set::AbstractHyperrectangle)
    @variable(model, low(input_set)[i] <= x_in[i = 1:dim(input_set)] <= high(input_set)[i])
    return x_in
end


"""
Encodes a linear layer using linear constraints.

Variables describing the output of the previous layer are required to formulate the constraints.
The index of the current layer is also required to name the constraints and variables.

args:
    solver - the encoder to use
    L - the current layer
    model - a JuMP model to add the constraints to
    x - the output variables of the previous layer
    idx - the index of the current layer
    
returns:
    y - the output variables of the linear layer
"""
function encode_linear(solver::MIPEncoder, L::CROWNLayer, model, x, idx)
    n = length(L.bias)
    y = @variable(model, [1:n], base_name="y_aff_$idx")

    W = L.weights
    b = L.bias
    con = @constraint(model, y .== W * x .+ b)
    return y
end


"""
Encodes a ReLU layer using mixed integer linear constraints.

Variables describing the output of the previous layer are required to formulate the constraints.
The index of the current layer is also required to name the constraints and variables.

args:
    solver - the encoder to use
    L - the current layer
    model - a JuMP model to add the constraints to
    x - the output variables of the previous layer
    idx - the index of the current layer
    lbs - concrete lower bounds on the ReLU inputs
    ubs - concrete upper bounds on the ReLU inputs
    
returns:
    y - the output variables of the ReLU layer
"""
function encode_act(solver::MIPEncoder, L::CROWNLayer{NV.ReLU, MN, BN, AN}, model, x, idx, lbs, ubs) where {MN,BN,AN}
    n = length(L.bias)

    fixed_inactive = ubs .<= 0
    fixed_active   = lbs .>= 0
    crossing       = trues(n) .⊻ fixed_active .⊻ fixed_inactive

    y = @variable(model, [1:n], base_name="y_relu_$idx")
    δ = @variable(model, [(1:n)[crossing]], base_name="δ_$idx", binary=true)  # TODO: make integer with bounds

    c_inactive = @constraint(model, y[fixed_inactive] .== 0, base_name="fixed_inactive_$idx")
    c_active   = @constraint(model, y[fixed_active] .== x[fixed_active], base_name="fixed_active_$idx")

    c_crossing = @constraints(model, begin
        y[crossing] .>= 0
        y[crossing] .>= x[crossing]
        y[crossing] .<= ubs[crossing] .* δ
        y[crossing] .<= x[crossing] .- lbs[crossing] .* (1 .- δ)
    end)

    return y
end


function encode_act(solver::MIPEncoder, L::CROWNLayer{NV.Id, MN, BN, AN}, model, x, idx, lbs, ubs) where {MN,BN,AN}
    return x
end


function encode_network(solver::MIPEncoder, net::Chain, input_set, lbs, ubs)
    model = Model(solver.optimizer)

    y = encode_input(solver, model, input_set)
    for (i, L) in enumerate(net.layers)
        ŷ = encode_linear(solver, L, model, y, i)
        y = encode_act(solver, L, model, ŷ, i, lbs[i], ubs[i])
    end

    # make output variables accessible by registering their name
    model[:y] = y
    return model
end