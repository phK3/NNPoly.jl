
"""
Instance of MIPEncoder.

args:
    optimizer - MIP solver to use 
    bounds_tightening_method - (:none, :milp, :lp) which method to use for bounds tightening
    bounds_tightening_timeout - (seconds) time to spend on each min/max problem during bounds tightening
    mip_focus - Gurobi parameter: Default 0, find feasible solutions 1, prove optimality 2, improve bounds 3
"""
@with_kw struct MIPEncoder <: NV.Solver
    optimizer
    bounds_tightening_method = :milp
    bounds_tightening_timeout = 1
    mip_focus = 3
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
function encode_act(solver::MIPEncoder, L::CROWNLayer{NV.ReLU, MN, BN, AN}, model, x, idx, lbs, ubs; verbosity=0) where {MN,BN,AN}
    n = length(L.bias)

    if solver.bounds_tightening_method != :none
        fixed_inactive = ubs .<= 0
        fixed_active   = lbs .>= 0
        crossing       = trues(n) .⊻ fixed_active .⊻ fixed_inactive
        lbs, ubs = tighten_bounds(solver, model, x, lbs, ubs, method=solver.bounds_tightening_method, verbosity=verbosity)
    end

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


function encode_act(solver::MIPEncoder, L::CROWNLayer{NV.Id, MN, BN, AN}, model, x, idx, lbs, ubs; verbosity=verbosity) where {MN,BN,AN}
    return x
end


"""
Solves JuMP optimization problem to get tighter bounds on variable x.

This function internally modifies the model to carry out the optimization problems, 
but resets these changes before it returns.

args:
    model - JuMP model containing x as a variable 
    x - variables to tighten bounds for 
    lbs - current concrete lower bounds on x 
    ubs - current concrete upper bounds on x 

kwargs:
    method - currently :lp or :milp 
    timeout - timeout in seconds for optimizing each xᵢ in one direction

returns:
    lbs_opt, ubs_opt - component-wise tighter bounds of optimized and initial bounds
"""
function tighten_bounds(solver::MIPEncoder, model, x, lbs, ubs; method=:lp, timeout=1, verbosity=0, bnd_stop_lb=0, bnd_stop_ub=0)
    # save objective for restoring later 
    obj = objective_function(model)
    sense = objective_sense(model)

    # TODO: this only works when we use Gurobi
    best_bd_stop = get_attribute(model, "BestBdStop")
    mip_focus = get_attribute(model, "MIPFocus")

    if method == :lp 
        undo = relax_integrality(model)
    end

    verbosity <= 1 && set_silent(model)

    lbs_opt = fill(-Inf, size(lbs))
    ubs_opt = fill( Inf, size(ubs))
    set_time_limit_sec(model, timeout)
    # focus on improving bound
    set_attribute(model, "MIPFocus", solver.mip_focus)
    for (i, xᵢ) in enumerate(x)
        set_attribute(model, "BestBdStop", bnd_stop_lb)
        @objective(model, Min, xᵢ)
        optimize!(model)
        lbs_opt[i] = objective_bound(model)

        set_attribute(model, "BestBdStop", bnd_stop_ub)
        @objective(model, Max, xᵢ)
        optimize!(model)
        ubs_opt[i] = objective_bound(model)
    end

    if method == :lp 
        # undo LP relaxation
        undo()
    end

    set_attribute(model, "BestBdStop", best_bd_stop)
    set_attribute(model, "MIPFocus", mip_focus)
    verbosity <= 1 && unset_silent(model)
    verbosity > 0 && println("--- Avg. bound improvement: ", sum((ubs .- lbs) ./ (ubs_opt .- lbs_opt)) / length(lbs))
    @objective(model, sense, obj)

    return max.(lbs, lbs_opt), min.(ubs, ubs_opt)
end


function encode_network(solver::MIPEncoder, net::Chain, input_set, lbs, ubs; verbosity=0)
    model = Model(solver.optimizer)

    y = encode_input(solver, model, input_set)
    for (i, L) in enumerate(net.layers)
        ŷ = encode_linear(solver, L, model, y, i)
        y = encode_act(solver, L, model, ŷ, i, lbs[i], ubs[i], verbosity=verbosity)
    end

    # make output variables accessible by registering their name
    model[:y] = y
    return model
end


"""
Generate MIP encoding of a neural network given an input set represented as the feasible solutions to a JuMP model.

The JuMP model for the input set has to have a vector x_in of registered variables representing the inputs to the neural network.
I.e. input_set[:x_in] has to allow access to the inputs of the NN.

The input_set model further must not contain a registered variable y (this is reserved for the output of the NN)
"""
function encode_network(solver::MIPEncoder, net::Chain, input_set::JuMP.Model, lbs, ubs; verbosity=0)
    model = copy(input_set)
    set_optimizer(model, solver.optimizer)
    y = model[:x_in]
    for (i, L) in enumerate(net.layers)
        ŷ = encode_linear(solver, L, model, y, i)
        y = encode_act(solver, L, model, ŷ, i, lbs[i], ubs[i], verbosity=verbosity)
    end

    # make output variables accessible by registering their name
    model[:y] = y
    return model
end