

"""
Generates specification for NeuralPriorityOptimizer from result value from vnnlib parser.

Returns a list [(input_set, output_set)], where input_set is a Hyperrectangle and
output_set is either of
- HPolytope (for a disjunctive constraint like adversarial robustness)
    -> want to use contained_within_polytope(), if it is violated, there is an adversarial example
- Complement(HPolytope) (for a conjunctive constraint)
    -> want to use reaches_polytope(), if it is satisfied, we have reached the unsafe state

Mixed disjunctive and conjunctive constraints are not fully supported!

"""
function generate_specs(rv)
    # returns list of specs
    # (input_set, output_set)
    specs = []

    if length(rv) > 1
        println("WARNING: No efficient verification for disjunction of conjunctions of input spaces implemented yet!\n
                     Creating multiple sub-problems.")
    end

    for rv_tuple in rv
        l, u, output_specs = rv_tuple

        input_set = Hyperrectangle(low=l, high=u)

        if length(output_specs) == 1
            # a single polytope
            A, b = output_specs[1]

            # spec is SAT, if we reach the polytope
            # we want to use
            output_set = Complement(HPolytope(A, b))
            push!(specs, (input_set, output_set))
        elseif all([length(b) for (A, b) in output_specs] .== ones(Integer, length(output_specs)))
            # disjunction of halfspaces
            A₁, b₁ = output_specs[1]
            Â = zeros(length(output_specs), length(A₁))
            b̂ = zeros(length(output_specs))

            for (i, (A, b)) in enumerate(output_specs)
                Â[i, :] .= vec(A)
                b̂[i] = b[1]
            end

            # we want to use contained within polytope
            # spec is SAT, if we are not contained -> violation is greater than 0
            output_set = HPolytope(-Â, -b̂)

            push!(specs, (input_set, output_set))
        else
            # disjunction of conjunction of halfspaces
            println("WARNING: No efficient verification for disjunction of conjunctions of halfspaces implemented yet!\n
                     Creating multiple sub-problems.")

            for (A, b) in output_specs
                output_set = Complement(HPolytope(A, b))
                push!(specs, (input_set, output_set))
            end
        end

    end

    return specs
end


onnx2CROWNNetwork(solver::Solver, onnx_file; dtype=Float64, degree=1, first_layer_degree=-1) = onnx2CROWNNetwork(onnx_file, dtype=dtype, degree=degree, first_layer_degree=first_layer_degree)
onnx2CROWNNetwork(solver::PolyCROWN, onnx_file; dtype=Float64, degree=1, first_layer_degree=2) = onnx2CROWNNetwork(onnx_file, dtype=dtype, degree=degree, first_layer_degree=first_layer_degree)

function get_sat(lbs, ubs)
    # TODO: do we need to consider other properties?
    if all(ubs .<= 0)
        return "unsat"
    else
        return "unknown"
    end
end



# max_properties is maximum number of properties we want to verify in this run (useful for debugging and testing)
"""
Verifies properties for network in directory with instances.csv file.

params:
    solver - solver instance to use for verification
    dir - directory containing instances.csv file with combinations of onnx networks and vnnlib properties to test
    params - parameters for solver

kwargs:
    logfile - where to store verification results
    max_properties - maximum number of instances to verify (useful for debugging, so we don't have to run all the tasks)
    split - splitting heuristic for DPNeurifyFV
    concrete_sample - sampling for concrete solutions for DPNeurifyFV
    eager - use eager Bounds checking in ZoPE
    only_pattern - only look at properties whose network_path starts with only_pattern
    timeout - timeout **per instance** in seconds
    force_gc - force garbage collection after each run
    ignore_output_property - wether to ignore the output property in the vnnlib file and just optimise the output bounds of the network instead

returns:
    counterexample - or nothing, if no counterexample could be found
    all_steps - number of steps performed by verifier
    result - (String) SAT, UNSAT or inconclusive
"""
function verify_vnnlib(solver, dir, params::OptimisationParams; logfile=nothing, max_properties=Inf, only_pattern=nothing, 
                        save_history=false, save_times=false, force_gc=false, start_idx=1, stop_idx=nothing, ignore_output_property=false,
                        loss_fun=bounds_loss)
    f = CSV.File(string(dir, "/instances.csv"), header=false)

    # need y history to get access to final loss values
    params.save_ys = true

    n = length(f)
    stop_idx = isnothing(stop_idx) ? n : stop_idx
    networks = String[]
    properties = String[]
    results = String[]
    y_starts = zeros(n)
    ys = zeros(n)
    y_hists = []
    t_hists = []
    #all_steps = zeros(Integer, n)
    times = zeros(n)

    old_netpath = nothing
    net = nothing
    net_original = nothing
    # net_npi = nothing
    n_in = 0
    n_out = 0

    cnt = 0

    for (i, instance) in enumerate(f)
        netpath, propertypath, time_limit = instance

        if !isnothing(only_pattern) && !contains(netpath, only_pattern)
            # just for now
            continue
        end
        if (i < start_idx) || (i > stop_idx)
            continue
        end

        if netpath != old_netpath
            println("-- loading network ", netpath)
            net_original = onnx2CROWNNetwork(solver, string(dir, "/", netpath), dtype=Float64)
            # net = read_onnx_network(string(dir, "/", netpath), dtype=Float64)
            old_netpath = netpath

            #net_npi = NV.NetworkNegPosIdx(net)

            n_in = size(net_original.layers[1].weights, 2)
            n_out = length(net_original.layers[end].bias)
            # n_out = NV.n_nodes(net.layers[end])
        end

        rv = read_vnnlib_simple(string(dir, "/", propertypath), n_in, n_out)
        specs = generate_specs(rv)
        input_set, output_set = specs[1]  # for now just use one set (we only care about the input set anyways here)

        if !ignore_output_property
            # merge output property constraints into the last layer
            net = merge_spec_output_layer(net_original, output_set)
        else
            net = net_original
        end

        println("\n### Property ", propertypath, " ###\n")
        #println("--- initial α ---")
        #s = initialize_symbolic_domain(solver, net_npi, input_set)
        #α0 = initialize_params(solver, net_npi, 2, s)
        #y_start = propagate(solver, net_npi, s, α0; printing=true)

        #println("--- optimisation ---")
        time = @elapsed res, lbs, ubs = optimise_bounds(solver, net, input_set, params=params, loss_fun=loss_fun)
        #α₁ = res.x_opt

        println("\ttime = ", time)
        println("--- optimised α ---")
        #propagate(solver, net_npi, s, α₁; printing=true)
        println("\tlbs = ", lbs[end])
        println("\tubs = ", ubs[end])

        push!(networks, netpath)
        push!(properties, propertypath)
        push!(results, get_sat(lbs[end], ubs[end]))
        times[i] = time
        y_starts[i] = res.y_hist[1]
        ys[i] = res.y_hist[end]
        save_history && push!(y_hists, res.y_hist)
        save_times && push!(t_hists, res.t_hist)

        cnt += 1
        if cnt >= max_properties
            break
        end

        # also backup, if sth goes wrong later on
        if !isnothing(logfile)
            save(logfile, "properties", properties, "times", times, "y_starts", y_starts, "ys", ys, "y_hists", y_hists, "t_hists", t_hists)
        end

        if force_gc
            # force garbage collection 
            GC.gc()
        end
    end

    #=if !isnothing(logfile)
        open(logfile, "w") do f
            println(f, "network,property,result,time,steps")
            [println(f, string(network, ", ", property, ", ", result, ", ", time, ", ", steps))
                    for (network, property, result, time, steps) in zip(networks, properties, results, times, all_steps)]
        end
    end=#

    println("saving results ...")
    if !isnothing(logfile)
        save(logfile, "properties", properties, "times", times, "y_starts", y_starts, "ys", ys, "y_hists", y_hists, "t_hists", t_hists)
    end

    if save_history
        return properties, times, y_starts, ys, y_hists, t_hists
    else
        return properties, times, y_starts, ys
    end
end


function verify_vnnlib(solver, dir; logfile=nothing, max_properties=Inf, print_freq=50, n_steps=5000,
    only_pattern=nothing, save_history=false, save_times=false, timeout=60., force_gc=false, start_idx=1,
    stop_idx=nothing, ignore_output_property=false, loss_fun=bounds_loss_violation_stop)
    params = OptimisationParams(n_steps=n_steps, timeout=timeout, print_freq=print_freq, y_stop= ignore_output_property ? -Inf : 0.)
    save_history && (params.save_ys = true)
    save_times && (params.save_times = true)
    return verify_vnnlib(solver, dir, params, logfile=logfile, max_properties=max_properties, only_pattern=only_pattern, 
                        save_history=save_history, save_times=save_times, force_gc=force_gc, start_idx=start_idx, stop_idx=stop_idx, 
                        ignore_output_property=ignore_output_property, loss_fun=loss_fun)
end
