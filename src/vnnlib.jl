

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

returns:
    counterexample - or nothing, if no counterexample could be found
    all_steps - number of steps performed by verifier
    result - (String) SAT, UNSAT or inconclusive
"""
function verify_vnnlib(solver, dir; logfile=nothing, max_properties=Inf, print_freq=50, n_steps=5000,
                       only_pattern=nothing, save_history=false)
    f = CSV.File(string(dir, "/instances.csv"), header=false)

    n = length(f)
    networks = String[]
    properties = String[]
    #results = String[]
    y_starts = zeros(n)
    ys = zeros(n)
    y_hists = []
    #all_steps = zeros(Integer, n)
    times = zeros(n)

    old_netpath = nothing
    net = nothing
    net_npi = nothing
    n_in = 0
    n_out = 0

    cnt = 0

    for (i, instance) in enumerate(f)
        netpath, propertypath, time_limit = instance

        if !isnothing(only_pattern) && !contains(netpath, only_pattern)
            # just for now
            continue
        end

        if netpath != old_netpath
            println("-- loading network ", netpath)
            net = read_onnx_network(string(dir, "/", netpath), dtype=Float64)
            old_netpath = netpath

            net_npi = NV.NetworkNegPosIdx(net)

            n_in = size(net.layers[1].weights, 2)
            n_out = NV.n_nodes(net.layers[end])
        end

        rv = read_vnnlib_simple(string(dir, "/", propertypath), n_in, n_out)
        specs = generate_specs(rv)
        input_set, output_set = specs[1]  # for now just use one set (we only care about the input set anyways here)

        println("\n### Property ", propertypath, " ###\n")
        println("--- initial α ---")
        s = initialize_symbolic_domain(solver, net_npi, input_set)
        α0 = initialize_params(solver, net_npi, 2, s)
        y_start = propagate(solver, net_npi, s, α0; printing=true)

        println("--- optimisation ---")
        time = @elapsed α₁, y_hist, g_hist, d_hist, csims = optimise_bounds(solver, net_npi, input_set, print_freq=print_freq, n_steps=n_steps)

        println("\ttime = ", time)
        println("--- optimised α ---")
        propagate(solver, net_npi, s, α₁; printing=true)

        push!(networks, netpath)
        push!(properties, propertypath)
        #push!(results, result)
        times[i] = time
        y_starts[i] = y_start
        ys[i] = y_hist[end]
        save_history && push!(y_hists, y_hist)

        cnt += 1
        if cnt >= max_properties
            break
        end

        # also backup, if sth goes wrong later on
        if !isnothing(logfile)
            save(logfile, "properties", properties, "times", times, "y_starts", y_starts, "ys", ys, "y_hists", y_hists)
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
        save(logfile, "properties", properties, "times", times, "y_starts", y_starts, "ys", ys, "y_hists", y_hists)
    end

    if save_history
        return properties, times, y_starts, ys, y_hists
    else
        return properties, times, y_starts, ys
    end
end
