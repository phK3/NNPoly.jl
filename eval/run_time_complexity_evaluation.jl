using NNPoly, JLD2, MKL, Dates, CSV, LazySets, LinearAlgebra
import NNPoly: PolyCROWN, verify_vnnlib
const NP = NNPoly

MNIST_PATH = "./eval/mnist_fc"

function run_time_complexity(;properties=1:5, n_unfixed=5:5:100, pruning=true)
    solver = PolyCROWN(NP.DiffNNPolySym(common_generators=true, truncation_terms = 10000), prune_neurons=pruning)  # don't truncate here
    current_time = Dates.now()
    date_string = Dates.format(current_time, "yyyy-mm-dd_HH-MM-SS")

    #properties = 1:5
    #n_unfixed = 5:5:100
    model_paths = [MNIST_PATH * "/onnx/mnist-net_256x2.onnx", MNIST_PATH * "/onnx/mnist-net_256x4.onnx", MNIST_PATH * "/onnx/mnist-net_256x6.onnx"]
    params = NP.OptimisationParams(n_steps=20, timeout=600, print_freq=5, y_stop=-Inf, save_ys=true, save_times=true, start_lr=0.1, decay=0.98)
    force_gc = true
    logfile_prefix = "./eval/mnist_fc_runtime/"
    logfile = logfile_prefix * "logs_runtime_" * date_string * ".csv"

    println("running experiments ...")

    open(logfile, "w") do f
        println(f, "network,property,n_unfixed,result,time,steps,pruning,hist_file")
    end

    for model_path in model_paths
        net = NP.onnx2CROWNNetwork(model_path, dtype=Float64, degree=1, first_layer_degree=2, add_dummy_output_layer=true)

        for prop in properties
            for n_un in n_unfixed
                model_name = split(model_path, "/")[end]
                println("--- net: ", model_name, " ---")
                println("\tprop: ", prop)
                println("\tunfixed iputs: ", n_un)

                vnnlib_path = MNIST_PATH * "/vnnlib/prop_$(prop)_spiral_$(n_un).vnnlib"
                n_in = size(net.layers[1].weights, 2)
                n_out = length(net.layers[end].bias)
                rv = NP.read_vnnlib_simple(vnnlib_path, n_in, n_out);
                specs = NP.generate_specs(rv);
                input_set, output_set = specs[1]

                net_merged = NP.merge_spec_output_layer(net, output_set)
                t = @elapsed res, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(solver, net_merged, input_set, params=params, loss_fun=NP.violation_loss, print_results=true)

                verified = NP.get_sat(lbs_pcrown[end], ubs_pcrown[end])
                println("\tresult: ", verified)
                println("\ttime: ", t)

                if force_gc
                    GC.gc()
                end

                hist_file_name = logfile_prefix * "$(model_name)_$(prop)_$(n_un)_hist_" * date_string * ".jld2"
                save(hist_file_name, "t_hist", res.t_hist, "y_hist", res.y_hist)

                open(logfile, "a") do f
                    println(f, string(model_name, ", ", prop, ", ", n_un, ",", verified, ", ", t, ", ", length(res.t_hist), ",", pruning, ",", hist_file_name))
                end

            end
        end
    end
end


println("precompiling ...")
run_time_complexity(properties=[1], n_unfixed=[5])

println("\nrunning experiments ...")
println("pruning = true")
run_time_complexity(properties=0:4, n_unfixed=5:5:100, pruning=true)

println("pruning = false")
run_time_complexity(properties=0:4, n_unfixed=5:5:100, pruning=false)