using NNPoly, JLD2, MKL, Dates
import NNPoly: PolyCROWN, verify_vnnlib
const NP = NNPoly

CIFAR_PATH = "./eval/cifar10"

function run_cifar10_growing_lin_init_experiment(; patience=2, properties=1:100, n_unfixed=1:50)
    solver = PolyCROWN(NP.DiffNNPolySym(common_generators=true, init_method=:linear))

    current_time = Dates.now()
    date_string = Dates.format(current_time, "yyyy-mm-dd_HH-MM-SS")

    model_paths = [CIFAR_PATH * "/onnx/cifar_relu_6_100_unnormalized.onnx", CIFAR_PATH * "/onnx/cifar_relu_9_200_unnormalized.onnx"]
    params = NP.OptimisationParams(n_steps=typemax(Int), timeout=120., print_freq=25, y_stop=0., save_ys=true, save_times=true, start_lr=0.1, decay=0.98)
    force_gc = true
    logfile_prefix = "./eval/cifar10/"
    logfile = logfile_prefix * "logs_lin_init_" * date_string * ".csv"

    open(logfile, "w") do f
        println(f, "network,property,n_unfixed,result,time,steps,hist_file")
    end

    for model_path in model_paths
        net = NP.onnx2CROWNNetwork(model_path, dtype=Float64, degree=1, first_layer_degree=2, add_dummy_output_layer=true)

        for prop in properties
            steps_unknown = 0
            for n_un in n_unfixed
                model_name = split(model_path, "/")[end]
                println("--- net: ", model_name, " ---")
                println("\tprop: ", prop)
                println("\tunfixed iputs: ", n_un)

                vnnlib_path = "./eval/cifar10/vnnlib/prop_$(prop)_spiral_$(n_un).vnnlib"
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

                #save(logfile_prefix * "/$(model_name)_$(prop)_$(n_un).jld2", "result", verified, "time", t, "t_hist", res.t_hist, "y_hist", res.y_hist)
                hist_file_name = logfile_prefix * "$(model_name)_$(prop)_$(n_un)_hist_" * date_string * ".jld2"
            save(hist_file_name, "t_hist", res.t_hist, "y_hist", res.y_hist, "lbs_opt", lbs_pcrown[end], "ubs_opt", ubs_pcrown[end])

                open(logfile, "a") do f
                    println(f, string(model_name, ", ", prop, ", ", n_un, ",", verified, ", ", t, ", ", length(res.t_hist), ",", hist_file_name))
                end

                if verified == "unknown"
                    steps_unknown += 1
                    if steps_unknown >= patience
                        break
                    end
                end
            end
        end
    end
end


println("precompiling ...")
run_cifar10_growing_lin_init_experiment(patience=0, properties=1:1, n_unfixed=1:2)

println("running experiments ...")
run_cifar10_growing_lin_init_experiment(patience=2, properties=1:100, n_unfixed=1:50)
