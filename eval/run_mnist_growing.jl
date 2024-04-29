using NNPoly, JLD2, MKL, Dates
import NNPoly: PolyCROWN, verify_vnnlib
const NP = NNPoly

MNIST_PATH = "./eval/mnist_fc_growing"

println("precompiling ...")
solver = PolyCROWN(NP.DiffNNPolySym(common_generators=true))
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, "./eval/mnist_fc", logfile="./eval/mnist_results_PolyCROWN_growing.jld2",  max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=3600, force_gc=true)

current_time = Dates.now()
date_string = Dates.format(current_time, "yyyy-mm-dd_HH-MM-SS")

patience = 2
properties = 0:14
n_unfixed = 1:50
model_paths = ["./eval/mnist_fc/onnx/mnist-net_256x2.onnx", "./eval/mnist_fc/onnx/mnist-net_256x4.onnx", "./eval/mnist_fc/onnx/mnist-net_256x6.onnx"]
params = NP.OptimisationParams(n_steps=typemax(Int), timeout=60., print_freq=25, y_stop=0., save_ys=true, save_times=true, start_lr=0.1, decay=0.98)
force_gc = true
logfile_prefix = "./eval/mnist_fc_growing/"
logfile = logfile_prefix * "logs_" * date_string * ".csv"

println("running experiments ...")

open(logfile, "w") do f
    println(f, "network,property,n_unfixed,result,time,steps,hist_file")
end

for model_path in model_paths
    net = NP.onnx2CROWNNetwork(model_path, dtype=Float64, degree=1, first_layer_degree=2)

    for prop in properties
        steps_unknown = 0
        for n_un in n_unfixed
            model_name = split(model_path, "/")[end]
            println("--- net: ", model_name, " ---")
            println("\tprop: ", prop)
            println("\tunfixed iputs: ", n_un)

            vnnlib_path = "./eval/mnist_fc/vnnlib/prop_$(prop)_spiral_$(n_un).vnnlib"
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