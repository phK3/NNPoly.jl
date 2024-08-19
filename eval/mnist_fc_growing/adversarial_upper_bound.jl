
using NNPoly, CSV, JLD2, LazySets, Dates
const NP = NNPoly

current_time = Dates.now()
date_string = Dates.format(current_time, "yyyy-mm-dd_HH-MM-SS")

logfile = "./eval/mnist_fc_growing/adversarial_attacks_logs_" * date_string * ".csv"
open(logfile, "w") do f
    println(f, "network,property,n_unfixed,result")
end

last_net = nothing
last_prop = nothing
for row in CSV.File("./eval/mnist_fc_growing/logs.csv")
    if row.result == "unknown"
        net = row.network
        prop = row.property

        if net == last_net && prop == last_prop
            continue
        end

        last_net = net
        last_prop = prop

        mnist_path = "./eval/mnist_fc/onnx/" * net
        mnist = NP.onnx2CROWNNetwork(mnist_path, dtype=Float64)

        for n_un in row.n_unfixed:50
            vnnlib_path = "./eval/mnist_fc/vnnlib/prop_$(row.property)_spiral_$(n_un).vnnlib"
            rv = NP.read_vnnlib_simple(vnnlib_path, 784, 10);
            specs = NP.generate_specs(rv);
            input_set, output_set = specs[1]

            A, b = tosimplehrep(output_set)
            y_true = findfirst(x -> x < 0, A[1,:])

            x_attack, y_attack = pgd(mnist, input_set, y_true=y_true, n_restarts=10, verbosity=0, n_iter=1000)

            if y_attack != y_true
                open(logfile, "a") do f
                    println(f, string(net, ", ", prop, ", ", n_un, ",sat"))
                end

                println("######## net ", net, ", prop ", row.property, " SAT for spiral size ", n_un, " ############")
                break
            end

            println("## no counterexample found for spiral size ", n_un)

            if n_un == 50 && y_attack == y_true 
                open(logfile, "a") do f
                    println(f, string(net, ", ", prop, ", ", n_un, ",unknown"))
                end
            end
        end
    end
end
