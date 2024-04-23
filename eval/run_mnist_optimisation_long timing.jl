
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN, verify_vnnlib, time_single_pass

MNIST_PATH = "./eval/mnist_fc_long"

println("precompiling ...")
solver = aCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist256x6_results_aCROWN_timing.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=3600, force_gc=true)

println("precompiling ...")
solver = PolyCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_PolyCROWN_long_timing.jld2",  max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=3600, force_gc=true)




println("running experiments ...")

println("aCROWN ...")
solver = aCROWN()
properties_acrown, times_acrown, y_starts_acrown, ys_acrown, y_hists_acrown = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_aCROWN_long_timing.jld2", max_properties=5, print_freq=10, n_steps=100, save_history=true, timeout=120, force_gc=true)


println("PolyCROWN ...")
solver = PolyCROWN()
properties_poly, times_poly, y_starts_poly, ys_poly, y_hists_poly = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_PolyCROWN_long_timing.jld2", max_properties=5, print_freq=5, n_steps=100, save_history=true, timeout=120, force_gc=true)



# get calculate iteration speed

function get_avg_times(times, hists)
    iters_per_sec = []
    sec_per_iter = []
    for (t, h) in zip(times, hists)
        push!(iters_per_sec,length(h) / t)
        push!(sec_per_iter, t / length(h))
    end
    
    return iters_per_sec, sec_per_iter
end

iters_per_sec_acrown, sec_per_iter_acrown = get_avg_times(times_acrown, y_hists_acrown)
iters_per_sec_poly, sec_per_iter_poly = get_avg_times(times_poly, y_hists_poly)


# single pass experiments

println("running single pass experiments ...")

println("aCROWN ...")
solver = aCROWN()
properties_acrown, times_acrown, n_iters_acrown = time_single_pass(solver, MNIST_PATH, logfile="./eval/mnist_results_aCROWN_long_single_timing.jld2", max_properties=5, save_history=true, n_rep=50, timeout=10., force_gc=true)

println("PolyCROWN ...")
solver = PolyCROWN()
properties_poly, times_poly, n_iters_poly = time_single_pass(solver, MNIST_PATH, logfile="./eval/mnist_results_PolyCROWN_long_single_timing.jld2", max_properties=5, save_history=true, n_rep=50, timeout=20., force_gc=true)



