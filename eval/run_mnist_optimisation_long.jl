
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN, verify_vnnlib

MNIST_PATH = "./eval/mnist_fc_long"

println("precompiling ...")
solver = aCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist256x6_results_aCROWN.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=3600, force_gc=true)

println("precompiling ...")
solver = PolyCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_PolyCROWN_long.jld2",  max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=3600, force_gc=true)




println("running experiments ...")

println("aCROWN ...")
solver = aCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_aCROWN_long.jld2", max_properties=5, print_freq=10, n_steps=999999999, save_history=true, timeout=3600, force_gc=true)


println("PolyCROWN ...")
solver = PolyCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_PolyCROWN_long.jld2", max_properties=5, print_freq=5, n_steps=999999999, save_history=true, timeout=3600, force_gc=true, start_idx=2)
