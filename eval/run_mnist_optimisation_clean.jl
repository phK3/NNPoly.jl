
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN, verify_vnnlib

MNIST_PATH = "./eval/mnist_fc"

println("precompiling ...")
solver = aCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_aCROWN.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300, force_gc=true)


println("precompiling ...")
solver = PolyCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_PolyCROWN.jld2", max_properties=4, print_freq=1, n_steps=10, save_history=true, timeout=300, force_gc=true)




println("running experiments ...")

println("aCROWN ...")
solver = aCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_aCROWN.jld2", max_properties=Inf, print_freq=5, n_steps=4000, save_history=true, timeout=300, force_gc=true)


println("PolyCROWN ...")
solver = PolyCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, MNIST_PATH, logfile="./eval/mnist_results_PolyCROWN.jld2", max_properties=Inf, print_freq=5, n_steps=1000, save_history=true, timeout=300, force_gc=true)

