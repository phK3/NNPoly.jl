
using NNPoly
import NNPoly: DiffNNPolySym, verify_vnnlib


println("precompiling ...")
dsolver = DiffNNPolySym(truncation_terms=25, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(dsolver, "../../vnncomp22/acasxu", logfile="./eval/acas_results.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true)

println("running experiments ...")
dsolver = DiffNNPolySym(truncation_terms=25, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(dsolver, "../../vnncomp22/acasxu", logfile="./eval/acas_results.jld2", max_properties=Inf, print_freq=50, n_steps=2000, save_history=true)
