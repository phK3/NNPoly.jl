
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, verify_vnnlib

ACAS_PATH = "../vnncomp2022_benchmarks/benchmarks/acasxu"

println("precompiling ...")
solver = AlphaNeurify(use_tightened_bounds=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_lin_tightened.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)

println("precompiling ...")
solver = AlphaNeurify(use_tightened_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_lin_no_tighten.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)

println("precompiling ...")
dsolver = DiffNNPolySym(truncation_terms=25, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(dsolver, ACAS_PATH, logfile="./eval/acas_results_poly.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)



println("running experiments ...")

println("---- AlphaNeurify tightened ----")
solver = AlphaNeurify(use_tightened_bounds=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_lin_tightened.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

println("---- AlphaNeurify no tighten ----")
solver = AlphaNeurify(use_tightened_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_lin_no_tighten.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

println("---- DiffNNPolySym ----")
dsolver = DiffNNPolySym(truncation_terms=25, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(dsolver, ACAS_PATH, logfile="./eval/acas_results_poly.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

