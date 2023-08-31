
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, verify_vnnlib

BENCHMARK_PATH = "../vnncomp2022_benchmarks/benchmarks/rl_benchmarks"
PATTERN = "dubins"
LOGFILE_POLY = "./eval/dubins_results_poly.jld2"
#LOGFILE_POLY = "./eval/dunbins_only_results_poly.jld2"
LOGFILE_LIN  = "./eval/dubins_results_lin_no_tighten.jld2"
LOGFILE_TIGHT = "./eval/dubins_results_lin_tightened.jld2"
TRUNC = 50



println("precompiling ...")
solver = AlphaNeurify(use_tightened_bounds=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, BENCHMARK_PATH, logfile=LOGFILE_TIGHT, max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)

println("precompiling ...")
solver = AlphaNeurify(use_tightened_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, BENCHMARK_PATH, logfile=LOGFILE_LIN, max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)

println("precompiling ...")
dsolver = DiffNNPolySym(truncation_terms=TRUNC, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(dsolver, BENCHMARK_PATH, logfile=LOGFILE_POLY, max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)



println("running experiments ...")

println("---- AlphaNeurify tightened ----")
solver = AlphaNeurify(use_tightened_bounds=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, BENCHMARK_PATH, logfile=LOGFILE_TIGHT, max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

println("---- AlphaNeurify no tighten ----")
solver = AlphaNeurify(use_tightened_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, BENCHMARK_PATH, logfile=LOGFILE_LIN, max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

println("---- DiffNNPolySym ----")
dsolver = DiffNNPolySym(truncation_terms=TRUNC, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(dsolver, BENCHMARK_PATH, logfile=LOGFILE_POLY, max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)


