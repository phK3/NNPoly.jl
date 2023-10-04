
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN, verify_vnnlib

ACAS_PATH = "../vnncomp22/acasxu"

println("precompiling ...")
solver = AlphaNeurify(use_tightened_bounds=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_aneurify_tightened.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300, force_gc=true)


println("precompiling ...")
dsolver = DiffNNPolySym(truncation_terms=25, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(dsolver, ACAS_PATH, logfile="./eval/acas_results_alpha_poly.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300, force_gc=true)


println("precompiling ...")
acrown = aCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(acrown, ACAS_PATH, logfile="./eval/acas_results_acrown.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300, force_gc=true)


println("precompiling ...")
pcrown = PolyCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(pcrown, ACAS_PATH, logfile="./eval/acas_results_polycrown.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300, force_gc=true)



println("running experiments ...")


println("---- AlphaNeurify tightened ----")
solver = AlphaNeurify(use_tightened_bounds=true)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_aneurify_tightened.jld2", max_properties=Inf, print_freq=200, n_steps=40000, save_history=true, timeout=300, force_gc=true)


println("---- DiffNNPolySym ----")
dsolver = DiffNNPolySym(truncation_terms=25, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(dsolver, ACAS_PATH, logfile="./eval/acas_results_alpha_poly.jld2", max_properties=Inf, print_freq=50, n_steps=7000, save_history=true, timeout=300, force_gc=true)


println("---- α-CROWN ----")
acrown = aCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(acrown, ACAS_PATH, logfile="./eval/acas_results_acrown_server.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300, force_gc=true)


println("---- PolyCROWN ----")
pcrown = PolyCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(pcrown, ACAS_PATH, logfile="./eval/acas_results_polycrown_server.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300, force_gc=true)

