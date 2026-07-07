
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN, verify_vnnlib

using Plots

# make sure that this folder contains the acas benchmarks! It should have the following structure:
# acasxu/
# ├── instances.csv
# ├── onnx/
#       ... (networks)
# └── vnnlib/
#       ... (properties)
ACAS_PATH = "../vnncomp/acasxu"

println("precompiling ...")
bcrown = PolyCROWN(bernstein_bounds=true)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(bcrown, ACAS_PATH, logfile="./eval/acas_results_bernstein_2026.jld2", max_properties=2, print_freq=1, n_steps=0, save_history=true, timeout=300)


pcrown = PolyCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(pcrown, ACAS_PATH, logfile="./eval/acas_results_polycrown_2026.jld2", max_properties=2, print_freq=1, n_steps=0, save_history=true, timeout=300)

println("running experiments ...")

println("---- PolyCROWN (Bernstein) ----")
bcrown = PolyCROWN(bernstein_bounds=true)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(bcrown, ACAS_PATH, logfile="./eval/acas_results_polycrown_bernstein_performance_run.jld2", max_properties=Inf, print_freq=1, n_steps=0, save_history=true, timeout=300)


println("---- PolyCROWN ----")
pcrown = PolyCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(pcrown, ACAS_PATH, logfile="./eval/acas_results_polycrown_performance_run_2026.jld2", max_properties=Inf, print_freq=1, n_steps=0, save_history=true, timeout=300)


# quick evaluation
berncrown = load("eval/acas_results_polycrown_bernstein_performance_run.jld2")
polycrown = load("eval/acas_results_polycrown_performance_run_2026.jld2")

y_min = min(minimum(polycrown["ys"]), minimum(berncrown["ys"]))
y_max = max(maximum(polycrown["ys"]), maximum(berncrown["ys"]))

# all instances
p_all = scatter(polycrown["ys"], berncrown["ys"], xlabel="polycrown", ylabel="berncrown", title="All instances")
plot!([y_min, y_max], [y_min, y_max], color=:black, label=nothing)
display(p_all)

# small instances
mask = polycrown["ys"] .< 10000
p_small = scatter(polycrown["ys"][mask], berncrown["ys"][mask], xlabel="polycrown", ylabel="berncrown", title="Small instances")
plot!([y_min, 10000], [y_min, 10000], color=:black, label=nothing)
display(p_small)
