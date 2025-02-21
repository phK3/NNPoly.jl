
using NNPoly, JuMP, Gurobi
import NNPoly: aCROWN, PolyCROWN
const NP = NNPoly

#acas_path = "../../vnncomp22/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx";
acas_path = joinpath(@__DIR__, "../../vnncomp2022_benchmarks/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx");
#vnnlib_path = "../../vnncomp22/acasxu/vnnlib/prop_1.vnnlib"
vnnlib_path = joinpath(@__DIR__, "../../vnncomp2022_benchmarks/benchmarks/acasxu/vnnlib/prop_1.vnnlib")
rv = NP.read_vnnlib_simple(vnnlib_path, 5, 5);
specs = NP.generate_specs(rv);
input_set_acas, output_set = specs[1];

net_pcrown = NP.onnx2CROWNNetwork(acas_path, dtype=Float64, degree=[2,1,1,1,1,1])
t = @elapsed res, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(PolyCROWN(NP.DiffNNPolySym(common_generators=true)), net_pcrown, input_set_acas, params=NP.OptimisationParams(print_freq=25, timeout=300, n_steps=5000))


# for PolyCROWN, we first need to reconstruct the pruned network
s = NP.initialize_symbolic_domain(PolyCROWN(), net_pcrown[1:1], input_set_acas)
_, lbs, ubs, _, _, _, _, _ = NP.initialize_params_bounds(PolyCROWN(), net_pcrown, 2, s);
net_pruned, _, _ = NP.prune(NP.ZeroPruner(), net_pcrown, lbs, ubs);


# tighten formulation with just box input bounds
enc = NP.MIPEncoder(optimizer=() -> Gurobi.Optimizer(), mip_focus=3)
model_tight = NP.encode_network(enc, net_pruned, input_set_acas, lbs_pcrown, ubs_pcrown, verbosity=1)
@objective(model_tight, Max, model_tight[:y][1])
set_time_limit_sec(model_tight, 300)
optimize!(model_tight)

# define more complicated input set as JuMP model
model_input = Model()
@variable(model_input, low(input_set_acas)[i] <= x_in[i=1:dim(input_set_acas)] <= high(input_set_acas)[i])
@constraint(model_input, sum(x_in) <= 1)
@constraint(model_input, sum(x_in) >= -1)

# tighten formulation with more complicated input set 
model_inset = NP.encode_network(enc, net_pruned, model_input, lbs_pcrown, ubs_pcrown, verbosity=1)
@objective(model_inset, Max, model_inset[:y][1])
set_time_limit_sec(model_inset, 300)
optimize!(model_inset)