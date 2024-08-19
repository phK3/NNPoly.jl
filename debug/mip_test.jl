using NNPoly, JuMP, Gurobi
import NNPoly: aCROWN, PolyCROWN
const NP = NNPoly

acas_path = "../../vnncomp22/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx";
rv = NP.read_vnnlib_simple("../../vnncomp22/acasxu/vnnlib/prop_1.vnnlib", 5, 5);
specs = NP.generate_specs(rv);
input_set_acas, output_set = specs[1];


## Calculate bounds on the output and intermediate neurons using aCROWN and PolyCROWN
net_crown = NP.onnx2CROWNNetwork(acas_path, dtype=Float64)
t = @elapsed res, lbs_crown, ubs_crown = NP.optimise_bounds(aCROWN(), net_crown, input_set_acas, params=NP.OptimisationParams(print_freq=25, timeout=300, n_steps=2000))

net_pcrown = NP.onnx2CROWNNetwork(acas_path, dtype=Float64, degree=[2,1,1,1,1,1])
t = @elapsed res, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(PolyCROWN(NP.DiffNNPolySym(common_generators=true)), net_pcrown, input_set_acas, params=NP.OptimisationParams(print_freq=25, timeout=300, n_steps=5000))


## Use MIP solver with these bounds
enc = NP.MIPEncoder(() -> Gurobi.Optimizer())

model_crown = NP.encode_network(enc, net_crown, input_set_acas, lbs_crown, ubs_crown)
@objective(model_crown, Max, model_crown[:y][1])
set_time_limit_sec(model_crown, 300)
optimize!(model_crown)


# for PolyCROWN, we first need to reconstruct the pruned network
s = NP.initialize_symbolic_domain(PolyCROWN(), net_pcrown[1:1], input_set_acas)
_, lbs, ubs, _, _, _, _, _ = NP.initialize_params_bounds(PolyCROWN(), net_pcrown, 2, s);
net_pruned, _, _ = NP.prune(NP.ZeroPruner(), net_pcrown, lbs, ubs);

model_pcrown = NP.encode_network(enc, net_pruned, input_set_acas, lbs_pcrown, ubs_pcrown)
@objective(model_pcrown, Max, model_pcrown[:y][1])
set_time_limit_sec(model_pcrown, 300)
optimize!(model_pcrown)


