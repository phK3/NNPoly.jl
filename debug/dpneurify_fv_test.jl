
using NNPoly, NeuralVerification
const NP = NNPoly
const NV = NeuralVerification

acas_path = joinpath(@__DIR__, "../../vnncomp/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx");
#acas_path = joinpath(@__DIR__, "../../vnncomp2022_benchmarks/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx");
vnnlib_path = joinpath(@__DIR__, "../../vnncomp/acasxu/vnnlib/prop_1.vnnlib")
#vnnlib_path = joinpath(@__DIR__, "../../vnncomp2022_benchmarks/benchmarks/acasxu/vnnlib/prop_1.vnnlib")
rv = NP.read_vnnlib_simple(vnnlib_path, 5, 5);
specs = NP.generate_specs(rv);
input_set_acas, output_set = specs[1];

net = NP.read_onnx_network(acas_path, dtype=Float64)
net_npi = NV.NetworkNegPosIdx(net);

solver = NP.DPNFV()
s = NP.init_symbolic_interval_fvheur(net_npi, input_set_acas, max_vars=10)
NV.forward_network(solver, net_npi, s)