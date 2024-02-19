using NNPoly, NeuralVerification, BenchmarkTools, Zygote, Cthulhu
import NNPoly: aCROWN, PolyCROWN
import NeuralVerification: Network, NetworkNegPosIdx
const NP = NNPoly
const NV = NeuralVerification

acas_path = "../../vnncomp22/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx";
net = NP.read_onnx_network(acas_path, dtype=Float64)
net_npi = NV.NetworkNegPosIdx(net);

rv = NP.read_vnnlib_simple("../../vnncomp22/acasxu/vnnlib/prop_1.vnnlib", 5, 5);
specs = NP.generate_specs(rv);
input_set_acas, output_set = specs[1];

MNIST_PATH = "./eval/mnist_fc_long";
net = NP.read_onnx_network(MNIST_PATH * "/../mnist_fc/onnx/mnist-net_256x4.onnx", dtype=Float64);
net_npi = NV.NetworkNegPosIdx(net);

rv = NP.read_vnnlib_simple(MNIST_PATH * "/../mnist_fc/vnnlib/prop_3_spiral_25.vnnlib", 784, 10);
specs = NP.generate_specs(rv);
input_set_mnist, output_set = specs[1];

NP.optimise_bounds(aCROWN(), net_npi, input_set, params=NP.OptimisationParams(print_freq=1, timeout=300, n_steps=5));
t = @elapsed res = NP.optimise_bounds(aCROWN(), net_npi, input_set, params=NP.OptimisationParams(print_freq=10, timeout=300, n_steps=100))
@profview res = NP.optimise_bounds(aCROWN(), net_npi, input_set, params=NP.OptimisationParams(print_freq=10, timeout=300, n_steps=100))


## With new Flux layers
model = NP.onnx2CROWNNetwork(acas_path, dtype=Float64)
t = @elapsed res = NP.optimise_bounds(aCROWN(), model, input_set_acas, params=NP.OptimisationParams(print_freq=10, timeout=300, n_steps=100))

model = NP.onnx2CROWNNetwork(acas_path, dtype=Float64, degree=[2,1,1,1,1,1])
t = @elapsed res = NP.optimise_bounds(PolyCROWN(), model, input_set_acas, params=NP.OptimisationParams(print_freq=10, timeout=300, n_steps=10))
t = @elapsed res = NP.optimise_bounds(PolyCROWN(NP.DiffNNPolySym(common_generators=true)), model, input_set_acas, params=NP.OptimisationParams(print_freq=10, timeout=300, n_steps=1000))


## With new Flux layers
model = NP.onnx2CROWNNetwork(MNIST_PATH * "/../mnist_fc/onnx/mnist-net_256x4.onnx", dtype=Float64);
t = @elapsed res = NP.optimise_bounds(aCROWN(), model, input_set_mnist, params=NP.OptimisationParams(print_freq=10, timeout=700, n_steps=100))

# MNIST model has layers [ReLU, ReLU, ReLU, ReLU, Id]
model = NP.onnx2CROWNNetwork(MNIST_PATH * "/../mnist_fc/onnx/mnist-net_256x4.onnx", dtype=Float64, degree=[2,1,1,1]);
t = @elapsed res = NP.optimise_bounds(PolyCROWN(), model, input_set_mnist, params=NP.OptimisationParams(print_freq=10, timeout=10, n_steps=100))
t = @elapsed res = NP.optimise_bounds(PolyCROWN(NP.DiffNNPolySym(common_generators=true)), model, input_set_mnist, params=NP.OptimisationParams(print_freq=10, timeout=300, n_steps=100))

