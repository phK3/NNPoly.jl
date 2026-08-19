
using NNPoly, NeuralVerification, Flux, LazySets
const NP = NNPoly
const NV = NeuralVerification

property = :acas 
# property = :example
optimize = true 
# optimize = false
n_steps = 1000
print_freq = 100

if property == :acas
    acas_path = joinpath(@__DIR__, "../../vnncomp/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx");
    vnnlib_path = joinpath(@__DIR__, "../../vnncomp/acasxu/vnnlib/prop_1.vnnlib")
    rv = NP.read_vnnlib_simple(vnnlib_path, 5, 5);
    specs = NP.generate_specs(rv);
    input_set, output_set = specs[1];

    # prepare network in respective format for each solver
    net = NP.read_onnx_network(acas_path, dtype=Float64)
    net_npi = NV.NetworkNegPosIdx(net);
    net_crown = NP.onnx2CROWNNetwork(acas_path, dtype=Float64)
    net_pcrown = NP.onnx2CROWNNetwork(acas_path, dtype=Float64, degree=[2,1,1,1,1,1])
else 
    W1 = reshape([1.; 1], 2, 1)  # need matrix
    b1 = [0.5, -0.5]
    W2 = [1 -1.]
    b2 = [0.]

    net = NV.Network([NV.Layer(W1, b1, NV.ReLU()), NV.Layer(W2, b2, NV.Id())])
    net_npi = NV.NetworkNegPosIdx(net);

    L1 = NP.CROWNLayer(W1, b1, NV.ReLU(), zeros(2, 2, 2))
    L2 = NP.CROWNLayer(W2, b2, NV.Id(), similar(b2, 0))
    net_pcrown = Chain(L1, L2)

    L1 = NP.CROWNLayer(W1, b1, NV.ReLU(), similar(b1, length(b1), 1))
    L2 = NP.CROWNLayer(W2, b2, NV.Id(), similar(b2, 0))
    net_crown = Chain(L1, L2)

    input_set = Hyperrectangle(low=[-1.], high=[1.])
end

# compute bounds via DPNFV
println("==== DPNFV ====")
solver = NP.DPNFV()
s = NP.init_symbolic_interval_fvheur(net_npi, input_set, max_vars=10)
ŝ = NV.forward_network(solver, net_npi, s)

lbs_dpnfv, ubs_dpnfv = NP.bounds(ŝ)
println("lbs: ", lbs_dpnfv)
println("ubs: ", ubs_dpnfv)

# compute bounds via aCROWN
println("==== aCROWN ====")
params_acrown = NP.OptimisationParams(print_freq=print_freq, timeout=300, n_steps=0)
t = @elapsed res, lbs_crown, ubs_crown = NP.optimise_bounds(NP.aCROWN(), net_crown, input_set, params=params_acrown)

println("lbs: ", lbs_crown[end])
println("ubs: ", ubs_crown[end])

if optimize 
    println("==== aCROWN (opt) ====")
    params_acrown = NP.OptimisationParams(print_freq=print_freq, timeout=300, n_steps=n_steps, start_lr=0.1, decay=0.98)
    t = @elapsed res, lbs_crown, ubs_crown = NP.optimise_bounds(NP.aCROWN(), net_crown, input_set, params=params_acrown)

    println("lbs: ", lbs_crown[end])
    println("ubs: ", ubs_crown[end])
end

# compute bounds via PolyCROWN
println("==== PolyCROWN ====")
params_pcrown = NP.OptimisationParams(print_freq=print_freq, timeout=300, n_steps=0)
t = @elapsed res, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true)), net_pcrown, input_set, params=params_pcrown)

println("lbs: ", lbs_pcrown[end])
println("ubs: ", ubs_pcrown[end])

if optimize 
    println("==== PolyCROWN ====")
    params_pcrown = NP.OptimisationParams(print_freq=print_freq, timeout=300, n_steps=n_steps, start_lr=0.1, decay=0.98)
    t = @elapsed res, lbs_pcrown, ubs_pcrown = NP.optimise_bounds(NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true)), net_pcrown, input_set, params=params_pcrown)

    println("lbs: ", lbs_pcrown[end])
    println("ubs: ", ubs_pcrown[end])
end