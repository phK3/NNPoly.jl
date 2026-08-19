
using NNPoly, NeuralVerification, Flux, LazySets
const NP = NNPoly
const NV = NeuralVerification

function load_benchmark(property)
    if property == :acas
        acas_path = joinpath(@__DIR__, "../../vnncomp/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx");
        vnnlib_path = joinpath(@__DIR__, "../../vnncomp/acasxu/vnnlib/prop_1.vnnlib")
        rv = NP.read_vnnlib_simple(vnnlib_path, 5, 5);
        specs = NP.generate_specs(rv);
        input_set, output_set = specs[1];

        net_crown = NP.onnx2CROWNNetwork(acas_path, dtype=Float64)
    else 
        W1 = reshape([1.; 1], 2, 1)  # need matrix
        b1 = [0.5, -0.5]
        W2 = [1 -1.]
        b2 = [0.]

        L1 = NP.CROWNLayer(W1, b1, NV.ReLU(), similar(b1, length(b1), 2))
        L2 = NP.CROWNLayer(W2, b2, NV.Id(), similar(b2, 0))
        net_crown = Chain(L1, L2)

        input_set = Hyperrectangle(low=[-1.], high=[1.])
    end

    return net_crown, input_set
end


function compute_acrown_bounds(property; n_steps=20, per_output_alpha=false)
    net_crown, input_set = load_benchmark(property)

    println("==== aCROWN ====")
    params_acrown = NP.OptimisationParams(print_freq=25, timeout=300, n_steps=0)
    t = @elapsed res, lbs_crown, ubs_crown = NP.optimise_bounds(NP.aCROWN(), net_crown, input_set, params=params_acrown)

    println("lbs: ", lbs_crown[end])
    println("ubs: ", ubs_crown[end])

    
    println("==== aCROWN (opt) ====")
    params_acrown = NP.OptimisationParams(print_freq=1, timeout=300, n_steps=n_steps, start_lr=0.1, decay=0.98, save_ys=true)
    t = @elapsed res, lbs_acrown, ubs_acrown = NP.optimise_bounds(NP.aCROWN(), net_crown, input_set, params=params_acrown, loss_fun=NP.upper_bound_loss)

    println("lbs: ", lbs_acrown[end])
    println("ubs: ", ubs_acrown[end])

    return lbs_crown, ubs_crown, lbs_acrown, ubs_acrown, res
end


lbs_crown, ubs_crown, lbs_acrown, ubs_acrown, res = compute_acrown_bounds(:acas, n_steps=20, per_output_alpha=false);
# res.y_hist;