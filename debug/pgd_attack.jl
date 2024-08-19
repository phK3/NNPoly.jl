
using NNPoly, Zygote, Optim, LazySets, LinearAlgebra, Flux
const NP = NNPoly



"""
Uses values of a lower dimensional vector x to fill the entries of higher dimensional vectro y where mask is true.
"""
function expand_unfixed_inputs(x, y, mask)
    x_fill = I(length(mask))[:, mask]*x
    x̂ = ifelse.(mask, x_fill, y)
    return x̂
end


function pgd_lbfgs(model, lb, ub, y_true; verbosity=0, n_iter=100)
    mask = lb .!= ub  # unfixed entries

    lossfun = x -> begin
        x_stretch = lb[mask] .+ Flux.σ.(x) .* (ub[mask] .- lb[mask])
        xᵢₙ = expand_unfixed_inputs(x_stretch, lb, mask)
        #xᵢₙ = expand_unfixed_inputs(x, lb, mask)
        y = model(xᵢₙ)
        y[y_true]
    end

    function g!(G, x)
        ∇f = gradient(lossfun, x)
        G .= ∇f[1]
    end

    x₀ = lb[mask] .+ rand(sum(mask)) .* (ub[mask] .- lb[mask])
    #x₀ = (0.5 .* (lb .+ ub))[mask]
    #opt = Fminbox(LBFGS())
    opt = LBFGS()
    options = Optim.Options(show_trace = verbosity > 0, iterations=n_iter)
    res = optimize(lossfun, g!, x₀, opt, options)
    
    x_stretch = lb[mask] .+ Flux.σ.(Optim.minimizer(res)) .* (ub[mask] .- lb[mask])
    return expand_unfixed_inputs(x_stretch, lb, mask), res
end



mnist_path = "./eval/mnist_fc/onnx/mnist-net_256x4.onnx"
vnnlib_path = "./eval/mnist_fc/vnnlib/prop_2_spiral_49.vnnlib"

model = NP.onnx2CROWNNetwork(mnist_path, dtype=Float64)

rv = NP.read_vnnlib_simple(vnnlib_path, 784, 10);
specs = NP.generate_specs(rv);
input_set, output_set = specs[1]

A, b = tosimplehrep(output_set)
y_true = findfirst(x -> x < 0, A[1,:]);