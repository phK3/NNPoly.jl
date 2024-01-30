

# Custom neural network layers based on Flux.jl s.t. we can easily incorporate learnable parameters

struct CROWNLayer{F<:NV.ActivationFunction, MN<:AbstractArray, BN<:AbstractArray, AN<:AbstractArray}
    weights::MN  # can't use Matrix{N} as we want to use a CUDAArray later on
    bias::BN  # is there a way to force the element types of weights and bias to be the same?
    activation::F
    α::AN  # optimization parameters
end


function (m::CROWNLayer{NV.ReLU, MN, BN, AN})(x) where {MN,BN,AN}
    σ = NNlib.fast_act(Flux.relu, x)
    return σ.(m.weights * x .+ m.bias)
end


# make params available for flux transformations e.g. |> gpu or |> f32
Flux.@functor CROWNLayer

# only optimisation params are trainable, we consider weights to be fixed
Flux.trainable(m::CROWNLayer) = (; m.α)


"""
Loads a simple sequential network from a given onnx file.

!!! This method assumes that every layer is fully connected with ReLU activation function
and that the last layer is fully connected with no activation function !!!

args:
    onnx_file - path to the onnx file

kwargs:
    dtype - element type of the weights and biases (defaults to Float64)

returns:
    Flux.Chain of the layers with params for trainable activations
"""
function onnx2CROWNNetwork(onnx_file; dtype=Float64)
    ws, bs = load_network(onnx_file, dtype=dtype)

    layers = []
    for (W, b) in zip(ws[1:end-1], bs[1:end-1])
        push!(layers, CROWNLayer(W, b, NV.ReLU(), similar(b)))
    end

    push!(layers, CROWNLayer(ws[end], bs[end], NV.Id(), similar(bs[end], 0)))

    # need to convert to tuple?
    return Chain(layers...)
end




