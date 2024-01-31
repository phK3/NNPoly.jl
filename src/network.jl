

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
    degree - a vector of relaxation degrees for every layer (except for the last one) or a 
                single scalar (then all layers have params for that relaxation degree)

returns:
    Flux.Chain of the layers with params for trainable activations
"""
function onnx2CROWNNetwork(onnx_file; dtype=Float64, degree=1)
    ws, bs = load_network(onnx_file, dtype=dtype)
    # is there a better way to expand a scalar to an array?
    degrees = typeof(degree) <: Number ? [degree for w in ws[1:end-1]] : degree

    layers = []
    for (W, b, d) in zip(ws[1:end-1], bs[1:end-1], degrees)
        if d == 1
            α = similar(b)
        else
            # n_neurons × degree × 2 (params for lower and upper bound)
            α = similar(b, length(b), d, 2)
        end

        push!(layers, CROWNLayer(W, b, NV.ReLU(), α))
    end

    push!(layers, CROWNLayer(ws[end], bs[end], NV.Id(), similar(bs[end], 0)))

    # need to convert to tuple?
    return Chain(layers...)
end




