

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


function (m::CROWNLayer{NV.Id, MN, BN, AN})(x) where {MN,BN,AN}
    return m.weights * x .+ m.bias
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
    first_layer_degree - convenience argument to quickly specify degree for first layer 
                            (defaults to -1, s.t. no special case for first layer)
    add_dummy_output_layer - (bool) whether to add Identity matrix linear layer at the end 
                             (useful as some models end with a ReLU layer, which doesn't fit our framework)

returns:
    Flux.Chain of the layers with params for trainable activations
"""
function onnx2CROWNNetwork(onnx_file; dtype=Float64, degree=1, first_layer_degree=-1, add_dummy_output_layer=false)
    ws, bs = load_network(onnx_file, dtype=dtype)
    start_idx = 1
    stop_idx = add_dummy_output_layer ? length(bs) : length(bs) - 1

    # is there a better way to expand a scalar to an array?
    if first_layer_degree == -1
        degrees = typeof(degree) <: Number ? [degree for w in ws[start_idx:stop_idx]] : degree
    else
        @assert typeof(degree) <: Number "Setting first_layer_degree and a non-number arg for degree is not supported!"
        degrees = [[first_layer_degree]; [degree for w in ws[start_idx+1:stop_idx]]]
    end

    layers = []
    for (W, b, d) in zip(ws[start_idx:stop_idx], bs[start_idx:stop_idx], degrees)
        if d == 1
            α = similar(b)
        else
            # n_neurons × degree × 2 (params for lower and upper bound)
            α = similar(b, length(b), d, 2)
        end

        push!(layers, CROWNLayer(W, b, NV.ReLU(), α))
    end

    if add_dummy_output_layer
        n_out = length(bs[end])
        W = I(n_out)
        b = zero(bs[end])
        push!(layers, CROWNLayer(W, b, NV.Id(), similar(bs[end], 0)))
    else
        push!(layers, CROWNLayer(ws[end], bs[end], NV.Id(), similar(bs[end], 0)))
    end

    # need to convert to tuple?
    return Chain(layers...)
end




