using NNPoly
const NP = NNPoly

# need add_dummy_output_layer=true, since the onnx file ends with a ReLU layer
model = NP.onnx2CROWNNetwork("./eval/cifar10/onnx/cifar_relu_6_100_unnormalized.onnx", add_dummy_output_layer=true)

x = reshape(1:3072, 32, 32, 3, 1)  # column major -> WHCN instead of NCHW

# we removed normalization from the model, so need to do it ourself
sub_tensor = reshape([0.49140000343322754, 0.482200026512146, 0.4465000033378601], 1, 1, 3, 1);
div_tensor = reshape([0.20230001211166382, 0.19940000772476196, 0.20100000500679016], 1, 1, 3, 1);

# normalize
xᵢₙ = vec((x .- sub_tensor) ./ div_tensor)
model(xᵢₙ)

