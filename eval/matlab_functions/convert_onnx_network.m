function nnold = convert_onnx_network(onnxpath, informat, outformat)
%reads an onnx network and returns the corresponding network in
%neuralNetworkOld format
%
% onnxpath - path to onnx file
% informat - e.g. 'BC' for inputs of format (batch_dim, channel_dim) or
% 'BSSC' for ACAS (why is it S?)
% outformat - e.g. 'BC' for inputs of format (batch_dim, channel_dim)
    nn = neuralNetwork.readONNXNetwork(onnxpath, true, informat, outformat);

    cnt_lin = 1;
    cnt_act = 1;
    Ws = cell(0);
    bs = cell(0);
    actfuns = cell(0);
    for i = 1:length(nn.layers)
        layer = nn.layers{i};
        if layer.type == "LinearLayer"
            Ws{cnt_lin,1} = layer.W;
            bs{cnt_lin,1} = layer.b;
            cnt_lin = cnt_lin + 1;
        elseif layer.type == "LeakyReLULayer" && layer.alpha == 0
            actfuns{cnt_act,1} = 'ReLU';
            cnt_act = cnt_act + 1;
        else
            warning("Skipping layer of type %s", layer.type);
        end
    end

    actfuns{cnt_act} = 'identity';
    nnold = neuralNetworkOld(Ws, bs, actfuns);
end
