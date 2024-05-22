
evParams = struct;
evParams.num_generators = 10000;
evParams.reuse_bounds = true;

nn = neuralNetwork.readONNXNetwork("../NNPoly.jl/eval/mnist_fc/onnx/mnist-net_256x4.onnx", false, 'SSC');
lbs = zeros(784,1);
ubs = zeros(784,1);
lbs(390:394) = 0;
ubs(390:394) = 1;

X = polyZonotope(X0);
Y = nn.evaluate(X, evParams);
nn.refine(3, 'layer', 'both', X.randPoint(1), true);
Y = nn.evaluate(X, evParams);
nn.refine(3, 'layer', 'both', X.randPoint(1), true);
Y = nn.evaluate(X, evParams);

% results in the following error after the last evaluation
% Error using  + 
% Arrays have incompatible sizes for this operation.
% 
% Error in nnActivationLayer/evaluatePolyZonotope>aux_preOrderReduction (line 161)
%     id = [id; id_ + (1:q)'];
% 
% Error in nnActivationLayer/evaluatePolyZonotope (line 57)
% [c, G, GI, E, id, id_, ind, ind_] = aux_preOrderReduction(obj, c, G, GI, E, id, id_, ind, ind_, evParams);
% 
% Error in neuralNetwork/evaluate>aux_evaluatePolyZonotope (line 178)
%                 layer_i.evaluatePolyZonotope(c, G, GI, E, id, id_, ind, ind_, evParams);
% 
% Error in neuralNetwork/evaluate (line 92)
%     r = aux_evaluatePolyZonotope(obj, input, evParams, idxLayer);
% 
% Error in mwe_refine_error (line 17)
% Y = nn.evaluate(X, evParams);
% 
% Related documentation