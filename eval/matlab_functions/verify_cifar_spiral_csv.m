function results = verify_cifar_csv(instances, informat, varargin)
% verify_cifar_csv - Propagate input specification through the neural network
% to compute output bounds.
%
% verifyVnnlib(instances, informat, outformat) 
% verifyVnnlib(instances, informat, outformat, logfile)
%
% instances - .csv file containing properties, n_unfixed, label, lbs and
% ubs for each image
% informat - e.g. 'BC' (for Batch, Channel) or e.g. SSC (for spatial,
% spatial, channel)
% outformat - e.g. 'BC'
% logfile - path to store results
    [varargin,logfile] = readNameValuePair(varargin,'Logfile','isstring',"logs_cifar.txt");
    [varargin,timeout] = readNameValuePair(varargin,'Timeout','isscalar',120);

    data = readmatrix(instances);
    props = data(:,1);
    n_unfixeds = data(:,2);
    labels = data(:,3);
    lbs = data(:,4:4+3071);
    ubs = data(:,3076:end);

    networks = ["../NNPoly.jl/eval/cifar10/onnx/cifar_relu_6_100_unnormalized.onnx", ...
                "../NNPoly.jl/eval/cifar10/onnx/cifar_relu_9_200_unnormalized.onnx"];

    n_unfixed = -1;
    prop = -1;
    n_unfixed_old = -1;
    prop_old = -1;
    netOld = '';
    res = true;
    results = cell(size(instances, 1), 1);
    
    for netname = networks
        for i = 1:size(props, 1)
            prop = props(i);     
            n_unfixed = n_unfixeds(i);
    
            % only load onnx network, if it is not already loaded from a
            % previous iteration (it's quite time intensive
            if ~strcmp(netname, netOld)
                netOld = netname;
                fprintf("loading %s", netname)
                net = neuralNetwork.readONNXNetwork(netname, false, informat);
            end
    
            fprintf("checking property %d with %d unfixed inputs", prop, n_unfixed)
    
            if isequal(res, []) && isequal(prop_old, prop) && isequal(netOld, netname)
                % if we already weren't able to verify lower spiral, just skip
                % the larger ones
                continue
            end
    
            in_spec  = aux_create_input_spec(lbs(i,:), ubs(i,:));
            out_spec = aux_create_output_spec(labels(i));
    
            [res, x, bounds_hist, t_hist] = verify_bounds(net, in_spec, out_spec, 'Splits', 0, 'Verbose', true, 'RefinementSteps', 1000, 'Timeout', timeout);
    
            row = {netname, prop, n_unfixed, res, strcat("""", strjoin(string(bounds_hist)), """"), strcat("""", strjoin(string(t_hist)), """")};
            T = array2table(row);
            if i == 1
                T.Properties.VariableNames = {'network', 'property', 'n_unfixed','verified', 'bounds_hist', 't_hist'};
            end
            writetable(T, logfile, 'WriteMode', 'append');

            prop_old = prop;
            n_unfixed_old = n_unfixed;
        end
    end

end


function X0 = aux_create_input_spec(l, u)
    X0 = interval(l',u');
end


function spec = aux_create_output_spec(label)
    A = eye(9);
    A = [A(:,1:label) -ones(9,1) A(:,label+1:end)];
    b = zeros(9,1);
    
    S = polytope(A, b);
    spec = specification(S, 'safeSet');
end

    