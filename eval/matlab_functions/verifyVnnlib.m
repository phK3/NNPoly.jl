function results = verifyVnnlib(instances, method, informat, outformat, varargin)
% verifyVnnlib - Propagate input specification through the neural network
% to compute output bounds.
%
% verifyVnnlib(instances, method, informat, outformat) 
% verifyVnnlib(instances, method, informat, outformat, logfile)
%
% instances - instances.csv file for vnnlib properties
% method - either 'polyZono' or 'zono'
% informat - e.g. 'BC'
% outformat - e.g. 'BC'
% logfile - path to store results
    dirs = split(instances, '/');
    filepath = join(dirs(1:end-1), '/');

    opts = detectImportOptions(instances);
    opts.VariableNames = {'network', 'property', 'timeout'};
    opts.Delimiter = {','};
    instances = readtable(instances, opts);

    results = cell(size(instances, 1), 1);
    for i = 1:size(instances, 1)
        %TODO: only for the acas networks as they need vnnlib!!!
        row = instances(i,:);
        net = row.network{1};
        parts = split(net, '.');
        prefix = parts{1};
        net = strcat(filepath, '/', prefix, '_simple.onnx');
        display(net)
        net = convert_onnx_network(net{1}, informat, outformat);

        property = row.property{1};
        parts = split(property, '/');
        property = strcat(filepath, '/vnnlib_rewrite/',  parts{2});
        display(property)
        [in_spec, out_spec] = vnnlib2cora(property{1});
        if isequal(method, 'polyZono')
            box = verifyPolyZono(net, in_spec{1})
        elseif isequal(method,'zono')
            z = zonotope(in_spec{1});
            z_hat = net.evaluate(z);
            box = interval(z_hat)
        end
        results{i} = box;
    end

    if ~isempty(varargin{1})
        logfile = varargin{1};
        % write logfile
        tightness = zeros(size(instances, 1), 1);
        for i = 1:size(instances, 1)
            tightness(i) = sum(results{i}.sup - results{i}.inf);
        end

        T = array2table(tightness);
        T.Properties.VariableNames(1) = {'y'};
        writetable(T, logfile);
    end

end

    