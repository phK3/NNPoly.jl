function results = verify_spiral(instances, informat, varargin)
% verifyVnnlib - Propagate input specification through the neural network
% to compute output bounds.
%
% verifyVnnlib(instances, informat, outformat) 
% verifyVnnlib(instances, informat, outformat, logfile)
%
% instances - instances.csv file for vnnlib properties
% informat - e.g. 'BC' (for Batch, Channel) or e.g. SSC (for spatial,
% spatial, channel)
% outformat - e.g. 'BC'
% logfile - path to store results
    [varargin,logfile] = readNameValuePair(varargin,'Logfile','isstring',"logs_vnnlib.txt");
    [varargin,timeout] = readNameValuePair(varargin,'Timeout','isscalar',120);

    dirs = split(instances, '/');
    filepath = join(dirs(1:end-1), '/');

    opts = detectImportOptions(instances);
    opts.VariableNames = {'network', 'property', 'timeout'};
    opts.Delimiter = {','};
    instances = readtable(instances, opts);

    n_unfixed = -1;
    prop = -1;
    n_unfixed_old = -1;
    prop_old = -1;
    netOld = '';
    res = true;
    results = cell(size(instances, 1), 1);
    for i = 1:size(instances, 1)
        %TODO: only for the acas networks as they need vnnlib!!!
        row = instances(i,:);
        netname = row.network{1};
        % only load onnx network, if it is not already loaded from a
        % previous iteration (it's quite time intensive
        if ~strcmp(netname, netOld)
            netOld = netname;
            net = strcat(filepath, '/', netname);
            display(net)
            net = neuralNetwork.readONNXNetwork(net{1}, false, informat);
        end

        property = row.property{1};

        property = strcat(filepath, '/', property);
        property = convertCharsToStrings(property);
        display(property)

        prop_splits = split(property, ["_", "."]);
        n_unfixed_old = n_unfixed;
        n_unfixed = str2num(prop_splits(end-1));

        prop_old = prop;
        prop = str2num(prop_splits(end-3));

        if isequal(res, []) && isequal(prop_old, prop) && isequal(netOld, netname)
            % if we already weren't able to verify lower spiral, just skip
            % the larger ones
            continue
        end

        [in_spec, out_spec] = vnnlib2cora(property);

        [res, x, bounds_hist, t_hist] = verify_bounds(net, in_spec{1}, out_spec, 'Splits', 0, 'Verbose', true, 'RefinementSteps', 1000, 'Timeout', timeout);

        row = {netname, prop, n_unfixed, res, strcat("""", strjoin(string(bounds_hist)), """"), strcat("""", strjoin(string(t_hist)), """")};
        T = array2table(row);
        if i == 1
            T.Properties.VariableNames = {'network', 'property', 'n_unfixed','verified', 'bounds_hist', 't_hist'};
        end
        writetable(T, logfile, 'WriteMode', 'append');
    end

end

    