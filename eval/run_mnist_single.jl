
using NNPoly, NeuralVerification, LazySets, JLD2, OnnxReader, VnnlibParser
const NP = NNPoly
const NV = NeuralVerification

acrown = NP.aCROWN()
pcrown = NP.PolyCROWN()


"""
Fill n consecutive pixels in a spiral pattern around the center of the image.

Mainly used to avoid big jumps in number of pixels in a patch (as the number of
pixels in a square patch grows quadratically). Here the number of pixels grows 
linearly.
"""
function make_spiral(img, n; patch_value=1.)
    h, w = size(img)
    h_c = Int(floor(0.5*h))
    w_c = Int(floor(0.5*w))
    img_out = copy(img)

    dir = [0, 1]
    dir_iters = 0
    dir_range = Dict([0,1] => 1, [1,0] => 1, [0,-1] => 1, [-1,0] => 1)
    next_dir = Dict([0,1] => [1,0], [1,0] => [0,-1], [0,-1] => [-1,0], [-1,0] => [0,1])
    for i in 1:n
        img_out[h_c, w_c] = patch_value
        h_c += dir[1]
        w_c += dir[2]
        dir_iters += 1
        if dir_iters == dir_range[dir]
            dir_range[.-dir] = dir_range[dir] + 1
            dir = next_dir[dir]
            dir_iters = 0
        end
    end
    
    return img_out
end            


function make_spiral_input_set(img, n, lb=0., ub=1.)
    img_low = make_spiral(img, n, patch_value=lb)
    img_up  = make_spiral(img, n, patch_value=ub)
    
    return Hyperrectangle(low=vec(img_low), high=vec(img_up))
end


function spiral_verification(netpath, propertypath; n_min=1, n_max=50)
    net = read_onnx_network(netpath, dtype=Float64)
    net_npi = NV.NetworkNegPosIdx(net)

    n_in = size(net.layers[1].weights, 2)
    n_out = NV.n_nodes(net.layers[end])

    rv = read_vnnlib_simple(propertypath, n_in, n_out)
    specs = NP.generate_specs(rv)
    input_set, output_set = specs[1]
    
    loss_lins = []
    loss_polys = []
    times_lin = []
    times_poly = []
    
    for i in n_min:n_max
        println("### $i spiral size ###")
        spiral_input = make_spiral_input_set(reshape(input_set.center, 28, 28), i)
        t_lin = @elapsed α0, lbs0, ubs0 = NP.initialize_params(acrown, net, 1, spiral_input, return_bounds=true)

        @show lbs0[end]
        @show ubs0[end]
        @show t_lin
        loss_lin = sum(ubs0[end] .- lbs0[end])

        s = NP.initialize_symbolic_domain(pcrown.poly_solver, NV.NetworkNegPosIdx(net_npi.layers[1:pcrown.poly_layers]), spiral_input)
        t_poly = @elapsed αp, lbsp, ubsp = NP.initialize_params(pcrown, net_npi, 2, s; return_bounds=true)

        @show lbsp[end]
        @show ubsp[end]
        @show t_poly
        loss_poly = sum(ubsp[end] .- lbsp[end])

        push!(loss_lins, loss_lin)
        push!(loss_polys, loss_poly)
        push!(times_lin, t_lin)
        push!(times_poly, t_poly)
    end
    
    return loss_lins, loss_polys, times_lin, times_poly
end


props = [prop for prop in readdir("../../vnncomp22/mnist_fc/vnnlib", join=true) if contains(prop, "0.05")]
nets = readdir("../../vnncomp22/mnist_fc/onnx", join=true)

loss_lins_all = Dict()
loss_polys_all = Dict()
times_lin_all = Dict()
times_poly_all = Dict()

for net in nets
    net_name = split(net, "\\")[end]
    for prop in props
        prop_name = split(prop, "\\")[end]
        
        loss_lins, loss_polys, times_lin, times_poly = spiral_verification(net, prop);
        
        loss_lins_all[(net_name, prop_name)] = loss_lins
        loss_polys_all[(net_name, prop_name)] = loss_polys
        times_lin_all[(net_name, prop_name)] = times_lin
        times_poly_all[(net_name, prop_name)] = times_poly
    end
end


save("eval/mnist_fc_all_spirals.jld2", "loss_lins_all", loss_lins_all, "loss_polys_all", 
    loss_polys_all, "times_lin_all", times_lin_all, "times_poly_all", times_poly_all)