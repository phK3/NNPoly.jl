
using Revise, NNPoly, CUDA, LazySets, Adapt, Flux
const NP = NNPoly

MNIST_PATH = "./eval/mnist_fc/"
net = NP.onnx2CROWNNetwork(MNIST_PATH * "onnx/mnist-net_256x4.onnx", dtype=Float64, degree=[2,1,1,1])

rv = NP.read_vnnlib_simple(MNIST_PATH * "vnnlib/prop_3_spiral_25.vnnlib", 784, 10);
specs = NP.generate_specs(rv);
input_set, output_set = specs[1];

function Adapt.adapt_structure(to, sp::NP.SparsePolynomial)
    G = Adapt.adapt_structure(to, sp.G)
    E = Adapt.adapt_structure(to, sp.E)
    ids = Adapt.adapt_structure(to, sp.ids)
    NP.SparsePolynomial(G, E, ids)
end


Adapt.@adapt_structure NP.PolyInterval


function Adapt.adapt_structure(to, dfp::NP.DiffPolyInterval)
    poly_interval = Adapt.adapt_structure(to, dfp.poly_interval)
    lbs = [Adapt.adapt_structure(to, lb) for lb in dfp.lbs]  # only convert inner vectors to gpu
    ubs = [Adapt.adapt_structure(to, ub) for ub in dfp.ubs]
    NP.DiffPolyInterval(poly_interval, lbs, ubs)
end


function calculate_minimizer_quad(C, l, u)
    # if we divide by zero, x_opt will just be +/- inf
    x_opt = -0.5 .* C[:,2] ./ C[:,3]

    X = [one(x_opt) x_opt x_opt.^2]
    L = [one(l) l l.^2]
    U = [one(u) u u.^2]

    Yx = sum(A .* X, dims=2)
    Yl = sum(A .* L, dims=2)
    Yu = sum(A .* U, dims=2)

    # only consider x_opt if it is within bounds
    x_mask = (l .<= x_opt) .& (x_opt .<= u)
    # if x_opt valid and smaller than Yl and Yu, return x_opt, else the smaller of Yl and Yu
    x_min = ifelse.(x_mask .& (Yx .<= Yl) .& (Yx .<= Yu), 
                    x_opt, 
                    ifelse.(Yl .<= Yu, l, u)) 
    return x_min
end


# initialize an input set
solver = NP.PolyCROWN()
s = NP.initialize_symbolic_domain(solver, net[1:solver.poly_layers], input_set)

# transfer the input set to the gpu
scu = s |> gpu

# transfer the net to the gpu
netcu = net |> gpu

ŝcu, lbscu, ubscu, rscu, cscu, symmetric_factor_cu, unique_idxs_cu, duplicate_idxs_cu = NP.initialize_params_bounds(solver, netcu, 2, scu);

ŝ, lbs, ubs, rs, cs, symmetric_factor, unique_idxs, duplicate_idxs = NP.initialize_params_bounds(solver, net, 2, s);