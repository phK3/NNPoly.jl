
using NNPoly, NeuralVerification, LazySets, JLD2, Optimisers, OnnxReader, VnnlibParser, CSV
import NNPoly: SparsePolynomial, NNPolySym, init_poly_interval, DiffPolyInterval, DiffNNPolySym
import NeuralVerification: NetworkNegPosIdx
const NP = NNPoly
const NV = NeuralVerification

println("loading data ...")

function get_acas_sets(property_number)
    if property_number == 1
        input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
        output_set = HalfSpace([1.0, 0.0, 0.0, 0.0, 0.0], 3.9911256459)
    elseif property_number == 2
        input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
        output_set = Complement(HPolytope([-1.0 1.0 0.0 0.0 0.0; -1.0 0.0 1.0 0.0 0.0; -1.0 0.0 0.0 1.0 0.0; -1.0 0.0 0.0 0.0 1.0], [0.0; 0.0; 0.0; 0.0]))
    elseif property_number == 3
        input_set = Hyperrectangle(low=[-0.3035311561, -0.0095492966, 0.4933803236, 0.3, 0.3], high=[-0.2985528119, 0.0095492966, 0.5, 0.5, 0.5])
        output_set = Complement(HPolytope([1.0 -1.0 0.0 0.0 0.0; 1.0 0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 -1.0 0.0; 1.0 0.0 0.0 0.0 -1.0], [0.0; 0.0; 0.0; 0.0]))
    elseif property_number == 4
        input_set = Hyperrectangle(low=[-0.3035311561, -0.0095492966, 0.0, 0.3181818182, 0.0833333333], high=[-0.2985528119, 0.0095492966, 0.0, 0.5, 0.1666666667])
        output_set = Complement(HPolytope([1.0 -1.0 0.0 0.0 0.0; 1.0 0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 -1.0 0.0; 1.0 0.0 0.0 0.0 -1.0], [0.0; 0.0; 0.0; 0.0]))
    else
        @assert false "Unsupported property number"
    end

    return input_set, output_set
end


println("loading data ...")
acas = read_nnet("../NeuralPriorityOptimizer.jl/networks/CAS/ACASXU_experimental_v2a_1_1.nnet")
acasNegPosIdx = NV.NetworkNegPosIdx(acas); # network where layers store their layer indices

input_set, output_set = get_acas_sets(1);


println("precompilation ...")
dsolver = DiffNNPolySym(truncation_terms=10, common_generators=true)
NP.optimise_bounds(dsolver, acasNegPosIdx, input_set, print_freq=1, n_steps=3)


println("\nstarting experiment ...\n")
truncs = 5:5:100
times = Float64[]
times_opt = Float64[]
ys = Float64[]
ys_opt = Float64[]
y_hists = []

for tr in truncs
    println("### tr = ", tr)
    dsolver = DiffNNPolySym(truncation_terms=tr, common_generators=true, init=true)
    s = DiffPolyInterval(acasNegPosIdx, input_set)
    α0 = NP.initialize_params(acasNegPosIdx, 2, method=:zero)
    αs = NP.vec2propagation(acasNegPosIdx, 2, α0)

    time = @elapsed y = begin
        ŝ = forward_network(dsolver, acasNegPosIdx, s, αs)
        ll, lu = NP.bounds(ŝ.poly_interval.Low)
        ul, uu = NP.bounds(ŝ.poly_interval.Up)

        println("lbs = ", ll)
        println("ubs = ", uu)

        y = sum(uu - ll)
    end

    println("time = ", time)


    dsolver = DiffNNPolySym(truncation_terms=tr, common_generators=true)
    time_opt = @elapsed α, y_hist, _, _, _ = NP.optimise_bounds(dsolver, acasNegPosIdx, input_set, print_freq=50, n_steps=1000)
    y_opt = minimum(y_hist)
    println("time = ", time_opt)

    push!(times_opt, time_opt)
    push!(ys_opt, y_opt)
    push!(y_hists, y_hist)

    push!(times, time)
    push!(ys, y)
    save("acas_truncation_results.jld2", "ys", ys, "times", times, "ys_opt", ys_opt, "times_opt", times_opt, "y_hists", y_hists)
end

println("saving results ...")
println("ys = ", ys)
println("ys_opt = ", ys_opt)
println("times = ", times)
println("times_opt = ", times_opt)
save("acas_truncation_results.jld2", "truncs", truncs, "ys", ys, "times", times, "ys_opt", ys_opt, "times_opt", times_opt, "y_hists", y_hists)

println("experiment finished")
