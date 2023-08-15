
using NNPoly, NeuralVerification, LazySets, JLD2, Optimisers, Profile, PProf, OnnxReader, VnnlibParser, CSV, SparseArrays, BenchmarkTools
import NNPoly: SparsePolynomial, NNPolySym, init_poly_interval, DiffPolyInterval, DiffNNPolySym
import NeuralVerification: NetworkNegPosIdx
const NP = NNPoly
const NV = NeuralVerification

println("loading data ...")
@load "../NeuralVerification.jl/test/MNIST_1000.jld2" train_x train_y mnist_net

ϵ = 1. /255.
x = reshape(train_x[:,:,1], 28*28)
lb = x .- ϵ
ub = x .+ ϵ

input_set = Hyperrectangle(low=lb, high=ub)
mnist_npi = NetworkNegPosIdx(mnist_net)


println("precompilation ...")
dsolver = DiffNNPolySym(truncation_terms=10, common_generators=true)

s = DiffPolyInterval(mnist_npi, input_set)
α0 = NP.initialize_params(dsolver, mnist_npi, 2, s)
NP.propagate(dsolver, mnist_npi, s, α0; printing=true)


println("\nstarting experiment ...\n")
truncs = 50:25:700
times = Float64[]
ys = Float64[]

for tr in truncs
    println("### tr = ", tr)
    dsolver = DiffNNPolySym(truncation_terms=tr, common_generators=true, init=true)
    s = DiffPolyInterval(mnist_npi, input_set)
    α0 = NP.initialize_params(mnist_npi, 2, method=:zero)
    αs = NP.vec2propagation(mnist_npi, 2, α0)

    time = @elapsed y = begin
        ŝ = forward_network(dsolver, mnist_npi, s, αs)
        ll, lu = NP.bounds(ŝ.poly_interval.Low)
        ul, uu = NP.bounds(ŝ.poly_interval.Up)

        println("lbs = ", ll)
        println("ubs = ", uu)

        y = sum(uu - ll)
    end

    println("time = ", time)

    push!(times, time)
    push!(ys, y)
    # saving after each iteration in case there is an error
    save("mnist_results.jld2", "ys", ys, "times", times)
end

println("saving results ...")
println("ys = ", ys)
println("times = ", times)
save("mnist_results.jld2", "ys", ys, "times", times)

println("experiment finished")
