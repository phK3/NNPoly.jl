module NNPoly

using LazySets, DynamicPolynomials, RecipesBase, DataStructures, NeuralVerification, Parameters, LinearAlgebra
const NV = NeuralVerification

include("utils.jl")
include("sparse_polynomial.jl")
include("polynomial_optimization.jl")
include("approximations/chebyshev.jl")
include("reachability/nn_poly_zono.jl")
include("reachability/poly_interval.jl")
include("reachability/nn_poly_symb.jl")




end # module
