module NNPoly

using LazySets, DynamicPolynomials, RecipesBase, DataStructures, NeuralVerification, Parameters
const NV = NeuralVerification

include("utils.jl")
include("sparse_polynomial.jl")
include("polynomial_optimization.jl")
include("approximations/chebyshev.jl")



end # module
