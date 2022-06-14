module NNPoly

using LazySets, DynamicPolynomials, RecipesBase

include("utils.jl")
include("sparse_polynomial.jl")
include("polynomial_optimization.jl")
include("approximations/chebyshev.jl")



end # module
