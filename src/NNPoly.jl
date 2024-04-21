module NNPoly

using LazySets, DynamicPolynomials, RecipesBase, DataStructures, NeuralVerification,
        Parameters, LinearAlgebra, Zygote, SparseArrays, ChainRulesCore, Combinatorics,
        Optimisers, ComponentArrays, ImplicitDifferentiation, CSV, JLD2, VnnlibParser, 
        OnnxReader, Flux, JuMP
const NV = NeuralVerification

# Zygote also uses nothing for zero gradient, so need this to be defined
Base.:*(a::Nothing, x) = zero(x)
Base.:*(x, a::Nothing) = zero(x)
Base.:-(x::Nothing) = nothing  # can't really get zero(nothing)

include("utils.jl")
include("network.jl")
include("optimisation_loop.jl")

include("simplification/zero_pruner.jl")
include("simplification/merge_specification.jl")

include("polynomial_representation/sparse_polynomial.jl")
include("polynomial_representation/autodiff_speedup.jl")
include("reachability/autodiff_speedup.jl")

include("polynomial_optimization.jl")
include("approximations/chebyshev.jl")
include("approximations/crown_quadratic.jl")
include("approximations/handelman.jl")
include("approximations/shifting.jl")

include("symbolic_truncation/max_min_relaxations.jl")
include("symbolic_truncation/univariate_monomial_relaxations.jl")
include("symbolic_truncation/monomial_relaxation.jl")
include("symbolic_truncation/truncation.jl")

include("reachability/utils.jl")
include("reachability/loss_funs.jl")
include("reachability/nn_poly_zono.jl")
include("reachability/poly_interval.jl")
include("reachability/nn_poly_symb.jl")
include("reachability/symbolic_interval_diff.jl")
include("reachability/alpha_neurify.jl")
include("reachability/alpha_crown.jl")

include("reachability/diff_poly_interval.jl")
include("reachability/diff_poly_sym.jl")
include("reachability/poly_crown.jl")

include("mip.jl")

include("vnnlib.jl")

end # module
