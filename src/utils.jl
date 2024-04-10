

"""
Creates an array of all ones with the same element type and shape as A
"""
function Base.one(A::AN) where {N<:Number, AN<:AbstractArray{N}}
    return zero(A) .+ one(N)
end


"""
Creates an array of zeros with the same array type as A.

Important, if A is a CuArray as the original zeros only considers element type
which doesn't show if the original array lives on the gpu or the cpu.
"""
function Base.zeros(A::AN, dims...) where {N<:Number, AN<:AbstractArray{N}}
    return zeros(N, dims)
end

function Base.zeros(A::CN, dims...) where {N<:Number, CN<:CuArray{N}}
    return CUDA.fill(zero(N), dims)
end


"""
Transfers array a to the same device as array a_dev.

args:
    a_dev - an array already living on the desired device
    a - the array to transfer to the desired device
"""
function convert_device(a_dev::AbstractArray, a::AbstractArray)
    return a
end

function convert_device(a_dev::CuArray, a::AbstractArray)
    return cu(a)
end

## Matrix Helpers

"""
Transforms a vector of vectors into a matrix.

All vectors in the the outer vector must have the same dimension!
"""
function vecOfVec2Mat(vs)
    return Matrix(reduce(hcat, vs)')
end


"""
Transforms a vector v into a vector of vectors of size vecsize.

example:
    v = [1,2,3,4], vecsize = 2 -> [[1, 2], [3, 4]]
"""
function vec2vecOfVec(v, vecsize)
    n = length(v)
    mat = reshape(v, vecsize, Integer(n/vecsize))
    return collect(eachcol(mat))
end


"""
Creates a partial identity matrix with n rows.
All entries are zero except A[i,j] = 1, if idxs[i] = j
"""
function partial_I(n, idxs)
    m = length(idxs)
    M = zeros(n, m)
    for (i, idx) in enumerate(idxs)
        M[idx, i] = 1
    end

    return M
end


"""
Return the slope of the ReLU relaxation based on lower and upper bounds to its input.

Typesafe implementation.

args:
    l - concrete lower bound on the ReLU's input
    u - concrete upper bound on the ReLU's input

returns:
    0 if u<=0, 1 if l>=0 and u/(u-l) otherwise
"""
function relaxed_relu_gradient(l::N, u::N) where N<:Number
    u <= 0 && return zero(N)
    l >= 0 && return one(N)
    return u/(u-l)
end


"""
Computes a dictionary where the indices of occurrences in a for each value are
stored.

I.e. d[val] = [i₁, i₂, ...] with a[iⱼ] = val for each j

args:
    a - doesn't need to be an array, just needs to define keys(), which is also
        true for e.g. eachcol()
"""
function duplicates(a)
    inds = keys(a)
    eldict = Dict{eltype(a), Vector{eltype(inds)}}()
    for (val, ind) in zip(a, inds)
        if haskey(eldict, val)
            # detected other index where val is also stored
            push!(eldict[val], ind)
        else
            # val first occured at ind
            eldict[val] = [ind]
        end
    end

    return eldict
end


"""
Returns a sparse matrix S, s.t. A * S combines duplicate columns of A by summing them up.

Use e.g. `duplicates(eachcol(A))` to get duplicate_dict

args:
    A - matrix whose columns need to be deduplicated
    duplicate_dict - dictionary containing indexes of occurrence for each column in A.
"""
function compactification_matrix(A, duplicate_dict)
    row_idxs = reduce(vcat, values(duplicate_dict))
    col_idxs = reduce(vcat,[i .* ones(length(vlist)) for (i,vlist) in enumerate(values(duplicate_dict))])

    return sparse(row_idxs, col_idxs, 1, size(A, 2), length(duplicate_dict))
end


## Differentiable ComponentVector
# after https://github.com/mohamed82008/DifferentiableFactorizations.jl/blob/main/src/DifferentiableFactorizations.jl#L68

"""
Constructs a ComponentVector with components named C, l and u
"""
comp_vec_clu(C, l, u) = ComponentVector(C=C, l=l, u=u)

function ChainRulesCore.rrule(::typeof(comp_vec_clu), C, l, u)
    out = comp_vec_clu(C, l, u)
    T = typeof(out)

    function comp_vec_clu_pullback(Δ)
        _Δ = convert(T, Δ)
        return NoTangent(), _Δ.C, _Δ.l, _Δ.u
    end

    return out, comp_vec_clu_pullback
end


## Plotting Helpers

"""
Returns a closed list of the boundary vertices of a 2D zonotope.

Instead of trying all 2ⁿ combinations of error-terms, we stack together the
Generators sorted by their angle with the positive x-axis to trace the boundary
of the zonotope.
(Can't plot zonotopes with large number of generators via LazySets otherwise)

Algorithm taken from
-  https://github.com/JuliaReach/LazySets.jl/pull/2288 (LazySets issue about vertices list of 2D zonotopes) and
- https://github.com/TUMcps/CORA/blob/master/contSet/%40zonotope/polygon.m (CORA implementation for zonotope to polygon in MATLAB)
"""
function vertices_list_2d_zonotope(z::AZ) where {N, AZ<:AbstractZonotope{N}}
    c = z.center
    G = z.generators
    d, n = size(G)
    @assert d == 2 string("Only plot 2-D zonotopes!")

    # maximum in x and y direction (assuming 0-center)
    x_max = sum(abs.(G[1,:]))
    y_max = sum(abs.(G[2,:]))

    # make all generators pointing up
    Gnorm = copy(G)
    Gnorm[:, G[2,:] .< 0] .= -1 .* G[:, G[2,:] .< 0]

    # sort generators according to angle to the positive x-axis
    θ = atan.(Gnorm[2,:], Gnorm[1,:])
    θ[θ .< 0] .+= 2*π
    Gsort = Gnorm[:, sortperm(θ)]

    # get boundary of zonotope by stacking the generators together
    # first the generators pointing the most right, then up then left.
    ps = zeros(2, n+1)
    for i in 1:n
        ps[:, i+1] = ps[:, i] + 2*Gsort[:, i]
    end

    ps[1,:] .= ps[1,:] .+ x_max .- maximum(ps[1,:])
    ps[2,:] .= ps[2,:] .- y_max

    # since zonotope is centrally symmetric, we can get the left half of the
    # zonotope by mirroring the right half
    ps = [ps ps[:,end] .+ ps[:,1] .- ps[:,2:end]]

    # translate by the center of the zonotope
    ps .+= c
    return ps
end


"""
Computes a polygon enclosing each LazySet in the list.
The polygon can later be used for plotting.
"""
function _plot_list(list::AbstractVector{VN}) where {N, VN<:LazySet{N}}
    xs = Vector{N}()
    ys = Vector{N}()
    first = true
    for l in list
        x, y = LazySets.plot_recipe(l)
        if length(x) > 2
            # close polygon
            push!(x, x[1])
            push!(y, y[1])
        end

        if first
            first = false
        else
            push!(xs, N(NaN))
            push!(ys, N(NaN))
        end
        append!(xs, x)
        append!(ys, y)
    end

    return xs, ys
end


function _plot_list(list::AbstractVector{AZ}) where {N, AZ<:AbstractZonotope{N}}
    xs = Vector{N}()
    ys = Vector{N}()
    first = true
    for l in list
        ps = vertices_list_2d_zonotope(l)
        x = ps[1,:]
        y = ps[2,:]

        if first
            first = false
        else
            push!(xs, N(NaN))
            push!(ys, N(NaN))
        end
        append!(xs, x)
        append!(ys, y)
    end

    return xs, ys
end