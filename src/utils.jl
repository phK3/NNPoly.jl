

## Matrix Helpers

"""
Transforms a vector of vectors into a matrix.

All vectors in the the outer vector must have the same dimension!
"""
function vecOfVec2Mat(vs)
    return Matrix(reduce(hcat, vs)')
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
