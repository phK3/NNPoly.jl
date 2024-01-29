

"""
Compute an n+1 square matrix containing the binomial coefficients for 0:n

i.e. C[i+1,j+1] = binomial(i,j), 0 ≤ i,j ≤ n
**Julia is 1-indexed!!!**
"""
function binomial_coefficients(n)
    #C = ones(n+1, n+1)
    C = zeros(n+1, n+1)
    C[1, 1] = 1
    C[2, 1] = 1
    C[2, 2] = 1
    
    for i in 2:n
        C[i+1,i+1] = 1
        C[i+1, 1] = 1
        
        for k in 1:i-1
            C[i+1, k+1] = i/(i - k) * C[i, k+1]
        end
    end
    
    return C
end


function inverse_ux(n, C)
    U = zeros(n+1, n+1)
    U[1:n+1,    1] .= 1
    U[n+1,  2:n+1] .= 1
    
    for i in 1:n-1
        for j in 1:i
            U[i+1,j+1] = C[i+1, i-j+1] / C[n+1, j+1]
        end
    end
    
    return U
end


function inverse_vx(n, l::N, u::N) where N<:Number
    w = u - l
    ws = zeros(n+1)
    ws[1] = 1
    for i in 1:n
        ws[i+1] = ws[i] * w  # width^i
    end
    
    return Diagonal(ws)
end


function inverse_wx(n, l, u, C)
    infs = zeros(n+1)
    infs[1] = 1
    for i in 1:n
        infs[i+1] = infs[i] * l
    end
    
    W = zeros(n+1, n+1)
    W[1, 1] = 1
    for i in 0:n-1
        for j in i+1:n
            W[i+1, j+1] = C[j+1, i+1] * infs[j-i+1]
        end
        W[i+2, i+2] = 1
    end
    
    return W
end


"""
Calculate all coefficients of the Bernstein expansion of a given polynomial.

If no specific lower and upper bounds for the variables are given, we assume them to be 
normalized to xᵢ ∈ [-1, 1] as is customary for our SparsePolynomials.

We use the algorithm of 
    S. Ray and P. Nataraj, A Matrix Method for Efficient Computation of Bernstein Coefficients (2012)
    available at https://interval.louisiana.edu/reliable-computing-journal/volume-17/reliable-computing-17-pp-40-71.pdf
Which is also described in
    J. Titi and J. Garloff, Matrix Methods for the Tensorial Bernstein Form (2019)
    available at http://www-home.htwg-konstanz.de/~garloff/AMC-2.pdf

Time complexity: O(n^(l+1)) 
    - n: degree of the polynomial
    - l: number of variables

args:
    sp - (SparsePolynomial) polynomial to be expanded in Bernstein basis

kwargs:
    lbs - (vector) lower bounds for the variables
    ubs - (vector) upper bounds for the variables

returns:
    B - tensor d₁ × ... × dₗ × h where B[i₁, ..., iₗ, k] returns the i₁,...,iₗ-th Bernstein
        coefficient of the k-th input polynomial
"""
function calculate_bernstein_coeffs(sp::SparsePolynomial; lbs=nothing, ubs=nothing)
    n = size(sp.E, 1)  # number of variables
    d_max = maximum(sp.E)  # maximum degree of the polynomial
    h = size(sp.G, 1)  # number of polynomials (rows of the sparse polynomial)

    lbs = isnothing(lbs) ? -ones(n) : lbs
    ubs = isnothing(ubs) ? ones(n) : ubs
    
    # if the method is used many times, precalculate the binomial coefficients and store them somewhere
    C = binomial_coefficients(d_max)
    U = inverse_ux(d_max, C)
    
    # if lbs and ubs are the same for all xᵢ, only need to compute one M
    Ms = []
    for r in 1:n
        # save computation if bounds are the same
        if r == 1 ||lbs[r] != lbs[1] || ubs[r] != ubs[1]
            V = inverse_vx(d_max, lbs[r], ubs[r])
            W = inverse_wx(d_max, lbs[r], ubs[r], C)
            push!(Ms, U * V * W)
        else
            push!(Ms, copy(Ms[1]))
        end
    end
    
    # convert sparse polynomial to dense polynomial tensor form
    # tensor with one dimension for each variable
    a_size = [d_max+1 for i in 1:n]
    A = zeros([a_size; h]...)

    for i in 1:size(sp.E, 2)
        ind = CartesianIndex(Tuple(sp.E[:,i] .+ 1))
        A[ind, :] .= sp.G[:,i]
    end
    
    # calculate bernstein coeffs
    cyclic_reverse_permutation = [collect(2:n)...; 1]
    for r in 1:n
        A = reshape(A, d_max+1, :, h)
        # there should be some tensor operation instead of this loop!
        for j in 1:h
            A[:,:,j] = Ms[r] * A[:,:,j]
        end
        A = reshape(A, [a_size; h]...)
        # need [...; n+1] so that entries for different polynomials don't get cycled
        A = permutedims(A, [cyclic_reverse_permutation...; n+1])
    end
    
    return A
end


function bernstein_bounds(sp::SparsePolynomial)
    B = calculate_bernstein_coeffs(sp)
    n_vars = size(sp.E, 1)
    lbs = vec(minimum(B, dims=(1:n_vars)))
    ubs = vec(maximum(B, dims=(1:n_vars)))
    
    return lbs, ubs
end