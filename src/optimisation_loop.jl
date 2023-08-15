

"""
Projects x to be within [l, u]
Used for projected gradient descent:
    OptimiserChain()
"""
struct Projection{T} <: Optimisers.AbstractRule
    l::T
    u::T
end


function Optimisers.apply!(o::Projection, state, x, x̄)
    # in the next iteration we have x = x - x̄
    # want x - x̄ ≥ l <--> x̄ ≤ x - l
    # want x - x̄ ≤ u <--> x̄ ≥ x - u
    newx̄ = clamp.(x̄, x .- o.u, x .- o.l)
    # state is not altered
    return state, newx̄
end

# Projection doesn't need any state
Optimisers.init(o::Projection, x::AbstractArray) = nothing



"""
Gradient-based optimisation procedure for a differentiable function f.

args:
    f - function f(x) for vector input x
    opt - optimiser to use (all of Optimisers.jl optimisers are supported)
    x₀ - initial value

kwargs:
    print_freq - print every print_freq steps
    n_steps - maximum number of gradient steps
    verbosity - if > 1 also print step-length and cosine similarity of steps
    gradient_tol - if ||∇f|| < gradient_tol, stop optimisation
    patience - early stopping if the objective didn't improve in the last patience steps
"""
function optimise(f, opt, x₀; print_freq=50, n_steps=100, verbosity=1, gradient_tol=1e-5,
                  patience=50)
    x = copy(x₀)
    opt_state = Optimisers.setup(opt, x)

    x_best = x
    y_best = Inf
    steps_no_improvement = 0
    csim = 1.
    last_update = zero(x₀)
    y_hist = Float64[]
    g_hist = Float64[]
    d_hist = Float64[]
    csims = [csim]

    for i in 1:n_steps
        last_x = copy(x)
        y, g = withgradient(f, x)
        # Zygote treats nothing as zero
        ∇f = isnothing(g[1]) ? zeros(size(x)) : g[1]

        grad_norm = norm(∇f)

        push!(y_hist, y)
        push!(g_hist, grad_norm)

        if y < y_best
            y_best = y
            x_best = copy(x)
            steps_no_improvement = 0
        else
            steps_no_improvement += 1
        end

        if grad_norm < gradient_tol
            println("Optimisation converged! ||∇f|| = ", grad_norm, " < ", gradient_tol)
            break
        elseif steps_no_improvement > patience
            println("No improvement over the last ", patience, " iterations. Stopping early.")
            break
        end

        Optimisers.update!(opt_state, x, ∇f)

        if i > 1
            csim = (last_update' * (last_x .- x)) / (norm(last_update) * norm(last_x .- x))
            push!(csims, csim)
        end
        last_update = last_x .- x

        push!(d_hist, norm(last_x .- x))

        if i % print_freq == 0
            println(i, ": ", y, " - ||∇f|| = ", grad_norm)
            verbosity > 1 && println("\t ||xᵢ - xᵢ₊₁|| = ", norm(last_x .- x))
            verbosity > 1 && println("\t cos-similarity = ", csim)
        end
    end

    return x_best, y_hist, g_hist, d_hist, csims
end
