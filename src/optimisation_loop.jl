
@with_kw mutable struct OptimisationParams
    n_steps=100
    # timeout in seconds
    timeout=60.
    # number of iterations w/o improvement before early stopping
    patience=50
    gradient_tol=1e-5
    # console output
    verbosity=1
    print_freq=50
    # save history of loss values
    save_ys = false
    # save history of gradient norms
    save_gs = false
    # saving history of times for plotting and analysis
    save_times = false
    # saving history of distance between iterates
    save_dist = false
    # save history of cosine similarity between iterates
    save_cosine_similarity = false
end



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
function optimise(f, opt, x₀; params=OptimisationParams())
    t_start = time()

    x = copy(x₀)
    opt_state = Optimisers.setup(opt, x)

    x_best = x
    y_best = Inf
    steps_no_improvement = 0
    csim = 1.
    last_update = zero(x₀)
    y_hist = Float64[]
    t_hist = Float64[]
    g_hist = Float64[]
    d_hist = Float64[]
    csims = params.save_cosine_similarity ? [csim] : Float64[]

    for i in 1:params.n_steps
        last_x = copy(x)
        y, g = withgradient(f, x)
        # Zygote treats nothing as zero
        ∇f = isnothing(g[1]) ? zeros(size(x)) : g[1]

        grad_norm = norm(∇f)

        params.save_ys && push!(y_hist, y)
        params.save_gs && push!(g_hist, grad_norm)

        if y < y_best
            y_best = y
            x_best = copy(x)
            steps_no_improvement = 0
        else
            steps_no_improvement += 1
        end

        t_now = time()
        elapsed_time = t_now - t_start
        params.save_times && push!(t_hist, elapsed_time)
        if grad_norm < params.gradient_tol
            println("Optimisation converged! ||∇f|| = ", grad_norm, " < ", params.gradient_tol)
            break
        elseif steps_no_improvement > params.patience
            println("No improvement over the last ", params.patience, " iterations. Stopping early.")
            break
        elseif elapsed_time >= params.timeout
            println("Timeout reached (", elapsed_time, " elapsed)")
            break
        end


        Optimisers.update!(opt_state, x, ∇f)

        if i > 1 && params.save_cosine_similarity
            csim = (last_update' * (last_x .- x)) / (norm(last_update) * norm(last_x .- x))
            push!(csims, csim)
        end
        last_update = last_x .- x

        params.save_dist && push!(d_hist, norm(last_x .- x))

        if i % params.print_freq == 0
            println(i, ": ", y, " - ||∇f|| = ", grad_norm)
            params.verbosity > 1 && println("\t ||xᵢ - xᵢ₊₁|| = ", norm(last_x .- x))
            params.verbosity > 1 && params.save_cosine_similarity && println("\t cos-similarity = ", csim)
        end
    end

    return (x_opt=x_best, y_hist=y_hist, t_hist=t_hist, g_hist=g_hist, d_hist=d_hist, csims=csims)
end
