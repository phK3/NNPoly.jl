
@with_kw mutable struct OptimisationParams
    n_steps::Int=100
    # timeout in seconds
    timeout::Float64=60.
    # number of iterations w/o improvement before early stopping
    patience::Int=50
    gradient_tol::Float64=1e-5
    # stop when y ≤ y_stop
    y_stop::Float64=-Inf
    # learning rate scheduling
    start_lr::Float64=1e-3
    decay::Float64=1.
    min_lr::Float64=1e-3
    # console output
    verbosity::Int=1
    print_freq::Int=50
    # save history of loss values
    save_ys::Bool = false
    # save history of gradient norms
    save_gs::Bool = false
    # saving history of times for plotting and analysis
    save_times::Bool = false
    # saving history of distance between iterates
    save_dist::Bool = false
    # save history of cosine similarity between iterates
    save_cosine_similarity::Bool = false
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


struct ExpDecayClipped{N<:Number} <: ParameterSchedulers.AbstractSchedule{false}
    start::N
    decay::N
    min_η::N
end

(schedule::ExpDecayClipped)(t) = max(schedule.min_η, schedule.start * schedule.decay^t)



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
        ŷ, g = withgradient(f, x)
        # Zygote treats nothing as zero
        y = ŷ::Float64
        ∇f = isnothing(g[1]) ? zeros(size(x)) : g[1]::Vector{eltype(x₀)}

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
        if y <= params.y_stop
            println("Stopping criterion reached! y = ", y, " ≤ ", params.y_stop)
            break
        elseif grad_norm < params.gradient_tol
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


"""
Gradient-based optimisation procedure, when parameters are implicitly contained in the model.

args:
    f - function to optimise (depends on parameters in the model)
    model - a Flux Chain with optimisable parameters
    opt - optimiser to use (all Optimisers.jl are supported)
"""
function optimise(f, model::Chain, opt; params=OptimisationParams())
    # TODO: is there an abstract type for all Flux models, not just Chain?
    t_start = time()
    
    sched = ParameterSchedulers.Stateful(ExpDecayClipped(params.start_lr, params.decay, params.min_lr))
    state_tree = Optimisers.setup(opt, model)

    # x_best = x  # how can we get the best params?
    y_best = Inf
    steps_no_improvement = 0
    y_hist = Float64[]
    t_hist = Float64[]

    #mem_before = Sys.free_memory() / 2^20
    #@show mem_before

    for i in 1:params.n_steps
        y, ∇model = withgradient(model) do m
            f(m)
        end

        if y < y_best
            y_best = y
            steps_no_improvement = 0
        else
            steps_no_improvement += 1
        end

        t_now = time()
        elapsed_time = t_now - t_start
        params.save_times && push!(t_hist, elapsed_time)
        params.save_ys && push!(y_hist, y)
        if y <= params.y_stop
            println(i, ": Stopping criterion reached! y = ", y, " ≤ ", params.y_stop)
            break
        elseif steps_no_improvement > params.patience
            println("No improvement over the last ", params.patience, " iterations. Stopping early.")
            break
        elseif elapsed_time >= params.timeout
            println("Timeout reached (", elapsed_time, " elapsed)")
            break
        end

        if i % params.print_freq == 0
            println(i, ": ", y)
        end

        #=
        if mem_before < 1000
            GC.gc()
        end
        mem_after = Sys.free_memory() / 2^20
        @show mem_before - mem_after
        mem_before = mem_after
        =#

        Optimisers.update!(state_tree, model, ∇model[1])

        next_lr = ParameterSchedulers.next!(sched)
        Optimisers.adjust!(state_tree, next_lr)
    end

    # don't really need to return params here since they should be modified in-place in the model
    return (y_hist=y_hist, t_hist=t_hist)
end
