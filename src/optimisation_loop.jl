
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
"""
function optimise(f, opt, x₀; print_freq=50, n_steps=100, verbosity=1, gradient_tol=1e-5)
    x = copy(x₀)
    opt_state = Optimisers.setup(opt, x)

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

        if grad_norm < gradient_tol
            println("Optimisation converged! ||∇f|| = ", grad_norm, " < ", gradient_tol)
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

    return x, y_hist, g_hist, d_hist, csims
end
