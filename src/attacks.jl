

function fgsm(model, input_set::AbstractHyperrectangle; y_true=nothing, y_target=nothing, n_restarts=1, verbosity=0)
    return fgsm(model, low(input_set), high(input_set), y_true=y_true, y_target=y_target, n_restarts=n_restarts, verbosity=verbosity)
end


"""
Adversarial attack via fast gradient sign method.

args:
    model - (Flux model) the neural network to attack
    lb - concrete lower bounds on the input set
    ub - concrete upper bounds on the input set

kwargs:
    y_true - index of the true classification label (if nothing, defaults to prediction on center of the input set)
    y_target - index of target class (if nothing, attack just tries to decrease logits of y_true)
    n_restarts - number of random restarts (if zero, just run fgsm on the center of the input set)
    verbosity - if > 0 print progress

returns:
    x_attack - the input with the best loss
    y_attack - the output of the model for x_attack
"""
function fgsm(model, lb::AbstractArray, ub::AbstractArray; y_true=nothing, y_target=nothing, n_restarts=0, verbosity=0)
    x₀ = 0.5 .* (lb .+ ub)

    if isnothing(y_true)
        # just take output value at center as label
        ŷ = model(x₀)
        y_true = argmax(ŷ)
    end

    if isnothing(y_target)
        # just decrease the logits of y_true
        lossfun = ŷ -> begin
            ŷ[y_true]
        end

        successfun = y_pred -> y_pred != y_true
    else
        # decrease distance between y_true and the target output
        lossfun = ŷ -> begin
            ŷ[y_true] - ŷ[y_target]
        end
        
        successfun = y_pred -> y_pred == y_target
    end

    x_attack = copy(x₀)
    y_attack = y_true
    best_loss = Inf
    for i in 1:n_restarts+1
        x̂ = fgsm(model, x₀, y_true, lb, ub, y_target=y_target)
        ŷ = model(x̂)


        y_pred = argmax(ŷ)
        cur_loss = lossfun(ŷ)
        if cur_loss < best_loss
            best_loss = cur_loss
            x_attack = x̂
            y_attack = y_pred

            verbosity > 0 && println(i, ": loss = ", cur_loss, " (pred: ", y_pred, ")")

            if successfun(y_pred) 
                verbosity > 0 && println(i, ": loss = ", cur_loss, " (attack successful, pred: ", y_pred, ")")
                break
            end
        end

        x₀ = lb .+ rand(length(lb)) .* (ub .- lb)
    end

    return x_attack, y_attack
end


function fgsm(model, x₀, y_true, lb, ub; y_target=nothing)

    if isnothing(y_target)
        # just decrease the logits of y_true
        lossfun = x -> begin
            ŷ = model(x)
            ŷ[y_true]
        end
    else
        # decrease distance between y_true and the target output
        lossfun = x -> begin
            ŷ = model(x)
            ŷ[y_true] - ŷ[y_target]
        end
    end

    ŷ, ∇f = withgradient(lossfun, x₀)

    # x̂ = x₀ - η ⋅ ∇f(x₀)
    # fgsm goes as far as possible in direction of gradient
    x̂ = ifelse.(∇f[1] .> 0, lb, ub)

    return x̂
end


function pgd(model, input_set::AbstractHyperrectangle; y_true=nothing, y_target=nothing, n_iter=50, n_restarts=1, y_stop=0., 
             eval_pred=true, gradient_tol=1e-5, opt=nothing, verbosity=0, print_freq=25)
    lb = low(input_set)
    ub = high(input_set)
    pgd(model, lb, ub, y_true=y_true, y_target=y_target, n_iter=n_iter, n_restarts=n_restarts, y_stop=y_stop, 
        eval_pred=eval_pred, gradient_tol=gradient_tol, opt=opt, verbosity=verbosity, print_freq=print_freq)
end


"""
Adversarial attack using Projected Gradient Descent.

args:
    model - (Flux model) neural network under x_attack
    lb - concrete lower bound of the input set
    ub - concrete upper bound of the input set

kwargs:
    y_true - index of the true label for the input region (if nothing, takes classification of the input space's center point per default)
    y_target - index of the target class (if nothing, just minimise the logits of y_true)
    n_iter - number of gradient descent iterations per attack 
    n_restarts - number of random restarts (if zero, just run PGD starting from the center of the input space)
    y_stop - stop if loss is smaller than y_stop 
    eval_pred - (bool) evaluate prediction class after every iteration where PGD improves its loss
    gradient_tol - stop if gradient norm is smaller than gradient_tol
    opt - the optimiser to use (all Optimisers.jl optimisers can be submitted - the optimiser has to clip the bounds itself though)
    verbosity - if > 0 print information

returns:
    x_attack - the input with the best loss
    y_attack - the classification for x_attack
"""
function pgd(model, lb, ub; y_true=nothing, y_target=nothing, n_iter=50, n_restarts=1, y_stop=0., 
             eval_pred=true, gradient_tol=1e-5, opt=nothing, verbosity=0, print_freq=25)
    x₀ = 0.5 .* (lb .+ ub)

    if isnothing(y_true)
        # just take output value at center as label
        ŷ = model(x₀)
        y_true = argmax(ŷ)
    end

    return pgd(model, x₀, y_true, lb, ub, y_target=y_target, n_iter=n_iter, n_restarts=n_restarts, y_stop=y_stop, 
               eval_pred=eval_pred, gradient_tol=gradient_tol, opt=opt, verbosity=verbosity, print_freq=print_freq)
end



function pgd(model, x₀::AbstractArray, y_true::Integer, lb::AbstractArray, ub::AbstractArray; 
             y_target=nothing, n_iter=50, opt=nothing, y_stop=0., eval_pred=true, gradient_tol=1e-5, verbosity=0, 
             n_restarts=0, print_freq=25)
    if isnothing(opt)
        opt = Optimisers.OptimiserChain(Optimisers.Adam(), Projection(lb, ub))
    end

    if isnothing(y_target)
        # just decrease the logits of y_true
        lossfun = x -> begin
            ŷ = model(x)

            if argmax(ŷ) != y_true
                # TODO: really stop at zero?
                return y_stop
            else
                return ŷ[y_true]
            end
        end
    else
        # decrease distance between y_true and the target output
        lossfun = x -> begin
            ŷ = model(x)
            ŷ[y_true] - ŷ[y_target]
        end
    end

    x_attack = copy(x₀)
    y_attack = y_true
    best_loss = Inf
    for j in 1:n_restarts+1
        verbosity > 0 && println(j, "th restart ...")
        x = copy(x₀)
        opt_state = Optimisers.setup(opt, x)

        for i in 1:n_iter
            ŷ, ∇f = withgradient(lossfun, x)
            ∇f = isnothing(∇f[1]) ? zeros(size(x)) : ∇f[1]

            grad_norm = norm(∇f)

            if ŷ < best_loss
                best_loss = ŷ
                x_attack = x
                
                verbosity > 0 && i % print_freq == 0 && print(i, ": loss = ", ŷ)
                if eval_pred
                    y = model(x_attack)
                    y_attack = argmax(y)

                    verbosity > 0 && i % print_freq == 0 && print(" (pred: ", y_attack, ")\n")
                end

                if ŷ <= y_stop
                    y = model(x_attack)
                    y_attack = argmax(y)
                    break
                end
            end

            if grad_norm <= gradient_tol
                break
            end

            Optimisers.update!(opt_state, x, ∇f)
        end

        if best_loss <= y_stop
            verbosity > 0 && println("reached stopping criterion: loss = ", best_loss, " ≤ ", y_stop, " (pred = ", y_attack, ")")
            # also break outer loop if sufficiently good attack found
            break
        end

        x₀ = lb .+ rand(length(lb)) .* (ub .- lb)
    end

    return x_attack, y_attack   
end
