
#=
#  Loss functions for optimisation of NN bounds 
#  Each loss function receives the concrete lower and upper bounds of the output neurons to calculate the loss
=#

"""
bounds_loss adds the width of all output bounds.
"""
function bounds_loss(l, u)
    return sum(u .- l)
end


"""
Minimizes the tightness of all output bounds, but stops early if all upper bounds are ≤ 0 (all constraints hold)

We've found that this can sometimes work better than directly minimizing the violation.
"""
function bounds_loss_violation_stop(l, u)
    if all(u .<= 0)
        return 0.
    else
        return sum(u .- l)
    end
end


"""
violation_loss only minizes the upper bound of the output neurons that are > 0 (violating some constraint)
"""
function violation_loss(l, u)
    return sum(max.(0., u))
end