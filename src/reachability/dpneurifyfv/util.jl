
function unit_vec(i, n)
    eᵢ = zeros(n)
    eᵢ[i] = 1.
    return eᵢ
end

"""
We say that the domain of a hyperrectangle is just itself.

Debatable, if the hyperrectangle represents some interval that originated from an input hyperrectangle.
But we just need it for print_progress in our BaB procedure. (but still might be better not to export it)
"""
domain(h::AbstractHyperrectangle) = h 

### Related to symbolic intervals with fresh variables ###

"""
Substitutes variables in symbolic bounds sym_lo and sym_hi with symbolic lower 
and upper bounds var_los and var_his.

sym_lo - symbolic lower bounds
sym_hi - symbolic upper bounds
var_los - symbolic lower bounds of variables to substitute
var_his - symbolic upper bounds of variables to substitute 
n_in - number of input variables (these cannot be substituted)
n_vars - number of variables 
"""
function substitute_variables(sym_lo, sym_hi, var_los, var_his, n_in, n_vars)
    # for sym_lo
    # should we change the mask to [:, n_in + 1: n_in + n_vars] ???
    var_terms = sym_lo[:, n_in + 1: end - 1]

    var_terms⁺ = max.(var_terms, 0)
    var_terms⁻ = min.(var_terms, 0)

    subs_lb = var_terms⁺ * var_los[1:n_vars, :] .+ var_terms⁻ * var_his[1:n_vars, :] .+ sym_lo[:, (1:n_in) ∪ [end]]

    # for sym_hi
    var_terms .= sym_hi[:, n_in + 1: end - 1]

    var_terms⁺ .= max.(var_terms, 0)
    var_terms⁻ .= min.(var_terms, 0)

    subs_ub = var_terms⁺ * var_his[1:n_vars, :] .+ var_terms⁻ * var_los[1:n_vars, :] .+ sym_hi[:, (1:n_in) ∪ [end]]

    return subs_lb, subs_ub
end


"""
Calculates an input within the hyperrectangle [lbs, ubs] that maximizes the linear
symbolic equation sym_eq.
"""
function maximizer(sym_eq, lbs, ubs)
    if size(sym_eq, 1) == 1
        W⁺ = (sym_eq[1:end-1] .> 0)
        W⁻ = (sym_eq[1:end-1] .< 0)
        maximizer = W⁺ .* ubs + W⁻ .* lbs
    else
        W⁺ = (sym_eq[:, 1:end-1] .> 0)
        W⁻ = (sym_eq[:, 1:end-1] .< 0)
        maximizer =( W⁺' .* ubs + W⁻' .* lbs)'
    end

    return maximizer
end

function minimizer(sym_eq, lbs, ubs)
    # minimizer of eq is just maximizer of -eq
    return maximizer(-sym_eq, lbs, ubs)
end


### related to ReLU state


function is_crossing(lb::Float64, ub::Float64)
    lb < 0 && ub > 0 && return true
    return false
end


function relaxed_relu_gradient_lower(l::Real, u::Real)
    ((u <= 0) || ((l < 0) && (u <= -l))) && return 0.
    return 1.
end