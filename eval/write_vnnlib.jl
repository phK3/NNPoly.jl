

function declare_variables(n_in, n_out)
    decls = String[]
    for i in 0:n_in-1
        decl = "(declare-const X_" * string(i) * " Real)"
        push!(decls, decl)
    end
    
    # empty line between inputs and outputs
    push!(decls, "")
    
    for i in 0:n_out-1
        decl = "(declare-const Y_" * string(i) * " Real)"
        push!(decls, decl)
    end
    
    return decls
end


function make_input_constraints(lbs, ubs)
    cs = String[]
    for (i, (lb, ub)) in enumerate(zip(lbs, ubs))
        c_low = "(assert (>= X_" * string(i-1) * " " * string(lb) * "))"
        c_up = "(assert (<= X_" * string(i-1) * " " * string(ub) * "))"
        push!(cs, c_low)
        push!(cs, c_up)
    end
    
    return cs
end


function make_robustness_constraint(label, n_out)
    cs = String[]
    push!(cs, "; output constraints")
    push!(cs, "; unsafe if Y_" * string(label) * " is not maximal")
    push!(cs, "(assert (or")
    for i in 0:n_out-1
        if i != label
            c = "\t(and (>= Y_" * string(i) * " Y_" * string(label) * "))"
            push!(cs, c)
        end
    end
    push!(cs, "))")
    
    return cs
end


function write_property(filename, lbs, ubs, label, n_out; name=nothing)
    decls = declare_variables(length(lbs), n_out)
    cs_in = make_input_constraints(lbs, ubs)
    cs_out = make_robustness_constraint(label, n_out)
    
    open(filename, "w") do f
        if !isnothing(name)
            write(f, "; " * string(name) * "\n")
        end
        
        for d in [decls; ""; cs_in; ""; cs_out]
            write(f, d * "\n")
        end
    end
end