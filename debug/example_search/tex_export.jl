
# ==========================================================================================
# LaTeX export: a step-by-step description of the bound-propagation ladder (rungs i-vi)
# for a given network, so the formulas can be copied into the thesis instead of being
# written by hand.
#
# Everything is re-derived from scratch in exact `Rational{BigInt}` arithmetic (the real
# solvers mix Float64 into their slope computations, so their symbolic states cannot be
# used directly for exact fractions). `validate_propagation` cross-checks every replicated
# quantity against the real solvers; the TeX numbers are only trusted if it passes.
#
# Sections written:
#   (i)     interval propagation                          (interval_trace)
#   (ii)    symbolic intervals, shared relaxation domain  (DPNFV, :OuterBoundsNeurify, mv=0)
#   (iii)   symbolic intervals, separate relaxations      (DPNFV, :NeurifyRelax, mv=0)
#   (iii')  (iii) + fresh variables                       (DPNFV, :NeurifyRelax, mv>0)
#   (iv)    zonotope propagation (Theorem 3.1 / DeepZ)    (zono_propagate, exact)
#   (v)     backsubstitution chain, n_steps=0             (aCROWN, initialize=true)
#   (vi)    NNPoly first layer + backsub, n_steps=0       (PolyCROWN init: CROWNQuad)
#   summary table of all concrete upper bounds
# ==========================================================================================

# --------------------------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------------------------

"""
Format an exact rational as LaTeX: `\\frac{p}{q}` (sign carried in the numerator),
plain integer when the denominator is 1.
"""
function tex_num(x::Rational)
    p, q = numerator(x), denominator(x)
    q == 1 && return string(p)
    return "\\frac{" * string(p) * "}{" * string(q) * "}"
end

tex_num(x::Real) = tex_num(Rational{BigInt}(x))
tex_num(x::AbstractVector) = "[" * join(tex_num.(x), ",\\; ") * "]"

"Like `tex_num`, but negatives get an explicit minus in front (for use inside sums)."
function tex_num_signed(x::Rational)
    x < 0 && return "-" * tex_num(-x)
    return tex_num(x)
end
tex_num_signed(x::Real) = tex_num_signed(Rational{BigInt}(x))

"""
Inline vector/list like `\\bigl(\\tfrac{1}{2},\\; -\\tfrac{3}{4}\\bigr)^{\\!\\top}`.
"""
function tex_list(v; T=true)
    s = "\\bigl(" * join(tex_num.(v), ",\\; ") * "\\bigr)"
    return T ? s * "^{\\!\\top}" : s
end

"""
`\\frac{1}{2}\\,x - \\frac{3}{4}` from coefficient vector and constant. `vars` names the
variables (one entry per coefficient).
"""
function tex_affine(coeffs, c0; vars=["x"])
    parts = String[]
    for (j, c) in enumerate(coeffs)
        c == 0 && continue
        v = get(vars, j, vars[end])
        term = (abs(c) == 1) ? v : tex_num(abs(c)) * "\\," * v
        if isempty(parts)
            push!(parts, (c < 0 ? "-" : "") * term)
        else
            push!(parts, (c < 0 ? " - " : " + ") * term)
        end
    end
    c0 == 0 && !isempty(parts) && return join(parts)
    cstr = tex_num_signed(c0)
    isempty(parts) && return cstr
    return join(parts) * (c0 < 0 ? " - " : " + ") * tex_num(abs(c0))
end

"A `bmatrix` with exact entries (rows joined by \\\\)."
function tex_matrix(M)
    rows = [join(tex_num.(row), " & ") for row in eachrow(M)]
    return "\\begin{bmatrix}\n" * join(rows, " \\\\\n") * "\n\\end{bmatrix}"
end

"An `align*` block of per-neuron lines: `name_i &= expr`."
function tex_aligned(lines::Vector{String})
    return "\\begin{align*}\n" * join(lines, " \\\\\n") * "\n\\end{align*}"
end

"Per-neuron affine expressions `\\ell^{(k)}_i(...) = ...` in an align* block."
function tex_symbolic_lines(prefix, coeffs, consts; vars=["x"])
    lines = String[]
    for i in 1:size(coeffs, 1)
        row = @view coeffs[i, :]
        expr = tex_affine(collect(row), consts[i]; vars=vars)
        push!(lines, "$(prefix)_{$i} &= " * expr)
    end
    return tex_aligned(lines)
end

# --------------------------------------------------------------------------------------------
# Exact ReLU slope helpers (NV.relaxed_relu_gradient returns Float64 literals for the
# non-crossing cases, which would poison exact-rational arithmetic)
# --------------------------------------------------------------------------------------------

_rrg(l::Real, u::Real) = u <= 0 ? zero(u) : (l >= 0 ? one(u) : u / (u - l))
_rrg_lower(l::Real, u::Real) = ((u <= 0) || ((l < 0) && (u <= -l))) ? zero(u) : one(u)

# --------------------------------------------------------------------------------------------
# Exact replication, rung (i): interval propagation
# --------------------------------------------------------------------------------------------

"""
Exact interval (Box) propagation. Returns per-layer `(l_pre, u_pre, l_post, u_post)`
pre-/post-activation bound vectors, all `Rational{BigInt}`.

Hidden layers are assumed ReLU, the last layer Id (as in `make_network`).
"""
function interval_trace(Ws, bs, input_set)
    T = Rational{BigInt}
    lo = T.(low(input_set)); hi = T.(high(input_set))
    n_layers = length(Ws)
    trace = []
    for k in 1:n_layers
        W = T.(Ws[k]); b = T.(bs[k])
        Wp = max.(W, 0); Wm = min.(W, 0)
        l_pre = Wp * lo .+ Wm * hi .+ b
        u_pre = Wp * hi .+ Wm * lo .+ b
        if k < n_layers
            l_post = max.(l_pre, 0); u_post = max.(u_pre, 0)
        else
            l_post, u_post = copy(l_pre), copy(u_pre)
        end
        push!(trace, (k=k, l_pre=l_pre, u_pre=u_pre, l_post=l_post, u_post=u_post))
        lo, hi = l_post, u_post
    end
    return trace
end

# --------------------------------------------------------------------------------------------
# Exact replication, rungs (ii)/(iii)/(iii'): DPNFV symbolic intervals
# --------------------------------------------------------------------------------------------

"""
`substitute_variables`: replace the fresh-variable columns of `sym_lo`/`sym_hi` by the
symbolic variable bounds `var_los`/`var_his` (upper-style substitution uses them crossed).
Returns the substituted lower-style and upper-style matrices.
"""
function _tex_substitute(sym_lo, sym_hi, var_los, var_his, n_in, n_vars)
    var_terms = sym_lo[:, n_in+1:end-1]
    subs_lb = max.(var_terms, 0) * var_los[1:n_vars, :] .+ min.(var_terms, 0) * var_his[1:n_vars, :] .+
              sym_lo[:, vcat(1:n_in, size(sym_lo, 2))]
    var_terms = sym_hi[:, n_in+1:end-1]
    subs_ub = max.(var_terms, 0) * var_his[1:n_vars, :] .+ min.(var_terms, 0) * var_los[1:n_vars, :] .+
              sym_hi[:, vcat(1:n_in, size(sym_hi, 2))]
    return subs_lb, subs_ub
end

"""
Concrete bounds of a matrix of affine expressions over the (hyperrectangular) domain,
exactly as `NP.bounds(eq, input_set)` computes them.
"""
function _tex_eq_bounds(eq, lo, hi)
    fixed = (lo .== hi)
    W = eq[:, 1:end-1]; b = eq[:, end]
    Wp = max.(W, 0); Wm = min.(W, 0)
    lb = Wp * lo[.~fixed] .+ Wm * hi[.~fixed] .+ b
    ub = Wp * hi[.~fixed] .+ Wm * lo[.~fixed] .+ b
    return lb, ub
end

"Clamp helpers mirroring `lower_bounds`/`upper_bounds`; `nothing` = no bound yet."
_clamp_lo(v, lo_b) = isnothing(lo_b) ? v : max.(v, lo_b)
_clamp_hi(v, hi_b) = isnothing(hi_b) ? v : min.(v, hi_b)

_tex_lower_bounds(eq, lo, hi, cl_lo, cl_hi) = _clamp_hi(_clamp_lo(_tex_eq_bounds(eq, lo, hi)[1], cl_lo), cl_hi)
_tex_upper_bounds(eq, lo, hi, cl_lo, cl_hi) = _clamp_hi(_clamp_lo(_tex_eq_bounds(eq, lo, hi)[2], cl_lo), cl_hi)

"""
Exact replication of a DPNFV run (rungs ii/iii/iii').

Returns `caps` (per-layer captured data), the final substituted bound matrices `L, U`,
the concrete output bounds and the final fresh-variable bookkeeping.
Captured per ReLU layer: pre-/post-act symbolic matrices, the four branch-domain
vectors, the slopes, the fresh-variable assignment and the stored pre-act bounds.
"""
function _tex_dpnfv_run(Ws, bs, input_set; max_vars=0, method=:NeurifyRelax, var_frac=0.5)
    T = Rational{BigInt}
    lo = T.(low(input_set)); hi = T.(high(input_set))
    n_in = length(lo)
    n_layers = length(Ws)

    Low = [Matrix{T}(I, n_in, n_in) zeros(T, n_in)]
    Up  = [Matrix{T}(I, n_in, n_in) zeros(T, n_in)]
    var_los = zeros(T, max_vars, n_in + 1)
    var_his = zeros(T, max_vars, n_in + 1)
    lbs = Vector{Union{Nothing,Vector{T}}}(nothing, n_layers)   # pre-act lower bounds (once set)
    ubs = Vector{Union{Nothing,Vector{T}}}(nothing, n_layers)
    caps = []

    for k in 1:n_layers
        W = T.(Ws[k]); b = T.(bs[k])
        Wp = max.(W, 0); Wm = min.(W, 0)
        is_relu = k < n_layers

        if is_relu
            # forward_linear: interval map with the split weights, bias into the const column
            Low_new = Wp * Low .+ Wm * Up
            Up_new = Wp * Up .+ Wm * Low
            Low_new[:, end] .+= b
            Up_new[:, end] .+= b
            n_node = size(Low_new, 1)
            n_sym = size(Low_new, 2) - 1
            current_n_vars = n_sym - n_in

            subs_Low_Low, subs_Up_Up = _tex_substitute(Low_new, Up_new, var_los, var_his, n_in, current_n_vars)
            # upper-style substitution of Low / lower-style of Up (roles swapped)
            subs_Up_Low, subs_Low_Up = _tex_substitute(Up_new, Low_new, var_los, var_his, n_in, current_n_vars)

            low_low = _tex_lower_bounds(subs_Low_Low, lo, hi, lbs[k], ubs[k])
            low_up  = _tex_upper_bounds(subs_Low_Up,  lo, hi, low_low, ubs[k])
            up_up   = _tex_upper_bounds(subs_Up_Up,   lo, hi, low_up,  ubs[k])
            up_low  = _tex_lower_bounds(subs_Up_Low,  lo, hi, low_low, up_up)

            if method == :OuterBoundsNeurify
                lam_l = _rrg.(low_low, up_up); lam_u = copy(lam_l); beta_u = -low_low
            elseif method == :OuterBoundsDeepPoly
                lam_l = _rrg_lower.(low_low, up_up); lam_u = _rrg.(low_low, up_up); beta_u = -low_low
            elseif method == :NeurifyRelax
                lam_l = _rrg.(low_low, low_up); lam_u = _rrg.(up_low, up_up); beta_u = -up_low
            elseif method == :DeepPolyRelax
                lam_l = _rrg_lower.(low_low, low_up); lam_u = _rrg.(up_low, up_up); beta_u = -up_low
            else
                error("unknown method $method")
            end

            # fresh variables (never in the next-to-last layer: no chance to cancel)
            if k >= n_layers - 1
                fv_idxs = Int[]
            else
                n_node = length(low_low)
                n_vars_target = min(max_vars - current_n_vars, floor(Int, var_frac * n_node))
                if n_vars_target <= 0
                    fv_idxs = Int[]
                else
                    ranges = up_up .- low_low
                    # largest range first, ties broken by neuron index (stable sort order)
                    order = sortperm(ranges; rev=true)
                    crossing = [i for i in order if low_low[i] < 0 && up_up[i] > 0]
                    fv_idxs = crossing[1:min(length(crossing), n_vars_target)]
                end
            end
            n_vars = length(fv_idxs)

            output_Up = lam_u .* Up_new
            output_Up[:, end] .+= lam_u .* max.(beta_u, zero(T))
            output_Low = lam_l .* Low_new

            # the real forward_act relaxes the substituted bounds (dp_neurify_fv.jl
            # lines 81-84) BEFORE storing fresh-variable bounds from them (line 97):
            # the stored xi bounds inherit the neuron's chord relaxation
            subs_Low_Low_rel = lam_l .* subs_Low_Low
            subs_Up_Up_rel = lam_u .* subs_Up_Up
            subs_Up_Up_rel[:, end] .+= lam_u .* max.(beta_u, zero(T))

            if n_vars > 0
                output_Low = [output_Low[:, 1:n_sym] zeros(T, n_node, n_vars) output_Low[:, end]]
                output_Up   = [output_Up[:, 1:n_sym]   zeros(T, n_node, n_vars) output_Up[:, end]]
            end
            var_rows_lo = []; var_rows_hi = []
            for (i, v) in enumerate(fv_idxs)
                push!(var_rows_lo, (v=v, bound=copy(subs_Low_Low_rel[v, :])))
                push!(var_rows_hi, (v=v, bound=copy(subs_Up_Up_rel[v, :])))
                var_los[current_n_vars + i, :] .= subs_Low_Low_rel[v, :]
                var_his[current_n_vars + i, :] .= subs_Up_Up_rel[v, :]
                output_Low[v, :] .= [zeros(T, n_sym); one(T); zeros(T, n_vars)]
                output_Up[v, :]  .= [zeros(T, n_sym); one(T); zeros(T, n_vars)]
            end

            push!(caps, (k=k, is_relu=true, W_p=Wp, W_m=Wm, b=b,
                         Low_in=Low, Up_in=Up, Low_out=Low_new, Up_out=Up_new,
                         n_sym_in=size(Low, 2) - 1, n_sym=size(Low_new, 2) - 1,
                         subs_Low_Low=subs_Low_Low, subs_Up_Up=subs_Up_Up,
                         subs_Up_Low=subs_Up_Low, subs_Low_Up=subs_Low_Up,
                         low_low=low_low, low_up=low_up, up_low=up_low, up_up=up_up,
                         lam_l=lam_l, lam_u=lam_u, beta_u=beta_u,
                         fv_idxs=fv_idxs, var_rows_lo=var_rows_lo, var_rows_hi=var_rows_hi,
                         Low_relaxed=output_Low, Up_relaxed=output_Up,
                         pre_low=copy(low_low), pre_up=copy(up_up)))
            lbs[k] = copy(low_low); ubs[k] = copy(up_up)
            Low, Up = output_Low, output_Up
        else
            # Id layer: forward_linear first (y = W h + b in terms of the previous state),
            # then substitute + bounds (forward_act leaves the state unchanged)
            Low_new = Wp * Low .+ Wm * Up
            Up_new = Wp * Up .+ Wm * Low
            Low_new[:, end] .+= b
            Up_new[:, end] .+= b
            n_sym = size(Low_new, 2) - 1
            current_n_vars = n_sym - n_in
            subs_Low, subs_Up = _tex_substitute(Low_new, Up_new, var_los, var_his, n_in, current_n_vars)
            low_low = _tex_lower_bounds(subs_Low, lo, hi, lbs[k], ubs[k])
            up_up   = _tex_upper_bounds(subs_Up,  lo, hi, low_low, ubs[k])
            push!(caps, (k=k, is_relu=false, W_p=Wp, W_m=Wm, b=b,
                         Low_in=Low, Up_in=Up, Low_out=Low_new, Up_out=Up_new,
                         subs_Low=subs_Low, subs_Up=subs_Up,
                         low_low=low_low, up_up=up_up,
                         pre_low=copy(low_low), pre_up=copy(up_up)))
            lbs[k] = copy(low_low); ubs[k] = copy(up_up)
            Low, Up = Low_new, Up_new
        end
    end

    n_sym = size(Low, 2) - 1
    current_n_vars = n_sym - n_in
    L_final, U_final = _tex_substitute(Low, Up, var_los, var_his, n_in, current_n_vars)
    lb_out = _clamp_hi(_clamp_lo(_tex_eq_bounds(L_final, lo, hi)[1], lbs[end]), ubs[end])
    ub_out = _clamp_hi(_clamp_lo(_tex_eq_bounds(U_final, lo, hi)[2], lbs[end]), ubs[end])

    return (caps=caps, L=L_final, U=U_final, lb=lb_out, ub=ub_out,
            var_los=var_los, var_his=var_his, n_layers=n_layers)
end

# --------------------------------------------------------------------------------------------
# Exact replication, rung (v): aCROWN(0) backsubstitution chain
# --------------------------------------------------------------------------------------------

"""
Concrete bounds of `A*x + b` over a hyperrectangle, as `NP.bounds(A, b, H)`.
"""
_tex_bounds_lin(A, b, lo, hi) = (min.(A, 0) * hi .+ max.(A, 0) * lo .+ b,
                                 min.(A, 0) * lo .+ max.(A, 0) * hi .+ b)

"""
Exact replication of the `aCROWN` initialisation forward pass (`n_steps=0`,
`initialize=true`): iterative prefix backsubstitution with tightening.

Returns per-stage data: the backward steps (slopes and intermediate bound functions for
the lower and upper branch) and the concrete bounds after each stage, all exact.
"""
function _tex_acrown0_run(Ws, bs, input_set)
    T = Rational{BigInt}
    lo = T.(low(input_set)); hi = T.(high(input_set))
    n_layers = length(Ws)
    lbs_cur = Vector{Vector{T}}()
    ubs_cur = Vector{Vector{T}}()
    stages = []

    for i in 1:n_layers
        # Z starts as the last layer of the sub-chain net[1:i] (weights Ws[i], bias bs[i]),
        # i.e. a bound in terms of the *output* of layer i
        Λl = T.(Ws[i]); γl = T.(bs[i])
        Λu = T.(Ws[i]); γu = T.(bs[i])
        steps = []
        for j in i-1:-1:1
            lbs_j = lbs_cur[j]; ubs_j = ubs_cur[j]
            a_l = _rrg_lower.(lbs_j, ubs_j)
            a_u = _rrg.(lbs_j, ubs_j)
            b_u = a_u .* max.(.-lbs_j, zero(T))
            # lower branch: positive coefficients scaled by a_l, negative by a_u (+ offset)
            Λl_out = max.(Λl, 0) .* a_l' .+ min.(Λl, 0) .* a_u'
            γl_out = min.(Λl, 0) * b_u .+ γl
            # upper branch: positive coefficients scaled by a_u, negative by a_l
            Λu_out = max.(Λu, 0) .* a_u' .+ min.(Λu, 0) .* a_l'
            γu_out = max.(Λu, 0) * b_u .+ γu
            # compose with layer j's linear map: z_j = Ws[j]*h_{j-1} + bs[j]
            Λl = Λl_out * T.(Ws[j]); γl = Λl_out * T.(bs[j]) .+ γl_out
            Λu = Λu_out * T.(Ws[j]); γu = Λu_out * T.(bs[j]) .+ γu_out
            push!(steps, (j=j, a_l=a_l, a_u=a_u, b_u=b_u,
                          Λl_out=Λl_out, γl_out=γl_out, Λu_out=Λu_out, γu_out=γu_out,
                          lbs_j=lbs_j, ubs_j=ubs_j))
        end
        ll, lu = _tex_bounds_lin(Λl, γl, lo, hi)
        ul, uu = _tex_bounds_lin(Λu, γu, lo, hi)
        # update_bounds! with (originally infinite) starting bounds
        if length(lbs_cur) < i
            push!(lbs_cur, ll); push!(ubs_cur, uu)
        else
            lbs_cur[i] = max.(lbs_cur[i], ll); ubs_cur[i] = min.(ubs_cur[i], uu)
        end
        push!(stages, (i=i, steps=steps, Λl=Λl, γl=γl, Λu=Λu, γu=γu,
                       ll=ll, lu=lu, ul=ul, uu=uu))
    end
    return (stages=stages, lbs=lbs_cur, ubs=ubs_cur)
end

# --------------------------------------------------------------------------------------------
# Exact replication, rung (vi): NNPoly(0) / PolyCROWN initialisation (single-input nets)
# --------------------------------------------------------------------------------------------

"""
CROWNQuad quadratic ReLU relaxations (as used by the PolyCROWN initialisation),
monomial coefficients `c = [c_0, c_1, c_2]` with `ReLU(x) ≷ c_0 + c_1 x + c_2 x^2`.
Mirrors `relax_relu_crown_quad_lower_matrix` / `..._upper_matrix` exactly.
"""
function _crownquad_lower(l, u)
    u <= 0 && return [0, 0, 0]
    l >= 0 && return [0, 1, 0]
    # crossing neuron
    if u >= -2l
        return [0, 1, 0]
    elseif (-l < u) && (u < -2l)
        c1 = -l * u / (u^2 - l * u)
        c2 = u / (u^2 - l * u)
        return [0, c1, c2]
    else
        return [0, 0, 0]
    end
end

function _crownquad_upper(l, u)
    u <= 0 && return [0, 0, 0]
    l >= 0 && return [0, 1, 0]
    w2 = (l - u)^2
    if -l <= u
        return [-l * u^2, l^2 + u^2, -l] ./ w2
    else
        return [l^2 * u, -2l * u, u] ./ w2
    end
end

"""
Exact replication of the PolyCROWN initialisation forward pass (rung vi, `n_steps=0`)
for a **single-input** network: CROWNQuad quolynomial relaxation of the first ReLU layer,
backsubstitution through the remaining layers w.r.t. the quolynomial domain, and the
final output polynomials.

Returns the relaxation coefficients, the quolynomial coefficients (in `ϵ ∈ [-1,1]`,
where `x = center + radius * ϵ`), the poly-domain bounds per layer, the slopes, the
output polynomial coefficients and the concrete output bounds — all exact.
"""
function _tex_pcrown0_run(Ws, bs, input_set)
    @assert length(low(input_set)) == 1 "single-input networks only"
    T = Rational{BigInt}
    lo = T.(low(input_set)); hi = T.(high(input_set))
    ctr = (lo .+ hi) ./ 2
    rad = (hi .- lo) ./ 2
    n_layers = length(Ws)

    # ---- layer 1 pre-activation as exact affine polynomial in ϵ: z_i = z0_i + z1_i * ϵ
    W1 = T.(Ws[1]); b1 = T.(bs[1])
    z0 = vec(W1 * ctr .+ b1)
    z1 = vec(W1 * rad)
    l1 = z0 .- abs.(z1)          # exact poly bounds (const ± |linear|); equals interval rung
    u1 = z0 .+ abs.(z1)

    # ---- CROWNQuad relaxation coefficients per neuron
    c_l = [_crownquad_lower(l1[i], u1[i]) for i in eachindex(l1)]
    c_u = [_crownquad_upper(l1[i], u1[i]) for i in eachindex(l1)]

    # ---- quolynomial post-activation: c0 + c1*z(ϵ) + c2*z(ϵ)^2, monomials [1, ϵ, ϵ²]
    # L̂ rows = lower relaxation composition, Û rows = upper relaxation composition
    L_poly = zeros(T, length(l1), 3)
    U_poly = zeros(T, length(l1), 3)
    for i in eachindex(l1)
        # monomials [1, ϵ, ϵ²]: c0 + c1*(z0 + z1*ϵ) + c2*(z0 + z1*ϵ)²
        L_poly[i, :] .= [c_l[i][1] + c_l[i][2] * z0[i] + c_l[i][3] * z0[i]^2,
                         c_l[i][2] * z1[i] + c_l[i][3] * 2 * z0[i] * z1[i],
                         c_l[i][3] * z1[i]^2]
        U_poly[i, :] .= [c_u[i][1] + c_u[i][2] * z0[i] + c_u[i][3] * z0[i]^2,
                         c_u[i][2] * z1[i] + c_u[i][3] * 2 * z0[i] * z1[i],
                         c_u[i][3] * z1[i]^2]
    end

    # monomial ranges over ϵ ∈ [-1, 1]: 1 ↦ [1,1], ϵ ↦ [-1,1], ϵ² ↦ [0,1]
    _mb = (1, -1, 0)   # monomial lower bounds (const, ϵ, ϵ²)
    function _poly_bounds(G)
        lb = vec(min.(G, 0) * [1, 1, 1] + max.(G, 0) * [_mb[1], _mb[2], _mb[3]])
        ub = vec(min.(G, 0) * [_mb[1], _mb[2], _mb[3]] + max.(G, 0) * [1, 1, 1])
        return lb, ub
    end
    l_hat, _ = _poly_bounds(L_poly)
    _, u_hat = _poly_bounds(U_poly)

    # ---- layer-2 pre-act bounds w.r.t. the *quolynomial* domain ( tighter than intervals )
    W2 = T.(Ws[2]); b2 = T.(bs[2])
    W2p = max.(W2, 0); W2m = min.(W2, 0)
    G2_l = W2m * U_poly .+ W2p * L_poly; G2_l[:, 1] .+= b2     # lower branch (const monomial col)
    G2_u = W2m * L_poly .+ W2p * U_poly; G2_u[:, 1] .+= b2     # upper branch
    l2, _ = _poly_bounds(G2_l)
    _, u2 = _poly_bounds(G2_u)

    # ---- backsubstitution through the remaining layers (aCROWN, initialize=true)
    @assert n_layers == 3 "tex_export: rung (vi) replication supports exactly two ReLU layers"
    Γl = T.(Ws[end]); γl = T.(bs[end])
    Γu = T.(Ws[end]); γu = T.(bs[end])
    steps = []
    for j in n_layers-1:-1:2
        a_l = _rrg_lower.(l2, u2)
        a_u = _rrg.(l2, u2)
        b_u = a_u .* max.(.-l2, zero(T))
        Λl_out = max.(Γl, 0) .* a_l' .+ min.(Γl, 0) .* a_u'
        γl_out = min.(Γl, 0) * b_u .+ γl
        Λu_out = max.(Γu, 0) .* a_u' .+ min.(Γu, 0) .* a_l'
        γu_out = max.(Γu, 0) * b_u .+ γu
        push!(steps, (j=j, a_l=a_l, a_u=a_u, b_u=b_u, Λl_out=Λl_out, γl_out=γl_out,
                      Λu_out=Λu_out, γu_out=γu_out))
        # compose with layer j's linear map (into the quolynomial domain of layer 1)
        Γl = Λl_out * W2; γl = Λl_out * b2 .+ γl_out
        Γu = Λu_out * W2; γu = Λu_out * b2 .+ γu_out
    end

    # ---- output polynomials: compose the bound function with the quolynomial
    # upper: positive Γ coefficients read Û, negative read L̂ (common generators);
    # translation by γ adds to the constant column
    U_out = max.(Γu, 0) * U_poly .+ min.(Γu, 0) * L_poly
    U_out[:, 1] .+= γu
    L_out = max.(Γl, 0) * L_poly .+ min.(Γl, 0) * U_poly
    L_out[:, 1] .+= γl
    lb_out, _ = _poly_bounds(L_out)
    _, ub_out = _poly_bounds(U_out)

    return (ctr=ctr, rad=rad, z0=z0, z1=z1, l1=l1, u1=u1,
            c_l=c_l, c_u=c_u, L_poly=L_poly, U_poly=U_poly,
            l_hat=l_hat, u_hat=u_hat, G2_l=G2_l, G2_u=G2_u, l2=l2, u2=u2,
            steps=steps, Γl=Γl, γl=γl, Γu=Γu, γu=γu,
            L_out=L_out, U_out=U_out, lb=lb_out, ub=ub_out)
end

# --------------------------------------------------------------------------------------------
# TeX section writers
# --------------------------------------------------------------------------------------------

"bounds pair lines: `\\underline{...}^{(k)} = (…)^{\\top}`"
function _tex_pair_lines(prefix, l, u)
    return ["\\underline{$prefix} &= " * tex_list(l),
            "\\overline{$prefix} &= " * tex_list(u)]
end

function _tex_preamble()
    return raw"""\documentclass[11pt]{article}
\usepackage{amsmath, amssymb, booktabs}
\allowdisplaybreaks
\begin{document}
"""
end

function _tex_postamble()
    return "\n\\end{document}\n"
end

# ---- network definition -------------------------------------------------------------------
function _section_network(Ws, bs, input_set)
    T = Rational{BigInt}
    lo = T.(low(input_set)); hi = T.(high(input_set))
    n_layers = length(Ws)
    io = IOBuffer()
    println(io, "\\section{Network and input set}")
    println(io, "The network \$f:\\mathbb{R} \\to \\mathbb{R}\$ with two hidden ReLU layers:")
    print(io, "\\begin{align*}\nf(x) &= ")
    for k in n_layers:-1:2
        print(io, "W^{($k)}\\,\\mathrm{ReLU}\\bigl(")
    end
    print(io, "W^{(1)}\\,x + b^{(1)}")
    for k in 2:n_layers
        print(io, "\\bigr) + b^{($k)}")
    end
    println(io, "\n\\end{align*}")
    println(io, "with")
    for k in 1:n_layers
        println(io, "\\begin{align*}")
        println(io, "W^{($k)} &= " * tex_matrix(T.(Ws[k])) * ", & b^{($k)} &= " * tex_list(T.(bs[k]); T=false) * ",")
        println(io, "\\end{align*}")
    end
    println(io, "and the input set")
    println(io, "\\begin{equation*}\n x \\in [\\underline{x}, \\overline{x}] = [" *
            tex_num(lo[1]) * ",\\; " * tex_num(hi[1]) * "].\n\\end{equation*}")
    return String(take!(io))
end

# ---- rung (i): interval propagation --------------------------------------------------------
function _section_interval(trace)
    io = IOBuffer()
    n_layers = length(trace)
    println(io, "\\section{(i) Interval propagation}")
    println(io, raw"""Interval propagation (Box) with $W^{(k)+} = \max(W^{(k)}, 0)$ and
$W^{(k)-} = \min(W^{(k)}, 0)$, starting from $\underline{h}^{(0)} = \underline{x}$,
$\overline{h}^{(0)} = \overline{x}$:""")
    println(io, "\\begin{align*}")
    println(io, raw"\underline{z}^{(k)} &= W^{(k)+}\, \underline{h}^{(k-1)} + W^{(k)-}\, \overline{h}^{(k-1)} + b^{(k)}, &")
    println(io, raw"\overline{z}^{(k)} &= W^{(k)+}\, \overline{h}^{(k-1)} + W^{(k)-}\, \underline{h}^{(k-1)} + b^{(k)},")
    println(io, raw"\underline{h}^{(k)} &= \max(\underline{z}^{(k)}, 0), & \overline{h}^{(k)} &= \max(\overline{z}^{(k)}, 0) \quad (\text{ReLU layers}).")
    println(io, "\\end{align*}")
    for t in trace
        println(io, "Layer $(t.k):")
        println(io, "\\begin{align*}")
        println(io, join(_tex_pair_lines("z^{($(t.k))}", t.l_pre, t.u_pre), ", & "))
        if t.k < n_layers
            println(io, " \\\\\n" * join(_tex_pair_lines("h^{($(t.k))}", t.l_post, t.u_post), ", & "))
        end
        println(io, "\\end{align*}")
    end
    println(io, "The interval upper bound on the output is \$\\overline{y}^{\\text{interval}} = " *
            tex_num(trace[end].u_post[1]) * "\$.")
    return String(take!(io))
end

# ---- rungs (ii)/(iii)/(iii'): DPNFV --------------------------------------------------------
_tex_chord_def() = raw"""\mathrm{chord}(l, u) =
\begin{cases}
0, & u \le 0,\\
1, & l \ge 0,\\
\dfrac{u}{u - l}, & \text{otherwise (crossing ReLU)}.
\end{cases}"""

function _section_dpnfv(Ws, bs, input_set; run)
    io = IOBuffer()
    n_layers = length(Ws)
    title = run.kind == :outer ? raw"(ii) Symbolic interval propagation (shared relaxation domain)" :
            run.kind == :sep ? raw"(iii) Symbolic interval propagation (separate relaxation domains)" :
            raw"(iii') Symbolic interval propagation with fresh variables"
    println(io, "\\section{$title}")
    method_name = run.method == :OuterBoundsNeurify ? raw"\texttt{OuterBoundsNeurify}" :
                  raw"\texttt{NeurifyRelax}"
    mv = run.max_vars
    println(io, "DPNFV with method $method_name and max\\_vars = $mv. " *
            raw"""Each neuron carries symbolic lower and upper bound functions
$\underline{z}^{(k)}_i(\cdot) \le z^{(k)}_i \le \overline{z}^{(k)}_i(\cdot)$ (before activation)
and $\underline{h}^{(k)}_i(\cdot) \le h^{(k)}_i \le \overline{h}^{(k)}_i(\cdot)$ (after activation),
affine in the current symbols (initially only $x$).""")

    if run.kind == :outer
        println(io, raw"""Both relaxation slopes use the *shared* (outer) domain
$[\underline{z}^{\ell}_i, \overline{z}^{\ell}_i] = [\underline{z}^{u}_i, \overline{z}^{u}_i] = [\underline{z}^{(k)}_i, \overline{z}^{(k)}_i]$
of the pre-activation bounds, with $\beta_i = -\underline{z}^{\ell}_i$.""")
    elseif run.kind == :sep
        println(io, raw"""The lower and the upper relaxation use *separate* domains:
the lower slope is computed on the range of the lower branch
$[\underline{z}^{\ell}_i, \overline{z}^{\ell}_i]$ (upper end from the upper-style substitution),
the upper slope on the range of the upper branch
$[\underline{z}^{u}_i, \overline{z}^{u}_i]$ (lower end from the lower-style substitution),
with $\beta_i = -\underline{z}^{u}_i$.""")
    else
        println(io, raw"""As in (iii), but the first neuron chosen below keeps its *exact*
symbolic dependence as a fresh variable $\xi$: instead of relaxing it right away, $\xi = h^{(k)}_v$
is carried symbolically and only substituted at the next layer. The stored symbolic bounds of $\xi$
are nevertheless already \emph{relaxed} ones: they are the chord relaxations of the neuron, i.e.
the substituted bounds of $z^{(k)}_v$ multiplied by the neuron's slopes (and shifted by the upper
chord offset), so they inherit exactly the relaxation that rung (iii) would have applied.""")
    end

    for (idx, cap) in enumerate(run.caps)
        k = cap.k
        if !cap.is_relu
            println(io, "\\paragraph{Layer $k (linear).}")
            println(io, "Symbolic bounds of the output (substituted):")
            println(io, tex_symbolic_lines("\\underline{y}", cap.subs_Low[:, 1:end-1], cap.subs_Low[:, end]))
            println(io, tex_symbolic_lines("\\overline{y}", cap.subs_Up[:, 1:end-1], cap.subs_Up[:, end]))
            continue
        end
        vars = run.kind == :fv && k >= 2 ? ["x", "\\xi"] : ["x"]
        println(io, "\\paragraph{Layer $k (ReLU).}")
        println(io, "Symbolic pre-activation bounds after the linear map:")
        println(io, tex_symbolic_lines("\\underline{z}^{($k)}", cap.Low_out[:, 1:end-1], cap.Low_out[:, end]; vars=vars))
        println(io, tex_symbolic_lines("\\overline{z}^{($k)}", cap.Up_out[:, 1:end-1], cap.Up_out[:, end]; vars=vars))

        if run.kind == :fv && idx == 1 && !isempty(cap.fv_idxs)
            v = cap.fv_idxs[1]
            println(io, "The crossing ReLU with the largest range is neuron $v; it becomes a fresh variable. " *
                    "Its symbolic bounds below are the already relaxed (chord) bounds of the neuron:")
            ξlo = cap.var_rows_lo[1].bound; ξhi = cap.var_rows_hi[1].bound
            println(io, "\\begin{align*}")
            println(io, "\\xi &= h^{($k)}_{$v}, & \\underline{\\xi}(x) &= " *
                    tex_affine(collect(ξlo[1:end-1]), ξlo[end]) *
                    ", & \\overline{\\xi}(x) &= " *
                    tex_affine(collect(ξhi[1:end-1]), ξhi[end]) * ".")
            println(io, "\\end{align*}")
        end

        if run.kind == :fv && k >= 2
            # substituted branch expressions (in x only)
            println(io, "Substituting \$\\xi\$ by its symbolic bounds " *
                    raw"($c\,\xi \mapsto c^{+}\,\underline{\xi}(x) + c^{-}\,\overline{\xi}(x)$ " *
                    "in lower-style, and crossed in upper-style substitutions):")
            println(io, tex_symbolic_lines("\\underline{z}^{($k),\\ell}", cap.subs_Low_Low[:, 1:end-1], cap.subs_Low_Low[:, end]))
            println(io, tex_symbolic_lines("\\overline{z}^{($k),\\ell}", cap.subs_Low_Up[:, 1:end-1], cap.subs_Low_Up[:, end]))
            println(io, tex_symbolic_lines("\\underline{z}^{($k),u}", cap.subs_Up_Low[:, 1:end-1], cap.subs_Up_Low[:, end]))
            println(io, tex_symbolic_lines("\\overline{z}^{($k),u}", cap.subs_Up_Up[:, 1:end-1], cap.subs_Up_Up[:, end]))
        end

        println(io, "Branch domains and relaxation parameters:")
        println(io, "\\begin{align*}")
        println(io, "[\\underline{z}^{\\ell}_i, \\overline{z}^{\\ell}_i] &= [" *
                tex_list(cap.low_low; T=false) * ",\\; " * tex_list(cap.low_up; T=false) * "], & " *
                "[\\underline{z}^{u}_i, \\overline{z}^{u}_i] &= [" *
                tex_list(cap.up_low; T=false) * ",\\; " * tex_list(cap.up_up; T=false) * "], \\\\")
        println(io, "\\lambda^{\\ell}_i &= " * tex_list(cap.lam_l; T=false) * ", & " *
                "\\lambda^{u}_i &= " * tex_list(cap.lam_u; T=false) * ", & " *
                "\\beta_i &= " * tex_list(cap.beta_u; T=false) * ".")
        println(io, "\\end{align*}")
        println(io, raw"""with the relaxation rule
$\underline{h}^{(k)}_i = \lambda^{\ell}_i\, \underline{z}^{(k)}_i$ and
$\overline{h}^{(k)}_i = \lambda^{u}_i\, \overline{z}^{(k)}_i + \lambda^{u}_i \max(\beta_i, 0)$,
$\lambda^{\ell/u}_i = \mathrm{chord}(\underline{z}_i^{\ell/u}, \overline{z}_i^{\ell/u})$,""")
        println(io, _tex_chord_def())
        println(io, "Relaxed post-activation bounds:")
        println(io, tex_symbolic_lines("\\underline{h}^{($k)}", cap.Low_relaxed[:, 1:end-1], cap.Low_relaxed[:, end]; vars=vars))
        println(io, tex_symbolic_lines("\\overline{h}^{($k)}", cap.Up_relaxed[:, 1:end-1], cap.Up_relaxed[:, end]; vars=vars))
    end

    println(io, "\\paragraph{Output bounds.}")
    println(io, "Substituting all remaining symbols gives the affine output bounds")
    println(io, tex_symbolic_lines("\\underline{y}", run.L[:, 1:end-1], run.L[:, end]))
    println(io, tex_symbolic_lines("\\overline{y}", run.U[:, 1:end-1], run.U[:, end]))
    println(io, "\\begin{equation*}")
    println(io, "\\underline{y}^{\\text{DP}} = " * tex_num(run.lb[1]) *
            ", \\qquad \\overline{y}^{\\text{DP}} = " * tex_num(run.ub[1]) * ".")
    println(io, "\\end{equation*}")
    return String(take!(io))
end

# ---- rung (iv): zonotope -------------------------------------------------------------------
function _section_zonotope(net, input_set)
    T = Rational{BigInt}
    _, trace = zono_propagate(net, input_set, T)
    io = IOBuffer()
    println(io, "\\section{(iv) Zonotope propagation (Theorem 3.1, DeepZ)}")
    println(io, raw"""A zonotope is $Z = \langle c, G \rangle = \{c + G\epsilon : \epsilon \in [-1,1]^m\}$
(center $c$, generators the columns of $G$); its exact interval bounds are
$c \pm \sum_j |G_{:j}|$. The input set becomes $Z^{(0)} = \langle c^{(0)}, G^{(0)} \rangle$ with
$c^{(0)} = (\underline{x} + \overline{x})/2$ and $G^{(0)}$ the diagonal of $(\overline{x} - \underline{x})/2$.
Per ReLU neuron with interval bounds $[l_i, u_i]$:
$\lambda_i = \mathrm{chord}(l_i, u_i)$ and, for crossing neurons,
$\mathrm{ReLU}(z_i) \le \lambda_i (z_i - l_i) = \lambda_i z_i + \mu_i$ with error-generator magnitude
$\mu_i = -\lambda_i l_i / 2 > 0$. The rectified zonotope is""")
    println(io, "\\begin{equation*}")
    println(io, raw"c' = \lambda \odot c + \mu, \qquad G' = \bigl[\, \lambda \odot G \;\big|\; \mu\, \bigr]")
    println(io, "\\end{equation*}")
    println(io, raw"""($\lambda_i = 1, \mu_i = 0$ for active, $\lambda_i = 0$ for inactive neurons;
the new error generator $\mu_i$ has entries only in row $i$).""")
    println(io, "For the example network:")
    labels = ["\\epsilon_{1}"]
    for t in trace
        println(io, "\\paragraph{Layer $(t.layer).}")
        pre = t.pre
        println(io, "Pre-activation zonotope: \$c = " * tex_list(vec(center(pre)); T=false) *
                "\$, generator columns:")
        G = genmat(pre)
        println(io, "\\begin{align*}")
        for j in 1:size(G, 2)
            lab = j <= length(labels) ? labels[j] : "g_{$j}"
            println(io, "g_{$j} &= " * tex_list(vec(G[:, j]); T=false) * " \\quad (" * lab * ")" *
                    (j < size(G, 2) ? ", &" : "."))
        end
        println(io, "\\end{align*}")
        println(io, "Interval bounds: \$l = " * tex_list(t.lo; T=false) * ", \\; u = " *
                tex_list(t.hi; T=false) * "\$. ")
        if any(t.crossing)
            println(io, "Crossing ReLUs at neurons " * join(findall(t.crossing), ", ") * " with")
            println(io, "\\begin{align*}")
            println(io, "\\lambda &= " * tex_list(t.λ; T=false) * ", & \\mu &= " * tex_list(t.μ; T=false) * ".")
            println(io, "\\end{align*}")
        else
            println(io, "No crossing ReLUs (\$\\lambda = " * tex_list(t.λ; T=false) * "\$).")
        end
        post = t.post
        println(io, "Post-activation zonotope: \$c' = " * tex_list(vec(center(post)); T=false) *
                "\$, generators:")
        Gp = genmat(post)
        println(io, "\\begin{align*}")
        for j in 1:size(Gp, 2)
            println(io, "g'_{$j} &= " * tex_list(vec(Gp[:, j]); T=false) * (j < size(Gp, 2) ? ", &" : "."))
        end
        println(io, "\\end{align*}")
        for i in findall(t.crossing)
            push!(labels, "\\eta^{($(t.layer))}_{$i}")
        end
    end
    Z, _ = zono_propagate(net, input_set, T)
    ub = zono_upper_bound(Z)
    println(io, "The zonotope upper bound on the output is \$\\overline{y}^{\\text{zono}} = c + \\sum_j |g_j| = " *
            tex_num(ub[1]) * "\$.")
    return String(take!(io))
end

# ---- rung (v): aCROWN(0) backsubstitution chain ---------------------------------------------
function _section_acrown0(Ws, bs, run)
    io = IOBuffer()
    T = Rational{BigInt}
    n_layers = length(Ws)
    n_in = size(Ws[1], 2)
    println(io, "\\section{(v) CROWN backsubstitution (aCROWN initialisation pass)}")
    println(io, raw"""CROWN computes the output bound by \emph{backsubstitution}: affine bound functions
$\underline{g}(h) = \Lambda h + \gamma \le z$ and $\overline{g}(h) = \Gamma h + \gamma \ge z$
are propagated backwards layer by layer, relaxing each ReLU by chord bounds whose slopes are taken
from the concrete pre-activation bounds found in earlier stages of the same pass. For a neuron with
relaxation domain $[\ell_i, u_i]$ (the concrete bounds of $z^{(k)}_i$) define""")
    println(io, "\\begin{align*}")
    println(io, raw"a^{\ell}_i &= \begin{cases} 1, & \ell_i \ge 0,\\ 0, & \text{otherwise}, \end{cases} & a^{u}_i &= \mathrm{chord}(\ell_i, u_i), & b^{u}_i &= a^{u}_i \max(-\ell_i, 0),")
    println(io, "\\end{align*}")
    println(io, raw"""so that $\mathrm{ReLU}(z_i) \ge a^{\ell}_i z_i$ and
$\mathrm{ReLU}(z_i) \le a^{u}_i z_i + b^{u}_i$. A bound in terms of the post-activations $h$ becomes
one in terms of the pre-activations $z$ via""")
    println(io, "\\begin{align*}")
    println(io, raw"\Lambda' &= \Lambda^{+} a^{\ell} + \Lambda^{-} a^{u}, & \gamma' &= \Lambda^{-} b^{u} + \gamma && \text{(lower)}, \\")
    println(io, raw"\Gamma' &= \Gamma^{+} a^{u} + \Gamma^{-} a^{\ell}, & \gamma' &= \Gamma^{+} b^{u} + \gamma && \text{(upper)},")
    println(io, "\\end{align*}")
    println(io, raw"""with $\Lambda^{+} = \max(\Lambda, 0)$, $\Lambda^{-} = \min(\Lambda, 0)$ (analogously $\Gamma$) and the
slope vectors acting column-wise (diagonal scaling). Composing with
$z^{(k)} = W^{(k)} h^{(k-1)} + b^{(k)}$ yields the bound in terms of $h^{(k-1)}$. The pass runs in
stages $i = 1, \dots, n$: stage $i$ backsubstitutes the pre-activation $z^{(i)}$ to the input, using
the concrete bounds recorded by stages $< i$ as relaxation domains, and records
$[\underline{z}^{(i)}, \overline{z}^{(i)}]$ for the later stages. Within each stage, $g$ and $g'$
denote the current lower/upper bound functions (redefined at every step).""")
    for st in run.stages
        tgt = st.i == n_layers ? raw"the output $y$" : "\$z^{($(st.i))}\$"
        println(io, "\\paragraph{Stage $(st.i): bounds on $tgt.}")
        if isempty(st.steps)
            println(io, "The bound functions are the layer map \$z^{(1)} = W^{(1)} x + b^{(1)}\$ itself; " *
                    "both branches coincide and their interval evaluation is the interval bound of rung (i).")
        end
        for s in st.steps
            j = s.j
            println(io, "Backsubstituting through layer $j (ReLU) with relaxation domain " *
                    "\$[\\underline{z}^{(j)}, \\overline{z}^{(j)}] = [" * tex_list(s.lbs_j; T=false) *
                    ",\\; " * tex_list(s.ubs_j; T=false) * "]\$ (the stage-$j result):")
            println(io, "\\begin{align*}")
            println(io, "a^{\\ell} &= " * tex_list(s.a_l; T=false) * ", & a^{u} &= " *
                    tex_list(s.a_u; T=false) * ", & b^{u} &= " * tex_list(s.b_u; T=false) * ".")
            println(io, "\\end{align*}")
            zvars = ["z^{($j)}_{$i}" for i in 1:size(s.Λl_out, 2)]
            println(io, "Applying the relaxation rules gives bound functions in terms of \$z^{($j)}\$:")
            println(io, tex_symbolic_lines("\\underline{g}", s.Λl_out, s.γl_out; vars=zvars))
            println(io, tex_symbolic_lines("\\overline{g}", s.Λu_out, s.γu_out; vars=zvars))
            if j == 1
                hvars = n_in == 1 ? ["x"] : ["x_{$i}" for i in 1:n_in]
                println(io, "Composing with \$z^{(1)} = W^{(1)} x + b^{(1)}\$ gives bound functions in terms of \$x\$:")
            else
                hvars = ["h^{$(j-1)}_{$i}" for i in 1:size(Ws[j], 2)]
                println(io, "Composing with \$z^{(j)} = W^{(j)} h^{$(j-1)} + b^{(j)}\$ gives bound functions in terms of \$h^{$(j-1)}\$:")
            end
            println(io, tex_symbolic_lines("\\underline{g}'", s.Λl_out * T.(Ws[j]), s.Λl_out * T.(bs[j]) .+ s.γl_out; vars=hvars))
            println(io, tex_symbolic_lines("\\overline{g}'", s.Λu_out * T.(Ws[j]), s.Λu_out * T.(bs[j]) .+ s.γu_out; vars=hvars))
        end
        println(io, "Interval evaluation of both branches over \$[\\underline{x}, \\overline{x}]\$ gives")
        println(io, "\\begin{align*}")
        println(io, "[\\underline{z}^{\\ell}, \\overline{z}^{\\ell}] &= [" * tex_list(st.ll; T=false) * ",\\; " *
                tex_list(st.lu; T=false) * "] \\quad\\text{(lower branch)}, & " *
                "[\\underline{z}^{u}, \\overline{z}^{u}] &= [" * tex_list(st.ul; T=false) * ",\\; " *
                tex_list(st.uu; T=false) * "] \\quad\\text{(upper branch)}, \\\\")
        tgt_rec = st.i == n_layers ? "y" : "z^{($(st.i))}"
        println(io, "\\text{recorded: } [\\underline{" * tgt_rec * "}, \\overline{" * tgt_rec * "}] &= [" *
                tex_list(st.ll; T=false) * ",\\; " * tex_list(st.uu; T=false) * "] " *
                raw"\quad\text{(lower end from the lower, upper end from the upper branch)}.")
        println(io, "\\end{align*}")
    end
    println(io, "\\paragraph{Output bound.}")
    println(io, "\\begin{equation*}")
    println(io, "\\underline{y}^{\\text{CROWN}} = " * tex_num(run.lbs[end][1]) *
            ", \\qquad \\overline{y}^{\\text{CROWN}} = " * tex_num(run.ubs[end][1]) * ".")
    println(io, "\\end{equation*}")
    return String(take!(io))
end

# ---- rung (vi): NNPoly / PolyCROWN initialisation (quolynomial relaxation) -------------------
function _section_pcrown0(run)
    io = IOBuffer()
    println(io, "\\section{(vi) NNPoly with quadratic relaxations (PolyCROWN initialisation pass)}")
    println(io, "With the single input \$x = c + r\\,\\epsilon\$, \$\\epsilon \\in [-1, 1]\$ " *
            "(center \$c = " * tex_list(run.ctr; T=false) * "\$, radius \$r = " * tex_list(run.rad; T=false) * "\$), " *
            "the first-layer pre-activations are exact affine polynomials in \$\\epsilon\$, " *
            raw"$z^{(1)}_i = z^{(1)}_{i,0} + z^{(1)}_{i,1}\,\epsilon$:")
    println(io, tex_aligned(["z^{(1)}_{$i} &= " * tex_affine([run.z1[i]], run.z0[i]; vars=["\\epsilon"])
                             for i in eachindex(run.z0)]))
    println(io, "Their exact bounds \$\\underline{z}^{(1)}_i = z^{(1)}_{i,0} - |z^{(1)}_{i,1}|\$ and " *
            "\$\\overline{z}^{(1)}_i = z^{(1)}_{i,0} + |z^{(1)}_{i,1}|\$ agree with the interval rung (i):")
    println(io, "\\begin{align*}")
    println(io, join(_tex_pair_lines("z^{(1)}", run.l1, run.u1), ", & ") * ".")
    println(io, "\\end{align*}")
    println(io, raw"""Each crossing ReLU neuron (with $\ell_i = \underline{z}^{(1)}_i < 0 < u_i = \overline{z}^{(1)}_i$)
receives a quadratic (CROWNQuad) relaxation; active and inactive neurons use $\mathrm{ReLU}(z) \le z$
and $\mathrm{ReLU}(z) = 0$. For crossing neurons:""")
    println(io, "\\begin{align*}")
    println(io, raw"\text{upper: } \mathrm{ReLU}(z) &\le \begin{cases} \dfrac{-l u^2 + (l^2 + u^2)\, z - l\, z^2}{(l - u)^2}, & -l \le u, \\[3mm] \dfrac{l^2 u - 2 l u\, z + u\, z^2}{(l - u)^2}, & -l > u, \end{cases} \\[2mm]")
    println(io, raw"\text{lower: } \mathrm{ReLU}(z) &\ge \begin{cases} z, & u \ge -2l, \\[3mm] \dfrac{-l u\, z + u\, z^2}{u^2 - l u}, & -l < u < -2l, \\[3mm] 0, & u \le -l. \end{cases}")
    println(io, "\\end{align*}")
    println(io, "Per-neuron monomial coefficients \$c_i = (c_0, c_1, c_2)\$ with " *
            raw"$\mathrm{ReLU}(z_i) \lesseqgtr c_0 + c_1 z_i + c_2 z_i^2$:")
    println(io, tex_aligned(["c^{\\ell}_{$i} &= " * tex_list(run.c_l[i]; T=false) * ", & c^{u}_{$i} &= " *
                             tex_list(run.c_u[i]; T=false) for i in eachindex(run.c_l)]))
    println(io, raw"""Substituting $z^{(1)}_i(\epsilon) = z^{(1)}_{i,0} + z^{(1)}_{i,1} \epsilon$ into the relaxation
polynomials (with $z^2 = z_0^2 + 2 z_0 z_1 \epsilon + z_1^2 \epsilon^2$) gives the quolynomial
(degree-2 polynomial) bounds of the post-activations:""")
    println(io, tex_symbolic_lines("\\hat{h}^{(1),\\ell}", run.L_poly[:, 2:3], run.L_poly[:, 1];
                                   vars=["\\epsilon", "\\epsilon^2"]))
    println(io, tex_symbolic_lines("\\hat{h}^{(1),u}", run.U_poly[:, 2:3], run.U_poly[:, 1];
                                   vars=["\\epsilon", "\\epsilon^2"]))
    println(io, raw"""Concrete bounds of a quolynomial $q(\epsilon) = q_0 + q_1 \epsilon + q_2 \epsilon^2$ over
$\epsilon \in [-1, 1]$ follow from the monomial ranges $1 \in [1, 1]$, $\epsilon \in [-1, 1]$,
$\epsilon^2 \in [0, 1]$, i.e. from $q_0 \pm |q_1|$ and the sign of $q_2$:""")
    println(io, "\\begin{equation*}")
    println(io, raw"q \in \bigl[\, q_0 - |q_1| + \min(q_2, 0), \;\; q_0 + |q_1| + \max(q_2, 0) \,\bigr].")
    println(io, "\\end{equation*}")
    println(io, "Applied to the quolynomial bounds of \$h^{(1)}\$ (lower branch for the lower end, upper branch for the upper end):")
    println(io, "\\begin{align*}")
    println(io, join(_tex_pair_lines("\\hat{h}^{(1)}", run.l_hat, run.u_hat), ", & ") * ".")
    println(io, "\\end{align*}")
    println(io, raw"""Composing the linear map $z^{(2)} = W^{(2)} h^{(1)} + b^{(2)}$ with the quolynomial bounds
(positive coefficients read the lower bound polynomial $\hat{h}^{\ell}$, negative coefficients the
upper one $\hat{h}^{u}$; the bias enters the constant monomial) gives quolynomial bounds of
$z^{(2)}$:""")
    println(io, tex_symbolic_lines("\\underline{z}^{(2)}", run.G2_l[:, 2:3], run.G2_l[:, 1];
                                   vars=["\\epsilon", "\\epsilon^2"]))
    println(io, tex_symbolic_lines("\\overline{z}^{(2)}", run.G2_u[:, 2:3], run.G2_u[:, 1];
                                   vars=["\\epsilon", "\\epsilon^2"]))
    println(io, "Their concrete bounds (same monomial rule, now over the \emph{quolynomial} domain, " *
            "which is tighter than the interval rung):")
    println(io, "\\begin{align*}")
    println(io, join(_tex_pair_lines("z^{(2)}", run.l2, run.u2), ", & ") * ".")
    println(io, "\\end{align*}")
    println(io, "The output \$y = W^{(3)} h^{(2)} + b^{(3)}\$ is backsubstituted through ReLU layer 2 " *
            raw"""with the linear chord rules of rung (v), using the quolynomial-domain bounds
$[\underline{z}^{(2)}, \overline{z}^{(2)}]$ as relaxation domains:""")
    s = run.steps[1]
    println(io, "\\begin{align*}")
    println(io, "a^{\\ell} &= " * tex_list(s.a_l; T=false) * ", & a^{u} &= " *
            tex_list(s.a_u; T=false) * ", & b^{u} &= " * tex_list(s.b_u; T=false) * ".")
    println(io, "\\end{align*}")
    z2vars = ["z^{(2)}_{$i}" for i in eachindex(run.l2)]
    println(io, "Relaxing ReLU layer 2 gives bound functions in terms of \$z^{(2)}\$:")
    println(io, tex_symbolic_lines("\\underline{g}", s.Λl_out, s.γl_out; vars=z2vars))
    println(io, tex_symbolic_lines("\\overline{g}", s.Λu_out, s.γu_out; vars=z2vars))
    h1vars = ["h^{(1)}_{$i}" for i in 1:size(run.Γl, 2)]
    println(io, "Composing with \$z^{(2)} = W^{(2)} h^{(1)} + b^{(2)}\$ gives bound functions in terms of \$h^{(1)}\$:")
    println(io, tex_symbolic_lines("\\underline{g}'", run.Γl, run.γl; vars=h1vars))
    println(io, tex_symbolic_lines("\\overline{g}'", run.Γu, run.γu; vars=h1vars))
    println(io, raw"""Composing these bound functions with the quolynomial bounds of $h^{(1)}$ (positive
coefficients read $\hat{h}^{u}$ resp.\ $\hat{h}^{\ell}$ as appropriate, the shift $\gamma$ enters the
constant monomial) yields polynomial bounds on the output:""")
    println(io, tex_symbolic_lines("\\underline{y}", run.L_out[:, 2:3], run.L_out[:, 1];
                                   vars=["\\epsilon", "\\epsilon^2"]))
    println(io, tex_symbolic_lines("\\overline{y}", run.U_out[:, 2:3], run.U_out[:, 1];
                                   vars=["\\epsilon", "\\epsilon^2"]))
    println(io, "Evaluating both over the monomial domain gives")
    println(io, "\\begin{equation*}")
    println(io, "\\underline{y}^{\\text{NNPoly}} = " * tex_num(run.lb[1]) *
            ", \\qquad \\overline{y}^{\\text{NNPoly}} = " * tex_num(run.ub[1]) * ".")
    println(io, "\\end{equation*}")
    return String(take!(io))
end

# ---- summary table ---------------------------------------------------------------------------
_tex_val(x::Rational) = tex_num(x)
_tex_val(x::Real) = tex_num(rationalize(Float64(x); tol=1e-9))

function _section_summary(vals; n_steps_opt=nothing, opt_vals=nothing)
    io = IOBuffer()
    println(io, "\\section{Summary: the bound ladder}")
    println(io, raw"""Collecting the upper bounds; they are ordered loose to tight from (i) to (vi)
(and further tightened by the optimised variants):""")
    println(io, "\\begin{table}[ht]\\centering")
    println(io, "\\begin{tabular}{llc}\\toprule")
    println(io, "rung & algorithm & \$\\overline{y}\$ \\\\")
    println(io, "\\midrule")
    rows = [("i", "interval propagation", vals.interval),
            ("ii", "symbolic interval (shared domains)", vals.outer),
            ("iii", "symbolic interval (separate domains)", vals.sep),
            ("iii'", "symbolic interval + fresh variables", vals.fv),
            ("iv", "zonotope propagation (DeepZ)", vals.zono),
            ("v", "CROWN backsubstitution", vals.acrown0),
            ("vi", "NNPoly (quadratic relaxations)", vals.pcrown0)]
    for (r, name, v) in rows
        vstr = v === nothing ? raw"\text{(multi-input)}" : _tex_val(v)
        println(io, "($r) & $name & $vstr \\\\")
    end
    if n_steps_opt !== nothing && opt_vals !== nothing
        println(io, "\\midrule")
        println(io, raw"(vii) & CROWN, optimised slopes & " * _tex_val(opt_vals.acrown_opt) * " \\\\")
        println(io, raw"(viii) & NNPoly, optimised slopes & " * _tex_val(opt_vals.pcrown_opt) * " \\\\")
        println(io, raw"\multicolumn{3}{l}{\footnotesize optimised with $n_{\text{steps}} = " *
                string(n_steps_opt) * raw"$ gradient steps} \\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
    println(io, "\\end{table}")
    return String(take!(io))
end

# ==========================================================================================
# Validation: every exact replication vs the real (Float64) solvers
# ==========================================================================================

"Relative deviation `|a - b| / max(1, |b|)` (Float64)."
_rd(a, b) = abs(Float64(a) - Float64(b)) / max(1.0, abs(Float64(b)))

"Maximum relative deviation over two equally-shaped arrays."
_rdm(A, B) = maximum(_rd.(A, B))

"""
Validate all exact replications of the tex exporter against the real solvers.

Compares rung by rung:
(i) interval, (ii)/(iii)/(iii') DPNFV (bounds, substituted matrices and per-layer pre-activation
bounds), (iv) zonotope, (v) aCROWN(0) (per-stage bounds and the final Λ/Γ), and — for single-input
networks — (vi) PolyCROWN(0) (pre-activation bounds, relaxation coefficients α, quolynomial bounds,
poly-domain bounds, backsubstitution matrices and the final bounds).

Returns `(dev, ok, tol)` with `dev` a Dict of label => relative deviation and `ok` the overall pass.
"""
function validate_propagation(Ws, bs, input_set=default_input_set(); tol=1e-9, verbose=true)
    dev = Dict{String,Float64}()
    net = make_network(Ws, bs)
    net_npi = make_network_npi(Ws, bs)

    # (i) interval propagation
    trace = interval_trace(Ws, bs, input_set)
    s_box = NV.forward_network(NV.Box(), net, input_set)
    dev["(i) ub"] = _rd(trace[end].u_post[1], Float64(high(s_box)[1]))

    # (ii)/(iii)/(iii') DPNFV
    for (label, mv, mth) in [("(ii)", 0, :OuterBoundsNeurify),
                             ("(iii)", 0, :NeurifyRelax),
                             ("(iii')", 10, :NeurifyRelax)]
        run = _tex_dpnfv_run(Ws, bs, input_set; max_vars=mv, method=mth)
        ref = dpnfv_details(net_npi, input_set; max_vars=mv, method=mth)
        dev[label * " ub"] = _rd(run.ub[1], ref.ub)
        dev[label * " L/U"] = max(_rdm(run.L, ref.L), _rdm(run.U, ref.U))
        for (cap, rl, ru) in zip(run.caps, ref.lbs_per_layer, ref.ubs_per_layer)
            dev[label * " pre[$(cap.k)]"] = max(_rdm(cap.pre_low, rl), _rdm(cap.pre_up, ru))
        end
    end

    # (iv) zonotope
    dev["(iv) ub"] = validate_zono(net, input_set)

    # (v) aCROWN(0)
    run_a = _tex_acrown0_run(Ws, bs, input_set)
    ref_a = acrown0_details(Ws, bs, input_set)
    dev["(v) ub"] = _rd(run_a.ubs[end][1], Float64(ref_a.ubs[end][1]))
    dev["(v) per-stage"] = max(maximum(_rdm.(run_a.lbs, ref_a.lbs)), maximum(_rdm.(run_a.ubs, ref_a.ubs)))
    dev["(v) Lambda/Gamma"] = max(_rdm(run_a.stages[end].Λl, ref_a.Λ), _rdm(run_a.stages[end].γl, ref_a.λ),
                                  _rdm(run_a.stages[end].Λu, ref_a.Γ), _rdm(run_a.stages[end].γu, ref_a.γ))

    # (vi) PolyCROWN(0) — single-input networks only
    if length(low(input_set)) == 1
        solver = _PCROWN_SOLVER()
        net_pcrown = make_crown_chain(Ws, bs; poly=true)
        ps = solver.poly_solver
        s_dom = NP.initialize_symbolic_domain(solver, net_pcrown[1:1], input_set)
        s_hat, lbs_p, ubs_p, rs, cs, sf, ui, di = NP.initialize_params_bounds(solver, net_pcrown, 2, s_dom)
        l1r, u1r = NP.bounds(s_hat.poly_interval.Low)

        run_p = _tex_pcrown0_run(Ws, bs, input_set)
        dev["(vi) l1/u1"] = max(_rdm(run_p.l1, l1r), _rdm(run_p.u1, u1r))

        # relaxation coefficients stored in the layer parameters α
        α = net_pcrown[1].α
        dev["(vi) alpha"] = max(
            _rdm([run_p.c_l[i][2] for i in eachindex(run_p.c_l)], α[:, 1, 1]),
            _rdm([run_p.c_l[i][3] for i in eachindex(run_p.c_l)], α[:, 2, 1]),
            _rdm([run_p.c_u[i][2] for i in eachindex(run_p.c_u)], α[:, 1, 2]),
            _rdm([run_p.c_u[i][3] for i in eachindex(run_p.c_u)], α[:, 2, 2]))

        # recompute the quolynomial state exactly like the initialisation run (init = true)
        ipsolver = NP.DiffNNPolySym(truncation_terms=ps.truncation_terms,
                                    separate_relaxations=ps.separate_relaxations,
                                    relaxations=ps.relaxations, splitting_depth=ps.splitting_depth,
                                    init=true, init_method=ps.init_method, save_bounds=ps.save_bounds,
                                    common_generators=ps.common_generators)
        rs2 = Vector{Int}(); cs2 = Vector{Int}(); sf2 = Vector{Int}()
        ui2 = Vector{Int}(); di2 = Vector{Int}()
        s_poly = NP.forward_act_stub(ipsolver, net_pcrown[1], s_hat, l1r, u1r, rs2, cs2, sf2, ui2, di2)
        l_hat_r, _ = NP.bounds(s_poly.poly_interval.Low)
        _, u_hat_r = NP.bounds(s_poly.poly_interval.Up)
        dev["(vi) l_hat/u_hat"] = max(_rdm(run_p.l_hat, l_hat_r), _rdm(run_p.u_hat, u_hat_r))
        dev["(vi) l2/u2"] = max(_rdm(run_p.l2, lbs_p[2]), _rdm(run_p.u2, ubs_p[2]))

        # backsubstitution through the linear part over the quolynomial domain
        s_crown = NV.forward_network(solver.lin_solver, net_pcrown[2:end], s_poly, lbs_p[2:end], ubs_p[2:end])
        dev["(vi) Gamma/gamma"] = max(_rdm(run_p.Γl, s_crown.Λ), _rdm(run_p.γl, s_crown.λ),
                                      _rdm(run_p.Γu, s_crown.Γ), _rdm(run_p.γu, s_crown.γ))
        dev["(vi) lb/ub"] = max(_rd(run_p.lb[1], Float64(lbs_p[end][1])), _rd(run_p.ub[1], Float64(ubs_p[end][1])))
    end

    ok = all(v -> v <= tol, values(dev))
    if verbose
        println("validate_propagation (tol = $tol):")
        for k in sort(collect(keys(dev)))
            v = dev[k]
            println("  ", rpad(k, 24), lpad(round(v; sigdigits=3), 10), v <= tol ? "   ok" : "   MISMATCH")
        end
        println(ok ? "all checks passed" : "SOME CHECKS FAILED")
    end
    return (dev=dev, ok=ok, tol=tol)
end

# ==========================================================================================
# API: assemble and write the propagation TeX document
# ==========================================================================================

"""
Assemble the full TeX document describing the propagation of all symbolic bounds through the
network `Ws, bs` on `input_set`, rung by rung (i)–(vi) plus the summary table.

All numbers are re-derived in exact `Rational{BigInt}` arithmetic; run `validate_propagation`
first to make sure the replications match the real solvers.

kwargs:
    max_vars_fv - fresh-variable budget for rung (iii') (default 10, as in the search)
    n_steps_opt - if given, add the optimised rungs (vii)/(viii) to the summary table
"""
function propagation_tex(Ws, bs, input_set=default_input_set(); max_vars_fv=10, n_steps_opt=nothing)
    parts = String[_tex_preamble(), _section_network(Ws, bs, input_set)]

    # (i)
    trace = interval_trace(Ws, bs, input_set)
    push!(parts, _section_interval(trace))

    # (ii)/(iii)/(iii')
    run_outer = _tex_dpnfv_run(Ws, bs, input_set; max_vars=0, method=:OuterBoundsNeurify)
    push!(parts, _section_dpnfv(Ws, bs, input_set;
                                run=merge(run_outer, (kind=:outer, max_vars=0, method=:OuterBoundsNeurify))))
    run_sep = _tex_dpnfv_run(Ws, bs, input_set; max_vars=0, method=:NeurifyRelax)
    push!(parts, _section_dpnfv(Ws, bs, input_set;
                                run=merge(run_sep, (kind=:sep, max_vars=0, method=:NeurifyRelax))))
    run_fv = _tex_dpnfv_run(Ws, bs, input_set; max_vars=max_vars_fv, method=:NeurifyRelax)
    push!(parts, _section_dpnfv(Ws, bs, input_set;
                                run=merge(run_fv, (kind=:fv, max_vars=max_vars_fv, method=:NeurifyRelax))))

    # (iv)
    net = make_network(Ws, bs)
    push!(parts, _section_zonotope(net, input_set))

    # (v)
    run_a = _tex_acrown0_run(Ws, bs, input_set)
    push!(parts, _section_acrown0(Ws, bs, run_a))

    # (vi) — single-input networks only
    run_p = nothing
    if length(low(input_set)) == 1
        run_p = _tex_pcrown0_run(Ws, bs, input_set)
        push!(parts, _section_pcrown0(run_p))
    else
        push!(parts, "\\paragraph{Rung (vi).} " *
              "The polynomial relaxation pipeline is displayed for single-input networks only.")
    end

    # summary values from the exact replications
    Z, _ = zono_propagate(net, input_set, Rational{BigInt})
    vals = (interval=Float64(trace[end].u_post[1]),
            outer=Float64(run_outer.ub[1]), sep=Float64(run_sep.ub[1]), fv=Float64(run_fv.ub[1]),
            zono=Float64(zono_upper_bound(Z)[1]),
            acrown0=Float64(run_a.ubs[end][1]),
            pcrown0=run_p === nothing ? Float64(evaluate_crown_init(Ws, bs, input_set).ub_pcrown0) :
                    Float64(run_p.ub[1]))
    opt_vals = nothing
    if n_steps_opt !== nothing
        o = evaluate_crown_opt(Ws, bs, input_set; n_steps=n_steps_opt)
        opt_vals = (acrown_opt=o.ub_acrown_opt, pcrown_opt=o.ub_pcrown_opt)
    end
    push!(parts, _section_summary(vals; n_steps_opt=n_steps_opt, opt_vals=opt_vals))

    push!(parts, _tex_postamble())
    return join(parts, "\n")
end

"""
Write the propagation TeX document for `Ws, bs` on `input_set` to `path`.
"""
function write_propagation_tex(path, Ws, bs, input_set=default_input_set(); kwargs...)
    write(path, propagation_tex(Ws, bs, input_set; kwargs...))
    return path
end

"""
Write the propagation TeX document for registry candidate `idx` (from `candidates.jld2`).

The input set is reconstructed from the stored `input_c`. Default path:
`propagation_idx<idx>.tex` next to this file.
"""
function write_propagation_tex(idx::Int; path=nothing, kwargs...)
    c = get_saved_candidate(idx)
    iset = input_set_of(c)
    p = path === nothing ? joinpath(@__DIR__, "propagation_idx$(idx).tex") : path
    return write_propagation_tex(p, c.Ws, c.bs, iset; kwargs...)
end
