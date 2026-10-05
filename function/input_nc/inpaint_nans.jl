# inpaint_nans.jl
#
# inpaint_nans (John D'Errico, release 2), method 4 only — the one this
# pipeline uses — plus MATLAB's fillmissing(...,'linear',dim,'EndValues','nearest').

"""
    inpaint_nans(A, method = 4; cache = nothing) -> B

In-paint over NaNs with the spring metaphor (method 4): springs of zero
nominal length connect every NaN node to its row/column neighbors, and
the NaN values are the least-squares equilibrium, so extrapolation is as
a constant function.

Like the MATLAB original, an N-D array is treated as the 2D array
`reshape(A, size(A,1), :)` and `B` is returned in that 2D shape — so for
an (N,Mp,Lp) field, "column" neighbors are neighbors along Mp, running on
from one Lp index into the next.

MATLAB solves the rectangular spring system with `\\` (sparse QR). Here
the same least-squares problem is solved through its normal equations
(a graph Laplacian) with a sparse Cholesky factorization, which gives the
same minimizer far more cheaply; sparse QR is kept as the fallback.

The factorization only depends on where the NaNs are. Pass a Dict as
`cache` when in-painting many fields with the same NaN pattern (temp and
salt, every time step of a boundary): it is then factorized once and
reused for as long as the pattern stays the same.
"""
function inpaint_nans(A::AbstractArray{<:Real}, method::Integer = 4; cache::Union{Nothing,AbstractDict} = nothing)
    method == 4 || error("inpaint_nans: only method 4 (springs) is implemented")
    n = size(A, 1)
    m = length(A) ÷ n
    B = Float64.(reshape(A, n * m))

    isnanB = isnan.(B)
    nan_count = count(isnanB)
    (nan_count == 0 || nan_count == n * m) && return reshape(B, n, m)   # nothing to fill / nothing to fill from

    if cache !== nothing && get(cache, :n, 0) == n && get(cache, :pattern, nothing) == isnanB
        sys = cache[:system]
    else
        sys = _spring_system(isnanB, n, m)
        if cache !== nothing
            cache[:n], cache[:pattern], cache[:system] = n, isnanB, sys
        end
    end

    B[isnanB] .= 0.0
    rhs = sys.K * B                    # sum of the known neighbors of every NaN node
    B[sys.nan_list] = sys.F === nothing ? _inpaint_springs_qr(B, n, m, sys.nan_list, isnanB) : sys.F \ rhs
    return reshape(B, n, m)
end

# Normal equations of the spring system for a given NaN pattern. Every
# spring between a NaN node p and a neighbor q contributes (x_p - x_q)^2,
# once per pair: L holds the NaN-NaN couplings, K picks the known neighbors.
function _spring_system(isnanB::AbstractVector{Bool}, n::Integer, m::Integer)
    nan_list = findall(isnanB)
    nan_count = length(nan_list)
    unknown = zeros(Int, n * m)       # node -> position among the unknowns
    unknown[nan_list] = 1:nan_count

    I = Int[]
    J = Int[]
    V = Float64[]
    KI = Int[]
    KJ = Int[]
    diagv = zeros(Float64, nan_count)
    sizehint!(I, 4 * nan_count)
    sizehint!(J, 4 * nan_count)
    sizehint!(V, 4 * nan_count)
    for (u, p) in enumerate(nan_list)
        r = (p - 1) % n + 1
        c = (p - 1) ÷ n + 1
        for (dp, ok) in ((-1, r > 1), (1, r < n), (-n, c > 1), (n, c < m))
            ok || continue
            q = p + dp
            diagv[u] += 1.0
            if isnanB[q]
                push!(I, u)
                push!(J, unknown[q])
                push!(V, -1.0)
            else
                push!(KI, u)
                push!(KJ, q)
            end
        end
    end
    L = sparse(I, J, V, nan_count, nan_count) + spdiagm(0 => diagv)
    K = sparse(KI, KJ, ones(length(KI)), nan_count, n * m)

    F = try
        cholesky(Symmetric(L))
    catch err
        err isa PosDefException || rethrow()
        @warn "inpaint_nans: some NaNs have no known value to connect to; falling back to sparse QR"
        nothing
    end
    return (nan_list = nan_list, K = K, F = F)
end

# The literal MATLAB formulation: one row per spring, solved with `\`.
# B holds the known values (anything at the NaN nodes flagged in isnanB).
function _inpaint_springs_qr(B, n, m, nan_list, isnanB)
    unknown = zeros(Int, n * m)
    unknown[nan_list] = 1:length(nan_list)
    I = Int[]
    J = Int[]
    V = Float64[]
    rhs = Float64[]
    row = 0
    for p in nan_list
        r = (p - 1) % n + 1
        c = (p - 1) ÷ n + 1
        for (dp, ok) in ((-1, r > 1), (1, r < n), (-n, c > 1), (n, c < m))
            ok || continue
            q = p + dp
            isnanB[q] && q < p && continue   # delete replicate springs
            row += 1
            lo, hi = minmax(p, q)            # +1 on the lower node, -1 on the higher
            b = 0.0
            for (node, s) in ((lo, 1.0), (hi, -1.0))
                if isnanB[node]
                    push!(I, row)
                    push!(J, unknown[node])
                    push!(V, s)
                else
                    b -= s * B[node]
                end
            end
            push!(rhs, b)
        end
    end
    springs = sparse(I, J, V, row, length(nan_list))
    return springs \ rhs
end

"""
    fillmissing_linear!(X, dim) -> X

MATLAB `fillmissing(X,'linear',dim,'EndValues','nearest')`: fill NaNs by
linear interpolation along `dim`; leading/trailing NaNs take the nearest
non-NaN value. Vectors that are all NaN are left alone.
"""
function fillmissing_linear!(X::AbstractArray{<:AbstractFloat}, dim::Integer)
    any(isnan, X) || return X
    for v in eachslice(X; dims = Tuple(d for d in 1:ndims(X) if d != dim))
        good = findall(!isnan, v)
        (isempty(good) || length(good) == length(v)) && continue
        v[1:good[1]-1] .= v[good[1]]
        v[good[end]+1:end] .= v[good[end]]
        for g in 1:length(good)-1
            a, b = good[g], good[g+1]
            for k in a+1:b-1
                v[k] = v[a] + (v[b] - v[a]) * (k - a) / (b - a)
            end
        end
    end
    return X
end
