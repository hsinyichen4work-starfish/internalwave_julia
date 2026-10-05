# hv_coef.jl
#
# 3D (horizontal + vertical) interpolation from parent z-levels to child
# s-levels. Ported from get_hv_coef.m, get_v_coef.m and get_1d_coef.m
# (c) 2007 Jeroen Molemaker.
#
# MATLAB builds one sparse matrix A = Av*Ah and applies it with
#   Fc = reshape(A*reshape(Fp,Np*Mp*Lp,1),Nc,Mc,Lc)
# For a 900 m child chunk that matrix has ~1e8 nonzeros, so here the same
# coefficients are kept in factored form (HVCoef) and applied column by
# column with `apply_hv_coef` — same numbers, a fraction of the memory.

"""
Coefficients of the parent (Np,Mp,Lp) -> child (Nc,Mc,Lc) interpolation.

The horizontal step (Ah) maps the parent onto Np+1 intermediate levels at
every child column: intermediate level k >= 2 is parent level k-1, and
level 1 is the extra point added below the deepest parent level. The
vertical step (Av) then interpolates each intermediate column onto the
child levels.
"""
struct HVCoef
    Np::Int
    Mp::Int
    Lp::Int
    Nc::Int
    Mc::Int
    Lc::Int
    elem2d::Array{Int,3}       # (Mc,Lc,3) parent points for intermediate levels 2:Np+1
    coef2d::Array{Float64,3}
    elemb::Array{Int,3}        # (Mc,Lc,3) same, for the extra bottom level 1
    coefb::Array{Float64,3}
    elem1d::Array{Int,4}       # (Nc,Mc,Lc,2) intermediate levels each child level uses
    coef1d::Array{Float64,4}
end

# get_1d_coef.m for one column, written into the (Nc,2) slices coef/elem.
# zp, zc must be ascending; points outside zp are nearest-neighbor filled.
function _coef1d!(coef, elem, zp::AbstractVector, zc::AbstractVector)
    Np = length(zp)
    ip = 1
    for ic in eachindex(zc)
        while zp[ip] < zc[ic] && ip < Np
            ip += 1
        end
        if ip == 1 || zp[ip] < zc[ic]
            coef[ic, 1] = 1.0
            elem[ic, 1] = ip
            coef[ic, 2] = 0.0
            elem[ic, 2] = 1
            continue
        end
        alp = (zc[ic] - zp[ip-1]) / (zp[ip] - zp[ip-1])
        coef[ic, 1] = alp
        elem[ic, 1] = ip
        coef[ic, 2] = 1 - alp
        elem[ic, 2] = ip - 1
    end
    return nothing
end

"""
    get_hv_coef(zp, zc, coef2d, elem2d, lonp, latp, lonc, latc) -> A::HVCoef

- `zp` :: (Np,Mp,Lp) parent z-values, `zc` :: (Nc,Mc,Lc) child z-values;
  both must be in ascending order along k
- `coef2d, elem2d` from `get_tri_coef`

Use like `Fc = apply_hv_coef(A, Fp)` with `Fp` shaped (Np,Mp,Lp).
"""
function get_hv_coef(zp::AbstractArray{<:Real,3}, zc::AbstractArray{<:Real,3},
                     coef2d::AbstractArray{<:Real,3}, elem2d::AbstractArray{<:Integer,3},
                     lonp, latp, lonc, latc)
    Np, Mp, Lp = size(zp)
    Nc, Mc, Lc = size(zc)

    lon0 = sum(lonc) / length(lonc)
    lat0 = sum(latc) / length(latc)
    xp, yp = gnomonic(lonp, latp, lon0, lat0)
    xc, yc = gnomonic(lonc, latc, lon0, lat0)

    # This is unfortunately a somewhat ugly fix to deal with gross
    # differences in topography between the 2 grids. It adds an extra
    # vertical point taken from the intermediate (horizontal)
    # interpolation, so Ah works from (Np,Mp,Lp) to (Np+1,Mc,Lc).
    zp_low = zp[1, :, :]
    zc_low = zc[1, :, :]
    zt_low = apply_tri_coef(elem2d, coef2d, zp_low)

    mismatch = 10000   # When the child grid's lowest point at (i,j) is more
                       # than 'mismatch' below the interpolated parent point,
                       # we put an extra point below it by means of nearest
                       # neighbor horizontal extrapolation.

    hmax = minimum(zp_low)
    nlev = round(Int, abs(hmax / mismatch), RoundNearestTiesAway)
    lev_trees = Dict{Int,Tuple{KDTree,Vector{Int}}}()   # built only if actually needed

    elemb = copy(elem2d)     # trivial extra point: same as the deepest parent level
    coefb = Float64.(coef2d)
    println("--- fixing parent/child topo mismatch (may take a while...)")
    for j in 1:Lc, i in 1:Mc
        if zt_low[i, j] > zc_low[i, j] + mismatch   # we need an extra point below zt_low
            nlev >= 1 || error("get_hv_coef: topo mismatch found but no parent level deep enough to fill from")
            lev = round(Int, zc_low[i, j] * (nlev + 1) / hmax, RoundNearestTiesAway)
            lev = min(nlev, max(1, lev))
            tree, deep = get!(lev_trees, lev) do
                depth = lev * hmax / (nlev + 1)
                deep = findall(<=(depth), vec(zp_low))   # the shallow points are out of reach
                KDTree(permutedims(hcat(vec(xp)[deep], vec(yp)[deep]))), deep
            end
            idx, _ = nn(tree, [xc[i, j], yc[i, j]])
            elemb[i, j, :] .= deep[idx]
            coefb[i, j, 1] = 1.0
            coefb[i, j, 2] = 0.0
            coefb[i, j, 3] = 0.0
        end
    end
    # End of adding a lower layer

    # vertical coefficients, column by column (get_v_coef.m)
    zp2 = reshape(zp, Np, Mp * Lp)
    elem1d = ones(Int, Nc, Mc, Lc, 2)
    coef1d = zeros(Float64, Nc, Mc, Lc, 2)
    Threads.@threads for j in 1:Lc
        zt = zeros(Float64, Np + 1)
        for i in 1:Mc
            fill!(zt, 0.0)
            for c in 1:3
                w, e = coef2d[i, j, c], elem2d[i, j, c]
                wb, eb = coefb[i, j, c], elemb[i, j, c]
                wb == 0 || (zt[1] += wb * zp2[1, eb])
                w == 0 && continue
                for k in 1:Np
                    zt[k+1] += w * zp2[k, e]
                end
            end
            zt[1] -= 0.1   # avoid double z point in every column
            _coef1d!(view(coef1d, :, i, j, :), view(elem1d, :, i, j, :), zt, view(zc, :, i, j))
        end
    end

    println("--- all done!!")
    return HVCoef(Np, Mp, Lp, Nc, Mc, Lc, collect(elem2d), Float64.(coef2d), elemb, coefb, elem1d, coef1d)
end

"""
    apply_hv_coef(A, Fp) -> Fc

Interpolate a parent field `Fp` (Np,Mp,Lp), or the same thing already
flattened to (Np, Mp*Lp), onto the child grid; returns (Nc,Mc,Lc).
MATLAB: `reshape(A*reshape(Fp,Np*Mp*Lp,1),Nc,Mc,Lc)`.
"""
function apply_hv_coef(A::HVCoef, Fp::AbstractArray{<:Real})
    length(Fp) == A.Np * A.Mp * A.Lp ||
        error("apply_hv_coef: field has $(length(Fp)) values, expected $(A.Np)x$(A.Mp)x$(A.Lp)")
    F = reshape(Fp, A.Np, A.Mp * A.Lp)
    Np, Nc = A.Np, A.Nc
    Fc = zeros(Float64, Nc, A.Mc, A.Lc)
    Threads.@threads for j in 1:A.Lc
        ft = zeros(Float64, Np + 1)
        for i in 1:A.Mc
            fill!(ft, 0.0)
            # zero coefficients are skipped: they are not stored in MATLAB's sparse A either
            for c in 1:3
                w, e = A.coef2d[i, j, c], A.elem2d[i, j, c]
                wb, eb = A.coefb[i, j, c], A.elemb[i, j, c]
                wb == 0 || (ft[1] += wb * F[1, eb])
                w == 0 && continue
                for k in 1:Np
                    ft[k+1] += w * F[k, e]
                end
            end
            for kc in 1:Nc
                acc = 0.0
                for c in 1:2
                    w = A.coef1d[kc, i, j, c]
                    w == 0 || (acc += w * ft[A.elem1d[kc, i, j, c]])
                end
                Fc[kc, i, j] = acc
            end
        end
    end
    return Fc
end
