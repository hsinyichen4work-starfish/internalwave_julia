"""
    interp_3d.jl

Case (2): from (x,y,z) to (x',y',z') — combined horizontal+vertical
3D interpolation, direct analog of `get_hv_coef.m` + `get_v_coef.m`
(built from `interp_2d.jl` + `interp_1d.jl`, exactly like the MATLAB
pipeline chains `get_tri_coef` -> `get_hv_coef` -> `get_v_coef` ->
`get_1d_coef`).

The topography-mismatch "extra deep level" hack in the original
`get_hv_coef.m` (for when parent/child bathymetry disagree a lot at a
boundary) is intentionally omitted for clarity — `get_1d_coef`'s
nearest-neighbor clamping at the top/bottom already keeps things
well-defined outside the parent's z-range. Add that refinement back in
if you're regridding across very different bathymetries and see
artifacts near the seabed.

Include this file directly; it pulls in interp_1d.jl / interp_2d.jl
itself, so you don't need to include those separately.
"""
module Interp3D

include(joinpath(@__DIR__, "interp_1d.jl"))
include(joinpath(@__DIR__, "interp_2d.jl"))
using .Interp1D
using .Interp2D
using SparseArrays

export get_3d_coef, apply_3d_coef, interp_3d

"""
    get_3d_coef(lonp, latp, zp, lonc, latc, zc; maskp=nothing) -> A

Shapes:
- `lonp,latp` :: (Mp,Lp)    parent horizontal grid
- `zp`        :: (Np,Mp,Lp) parent depth/sigma levels (ascending in k),
                 or (Np,) if the same at every parent point (e.g. HYCOM
                 z-levels)
- `lonc,latc` :: (Mc,Lc)    child horizontal grid
- `zc`        :: (Nc,Mc,Lc) child depth/sigma levels (ascending in k),
                 or (Nc,)

Returns a sparse operator `A` (size `Nc*Mc*Lc` x `Np*Mp*Lp`) such that,
for a parent field `Fp` shaped `(Np,Mp,Lp)`:

    Fc = reshape(A * vec(Fp), Nc, Mc, Lc)

Build once, reuse `A` for every 3D variable on that grid pair (temp,
salt, u, v, ...) — this is the whole point of the coefficient/apply
split used throughout the original MATLAB pipeline.
"""
function get_3d_coef(lonp, latp, zp, lonc, latc, zc; maskp=nothing)
    Mp, Lp = size(lonp)
    Mc, Lc = size(lonc)
    Nph = Mp * Lp
    Nch = Mc * Lc
    Np = ndims(zp) == 1 ? length(zp) : size(zp, 1)
    Nc = ndims(zc) == 1 ? length(zc) : size(zc, 1)

    elem2d, coef2d, _ = Interp2D.get_2d_coef(lonp, latp, lonc, latc; maskp=maskp)
    # elem2d/coef2d: (Nch,3) parent-point indices/weights per child horiz. point

    # --- horizontal operator Ah: (Np levels x Nph parent) -> (Np levels x Nch child) ---
    I = Int[]; J = Int[]; V = Float64[]
    for k in 1:Np, ic in 1:Nch
        row0 = k + (ic - 1) * Np
        for c in 1:3
            p = elem2d[ic, c]
            push!(I, row0); push!(J, k + (p - 1) * Np); push!(V, coef2d[ic, c])
        end
    end
    Ah = sparse(I, J, V, Np * Nch, Np * Nph)

    # --- intermediate z-levels on the child horizontal grid, still parent's Np levels ---
    zp_flat = ndims(zp) == 1 ? repeat(zp, 1, Nph) : reshape(zp, Np, Nph)
    z_interm = reshape(Ah * vec(zp_flat), Np, Nch)

    zc_flat = ndims(zc) == 1 ? repeat(zc, 1, Nch) : reshape(zc, Nc, Nch)

    # --- vertical operator Av: per-column 1D coefficients (get_v_coef.m equivalent) ---
    I2 = Int[]; J2 = Int[]; V2 = Float64[]
    for col in 1:Nch
        coef1d, elem1d = Interp1D.get_1d_coef(view(z_interm, :, col), view(zc_flat, :, col))
        for kc in 1:Nc
            row = kc + (col - 1) * Nc
            for c in 1:2
                k = elem1d[kc, c]
                push!(I2, row); push!(J2, k + (col - 1) * Np); push!(V2, coef1d[kc, c])
            end
        end
    end
    Av = sparse(I2, J2, V2, Nc * Nch, Np * Nch)

    return Av * Ah
end

"""
    apply_3d_coef(A, Fp, Nc, Mc, Lc) -> Fc

Apply a precomputed operator `A` (from `get_3d_coef`) to a parent field
`Fp` shaped `(Np,Mp,Lp)`.
"""
function apply_3d_coef(A, Fp, Nc, Mc, Lc)
    reshape(A * vec(Fp), Nc, Mc, Lc)
end

"""
    interp_3d(lonp, latp, zp, Fp, lonc, latc, zc; maskp=nothing) -> Fc

One-shot 3D interpolation. For repeated use on the same grid pair,
call `get_3d_coef` once and reuse it via `apply_3d_coef`.
"""
function interp_3d(lonp, latp, zp, Fp, lonc, latc, zc; maskp=nothing)
    Mc, Lc = size(lonc)
    Nc = ndims(zc) == 1 ? length(zc) : size(zc, 1)
    A = get_3d_coef(lonp, latp, zp, lonc, latc, zc; maskp=maskp)
    return apply_3d_coef(A, Fp, Nc, Mc, Lc)
end

end # module