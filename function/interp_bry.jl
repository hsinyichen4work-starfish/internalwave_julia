"""
    interp_bry.jl

Case (3): from (x,y,z) to (x',z') — interpolate a full 3D parent field
onto a 1D string of boundary points (e.g. a ROMS open-boundary
section), each with its own vertical column. This is exactly the
`get_hv_coef.m` machinery used inside `h2r_bry_hv.m`, just with the
child horizontal grid collapsed from (Mc,Lc) down to a single row of
`Nb` points.

Include this file directly; it pulls in interp_3d.jl (which itself
pulls in interp_1d.jl / interp_2d.jl), so you don't need to include
those separately.
"""
module InterpBry

include(joinpath(@__DIR__, "interp_3d.jl"))
using .Interp3D

export get_bry_coef, apply_bry_coef, interp_bry

"""
    get_bry_coef(lonp, latp, zp, lonb, latb, zb; maskp=nothing) -> A

- `lonb, latb` :: (Nb,)     boundary horizontal positions
- `zb`         :: (Nc, Nb)  boundary vertical levels at each point
                  (or (Nc,) if the same at every boundary point)

Returns sparse `A` such that `Fc = reshape(A*vec(Fp), Nc, Nb)` for a
parent field `Fp` shaped `(Np,Mp,Lp)`.
"""
function get_bry_coef(lonp, latp, zp, lonb::AbstractVector, latb::AbstractVector, zb; maskp=nothing)
    Nb = length(lonb)
    lonc = reshape(lonb, Nb, 1)
    latc = reshape(latb, Nb, 1)
    zc = ndims(zb) == 1 ? zb : reshape(zb, size(zb, 1), Nb, 1)
    return Interp3D.get_3d_coef(lonp, latp, zp, lonc, latc, zc; maskp=maskp)
end

"""
    apply_bry_coef(A, Fp, Nc, Nb) -> Fc

Apply a precomputed operator `A` (from `get_bry_coef`) to a parent
field `Fp` shaped `(Np,Mp,Lp)`.
"""
function apply_bry_coef(A, Fp, Nc, Nb)
    reshape(A * vec(Fp), Nc, Nb)
end

"""
    interp_bry(lonp, latp, zp, Fp, lonb, latb, zb; maskp=nothing) -> Fc

One-shot boundary interpolation, returning an `(Nc, Nb)` array.
"""
function interp_bry(lonp, latp, zp, Fp, lonb, latb, zb; maskp=nothing)
    Nb = length(lonb)
    Nc = ndims(zb) == 1 ? length(zb) : size(zb, 1)
    A = get_bry_coef(lonp, latp, zp, lonb, latb, zb; maskp=maskp)
    return apply_bry_coef(A, Fp, Nc, Nb)
end

end # module