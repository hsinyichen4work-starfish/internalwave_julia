# depth_slice.jl
#
# Extract a horizontal field at one fixed physical depth (e.g. 1 m, 10 m)
# from a sigma-coordinate 3D field, by vertical-only interpolation at
# each horizontal grid point. Unlike interp_3d.jl, this does NOT change
# horizontal (lon/lat) position — same grid in, same grid out — so it
# needs no triangulation/nearest-neighbor packages, just interp_1d.jl.
#
# z, F must have matching shape (M, L, N): M x L horizontal grid, N
# vertical levels last — matching the (xi_rho, eta_rho, s_rho) layout
# used after permutedims(zlevs3(...), (2,3,1)) in check_zeta.jl. z must
# be ascending (deepest first) at every column.
#
# target_depth is a physical z value in meters, negative below the
# surface (e.g. -1.0 for 1 m depth, -10.0 for 10 m depth) — matching
# ROMS's z convention (z=0 at the surface).

include(joinpath(@__DIR__, "interp_1d.jl"))
using .Interp1D

function slice_at_depth(z::AbstractArray{<:Real,3}, F::AbstractArray{<:Real,3}, target_depth::Real)
    M, L, N = size(z)
    out = fill(NaN32, M, L)
    for j in 1:L, i in 1:M
        out[i, j] = interp_1d(view(z, i, j, :), view(F, i, j, :), [target_depth])[1]
    end
    return out
end
