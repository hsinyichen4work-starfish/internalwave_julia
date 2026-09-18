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

# Re-including this file (e.g. re-running a script in a live REPL) would
# otherwise redefine the Interp1D module and `using` it again, leaving
# two distinct modules both bound to Main's `interp_1d` — Julia then
# refuses to resolve it ("both Interp1D and Interp1D export interp_1d").
# Guarding the include makes it load-once-per-session instead.
isdefined(@__MODULE__, :Interp1D) || include(joinpath(@__DIR__, "interp_1d.jl"))
using .Interp1D

function slice_at_depth(z::AbstractArray{<:Real,3}, F::AbstractArray{<:Real,3}, target_depth::Real)
    M, L, N = size(z)
    out = fill(NaN32, M, L)
    for j in 1:L, i in 1:M
        zcol = view(z, i, j, :)
        out[i, j] = interp_1d(zcol, view(F, i, j, :), [target_depth])[1]
        # z is ascending: zcol[1] = deepest (bottom). Anything requested deeper
        # than that is below the seafloor here — overwrite with NaN regardless
        # of whatever interp_1d extrapolated. No check needed at the shallow
        # end: interp_1d's own extrapolation there is exactly the "1 m" value
        # we want, since the top sigma layer is always valid ocean.
        if target_depth < zcol[1]
            out[i, j] = NaN32
        end
    end
    return out
end

# Same idea as slice_at_depth, but for NCOM's zm3/kb vertical grid instead
# of a ROMS sigma-coordinate z. Two things differ from the ROMS case:
#  - zm3 is stored shallow-to-deep (descending z: ~-0.5 m at k=1, more
#    negative going down) — the opposite of the ascending order interp_1d
#    requires — so each column gets reversed rather than being ascending
#    already like zlevs3's output.
#  - not every column has all N levels: kb[i,j] gives the number of valid
#    levels at that horizontal point, and zm3/F are missing/NaN beyond
#    it (below the seafloor there). So each column is trimmed to
#    1:kb[i,j] first — passing the fixed-length column straight through
#    (reversed, with missing/NaN left at the front) would break
#    interp_1d's ascending scan, which assumes no missing/invalid entries.
function slice_at_depth_ncom(zm3::AbstractArray{<:Union{Missing,Real},3}, kb::AbstractMatrix,
    F::AbstractArray{<:Real,3}, target_depth::Real)
M, L, N = size(zm3)
out = fill(NaN32, M, L)
for j in 1:L, i in 1:M
    k = kb[i, j]
    (ismissing(k) || k < 2) && continue   # no usable water column here
    zcol = Float64.(view(zm3, i, j, k:-1:1))
    fcol = Float64.(view(F, i, j, k:-1:1))
    out[i, j] = interp_1d(zcol, fcol, [target_depth])[1]
    # zcol is ascending after the k:-1:1 reversal: zcol[1] = deepest valid
    # point in this column (originally at index k). Anything requested
    # deeper than that is below the seafloor here — mask it out, same
    # reasoning as slice_at_depth. No check needed at the shallow end for
    # the same reason as before (shallowest valid cell is always real ocean).
    if target_depth < zcol[1]
        out[i, j] = NaN32
    end
end
    return out
end
