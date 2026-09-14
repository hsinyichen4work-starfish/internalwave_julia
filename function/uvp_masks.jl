# uvp_masks.jl
#
# Ported from uvp_masks.m (ROMS/TOMS Group) — computes the Land/Sea
# mask on U-, V-, and PSI-points from the RHO-point mask.
#
# rmask :: mask on RHO-points, size (Lp, Mp)
# returns (umask, vmask, pmask), sizes (Lp-1, Mp), (Lp, Mp-1), (Lp-1, Mp-1)

function uvp_masks(rmask)
    Lp, Mp = size(rmask)
    L = Lp - 1
    M = Mp - 1

    umask = rmask[2:Lp, 1:Mp] .* rmask[1:L, 1:Mp]
    vmask = rmask[1:Lp, 2:Mp] .* rmask[1:Lp, 1:M]
    pmask = rmask[1:L, 1:M] .* rmask[2:Lp, 1:M] .* rmask[1:L, 2:Mp] .* rmask[2:Lp, 2:Mp]

    return umask, vmask, pmask
end
