# rho2uvp.jl
#
# Averages a RHO-point field onto U-, V-, or PSI-points, the same
# convention as ROMS's rho2u_3d.m / rho2v_3d.m — plain averaging, not
# recomputing zlevs3 with separately-averaged h/zeta. Works on any
# array whose first two dimensions are (xi_rho, eta_rho), regardless of
# how many trailing dimensions it has (depth levels, time, ...) — so
# the same functions apply to h (2D), z (3D or 4D), temp, etc.

function rho2u(f::AbstractArray)
    nd = ndims(f)
    trailing = ntuple(_ -> Colon(), nd - 1)
    return 0.5 .* (view(f, 2:size(f, 1), trailing...) .+ view(f, 1:size(f, 1)-1, trailing...))
end

function rho2v(f::AbstractArray)
    nd = ndims(f)
    lead = ntuple(_ -> Colon(), 1)
    trailing = ntuple(_ -> Colon(), nd - 2)
    return 0.5 .* (view(f, lead..., 2:size(f, 2), trailing...) .+ view(f, lead..., 1:size(f, 2)-1, trailing...))
end

rho2p(f::AbstractArray) = rho2v(rho2u(f))
