# vorticity_cal.jl
#
# Relative vorticity on the ROMS PSI grid. Ported from vorticity_cal.m,
# auto-detecting whether u/v were passed on the RHO grid (and
# converting via rho2u/rho2v) or already on their native U/V grids.
#
# Unlike the MATLAB version, no explicit time loop is needed here —
# rho2u/rho2v and the finite-difference below all broadcast over any
# trailing dimensions (depth, time) automatically, same as everywhere
# else in this project.

include(joinpath(@__DIR__, "rho2uvp.jl"))

# psi point (i,j) is surrounded by 4 rho points: (i,j),(i+1,j),(i,j+1),(i+1,j+1)
function pmpn2psi(pm, pn)
    dx_psi = 1 ./ rho2p(pm)
    dy_psi = 1 ./ rho2p(pn)
    return dx_psi, dy_psi
end

function vorticity_cal_fast(u, v, dx_psi, dy_psi)
    nd_v = ndims(v)
    trailing_v = ntuple(_ -> Colon(), nd_v - 1)
    dvdx = (view(v, 2:size(v, 1), trailing_v...) .- view(v, 1:size(v, 1)-1, trailing_v...)) ./ dx_psi

    nd_u = ndims(u)
    trailing_u = ntuple(_ -> Colon(), nd_u - 2)
    dudy = (view(u, :, 2:size(u, 2), trailing_u...) .- view(u, :, 1:size(u, 2)-1, trailing_u...)) ./ dy_psi

    return dvdx .- dudy
end

"""
    vorticity_cal(u, v, pm, pn) -> vor_psi

u, v: EITHER on their native U-/V-grids, OR already on the RHO grid
(auto-converted via rho2u/rho2v). pm, pn: (xi_rho, eta_rho) inverse
grid spacing. Returns relative vorticity on the PSI grid.
"""
function vorticity_cal(u, v, pm, pn)
    Mp, Lp = size(pm)
    dx_psi, dy_psi = pmpn2psi(pm, pn)

    u_native = (size(u, 1) == Mp && size(u, 2) == Lp) ? rho2u(u) : u
    v_native = (size(v, 1) == Mp && size(v, 2) == Lp) ? rho2v(v) : v

    return vorticity_cal_fast(u_native, v_native, dx_psi, dy_psi)
end
