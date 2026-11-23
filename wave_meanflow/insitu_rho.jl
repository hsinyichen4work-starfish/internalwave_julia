#= insitu_rho.jl
In-situ density anomaly of the model's split EOS (ROMS UCLA rho_eos.F, NONLIN_EOS + SPLIT_EOS, DUKO_2001 off):
  rho_is = rho1 + qp1 dpth (1 − qp2 dpth),   dpth = −z_r = −(ζ + (ζ + h) z0)   [kg m-3, minus rho0]
  qp1    = 0.000649 + 0.1 (K00 rho1 − rho0 K0(T, S)) / (K00 (K00 + K0(T, S))),   qp2 = 1.72e-5 (eos_vars.F)
  rho1 = his "rho" (density at the surface pressure minus rho0), T = his temp, S = his salt.
This is the density the model's pressure gradient (prsgrd.F, rhoS / rhoA in rho_eos.F) integrates. rho1 alone misses
the compressible part qp1 dpth, whose wave perturbation is ~15% of rho_wave below 300 m and ~100% below 2000 m
(tests/test_insitu_rho.jl, deep tile, window 1): modal flux +14% (bp9_15).
Pseudo-variable "rhoi" in the pipeline: window mean rhoi_window (step 2), rhoi_wave and p_wave from it (steps 5-6).
=#

const EOS_K00 = 19092.56
const EOS_QP2 = 0.0000172

# secant bulk modulus K0(T, S) minus K00 [bar], as rho_eos.F (K01..K04, K10..K13, KS0..KS2)
@inline function eos_K0(T, S)
    sS = sqrt(max(S, 0.0))
    T * (209.8925 + T * (-3.041638 + T * (-1.852732e-3 + T * -1.361629e-5))) +
    S * (104.4077 + T * (-6.500517 + T * (0.1553190 + T * 2.326469e-4)) + sS * (-5.587545 + T * (0.7390729 + T * -1.909078e-2)))
end

# out[i, j, k] = rho_is of one record; r1, T, S (nx, ny, N) of the record, ζ (nx, ny) of the record, h, z0 on the same points
function rho_insitu!(out, r1, T, S, ζ, h, z0, rho0)
    nx, ny, N = size(out)
    @inbounds for k in 1:N, j in 1:ny, i in 1:nx
        ρ1 = Float64(r1[i, j, k]); K0 = eos_K0(Float64(T[i, j, k]), Float64(S[i, j, k]))
        qp1 = 0.000649 + 0.1 * (EOS_K00 * ρ1 - rho0 * K0) / (EOS_K00 * (EOS_K00 + K0))
        dpth = -(ζ[i, j] + (ζ[i, j] + h[i, j]) * z0[i, j, k])
        out[i, j, k] = ρ1 + qp1 * dpth * (1 - EOS_QP2 * dpth)
    end
    return out
end

# his variable behind a pipeline variable ("rhoi" is computed from rho, temp, salt, zeta)
his_name(var) = var == "rhoi" ? "rho" : var
his_needs(var) = var == "rhoi" ? ["rho", "temp", "salt", "zeta"] : [var]

# one block of records kr of var on (ix, jy) from an open his file: the his variable, or for "rhoi" (rho, temp, salt)
read_his_block(d, var, ix, jy, kr) = var == "rhoi" ?
    (d["rho"].var[ix, jy, :, kr], d["temp"].var[ix, jy, :, kr], d["salt"].var[ix, jy, :, kr]) : d[var].var[ix, jy, :, kr]

# record kb of a block from read_his_block as a 3-D field (rhoi: computed into buf with the record's ζ)
record_of(blk, var, kb, buf, ζ, h, z0, rho0) = var == "rhoi" ?
    rho_insitu!(buf, view(blk[1], :, :, :, kb), view(blk[2], :, :, :, kb), view(blk[3], :, :, :, kb), ζ, h, z0, rho0) :
    view(blk, :, :, :, kb)
