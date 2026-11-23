# Test: does p_wave need the compressible part of the model's split EOS?
# The model (NONLIN_EOS + SPLIT_EOS, rho_eos.F, DUKO_2001 off) has in-situ density anomaly
#   rho_is = rho1 + qp1 dpth (1 - qp2 dpth),  dpth = -z_r,  qp2 = 1.72e-5 (eos_vars.F)
#   qp1 = 0.000649 + 0.1 (K00 rho1 - rho0 K0) / (K00 (K00 + K0)),  K0(T, S) as in rho_eos.F
# and the his "rho" is rho1. Steps 5-6 used rho1 only. Here, on one deep tile (tile 7) and window 1, band bp9_15:
#   current: rho_wave from rho1            -> p_wave -> bp9_15 -> p_bc -> p_n, flux H<p_n u_n>
#   in-situ: rho_wave from rho_is (same fixed-z method, its own window means) -> same chain
# u_n from the saved u_n_bp9_15 files (u does not depend on the EOS). Nothing is written to meanflowdata.
using NCDatasets, Printf, Statistics, Dates, DSP
include("/home/hchen54/internalwave_julia/function/butter_filters.jl")
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
for f in ("meanflow_window", "wave_perturb", "mode_projection", "wave_bands")
    include("/home/hchen54/internalwave_julia/wave_meanflow/$f.jl")
end
D = "/expanse/lustre/projects/uso102/hchen54/amazon_2700m_cmems_0801/after_joint/"
R = "/expanse/lustre/projects/uso102/hchen54/amazon_2700m_cmems_0801/meanflowdata/L72h_ov0.00/"
grid_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/roms_cmems_grd_2700m.nc"
win = window_setup(D, "rho"; tstart = DateTime(2022, 9, 15), tend = DateTime(2022, 10, 15), window_length = 72, verbose = false)
td = win.td; W = 1; band = BANDS[2]
G = wave_grid(win, D, R, "rho", grid_file)
tiles, _ = var_tiles(joinpath(D, td.files[1]), "rho")
q = findfirst(((ix, jy),) -> 200 in ix && 250 in jy, tiles); ix, jy = tiles[q]
h = G.h[ix, jy]; z0 = G.z0[ix, jy, :]; dz0 = G.dz0[ix, jy, :]; ζt = G.ζ[ix, jy, :]
nx, ny, N = size(z0); nt = td.nt
rho0 = NCDataset(d -> Float64(d.attrib["rho0"]), joinpath(D, td.files[1]))
@printf("tile %d: xi %d:%d, eta %d:%d, h %.0f..%.0f m, rho0 %.1f\n", q, first(ix), last(ix), first(jy), last(jy), extrema(h)..., rho0)

# ---- in-situ density anomaly series on the tile (model EOS), every record
const K00 = 19092.56; const qp2 = 0.0000172
K0(T, S) = (sS = sqrt(max(S, 0.0));
            T * (209.8925 + T * (-3.041638 + T * (-1.852732e-3 + T * -1.361629e-5))) +
            S * (104.4077 + T * (-6.500517 + T * (0.1553190 + T * 2.326469e-4)) + sS * (-5.587545 + T * (0.7390729 + T * -1.909078e-2))))
ρis = Array{Float32}(undef, nx, ny, N, nt)
t = @elapsed for (fi, recs) in records_by_file(td)
    ks = first.(recs); kr = minimum(ks):maximum(ks)
    r1, T, S = NCDataset(d -> (d["rho"].var[ix, jy, :, kr], d["temp"].var[ix, jy, :, kr], d["salt"].var[ix, jy, :, kr]), joinpath(D, td.files[fi]))
    for (k, n) in recs, kk in 1:N, j in 1:ny, i in 1:nx
        kb = k - first(kr) + 1
        ρ1 = Float64(r1[i, j, kk, kb])
        qp1 = 0.000649 + 0.1 * (K00 * ρ1 - rho0 * K0(T[i, j, kk, kb], S[i, j, kk, kb])) / (K00 * (K00 + K0(T[i, j, kk, kb], S[i, j, kk, kb])))
        dpth = -(ζt[i, j, n] + (ζt[i, j, n] + h[i, j]) * z0[i, j, kk])
        ρis[i, j, kk, n] = ρ1 + qp1 * dpth * (1 - qp2 * dpth)
    end
end
@printf("in-situ density series: %.0f s, %.1f GB; at (200,250) bottom: rho1 %.3f, rho_is %.3f (compressible part %.2f kg/m3)\n", t, sizeof(ρis) / 1e9,
        NCDataset(d -> d["rho"].var[200, 250, 1, 1], joinpath(D, td.files[1])), ρis[200 - first(ix) + 1, 250 - first(jy) + 1, 1, 1],
        ρis[200 - first(ix) + 1, 250 - first(jy) + 1, 1, 1] - NCDataset(d -> d["rho"].var[200, 250, 1, 1], joinpath(D, td.files[1])))

# ---- rho_wave at fixed z: current (rho1, saved rho_window) and in-situ (own window means on s-levels)
wts = bg_weights(win, "interp")
xbar, zbar = tile_means(win, R, "rho", G, ix, jy)
x_cur = wave_series_tile("rho", ix, jy, win, D, G, xbar, zbar, wts)
xbar_is = [dropdims(mean(ρis[:, :, :, win.idx[w]], dims = 4), dims = 4) for w in 1:win.nwin]
x_is = similar(ρis); bg1 = Array{Float32}(undef, nx, ny, N); bg2 = similar(bg1)
for n in 1:nt
    w1, w2, a = wts[n]; ζ = view(ζt, :, :, n)
    bg_at_depth!(bg1, ζ, h, z0, zbar[w1], xbar_is[w1])
    a > 0 && (bg_at_depth!(bg2, ζ, h, z0, zbar[w2], xbar_is[w2]); bg1 .= (1 - a) .* bg1 .+ a .* bg2)
    @views x_is[:, :, :, n] .= ρis[:, :, :, n] .- bg1
end
ρis = nothing; GC.gc()
rs = win.idx[W]
zr1 = [ζt[i, j, rs[1]] + (ζt[i, j, rs[1]] + h[i, j]) * z0[i, j, k] for i in 1:nx, j in 1:ny, k in 1:N]
println("rho_wave (all frequencies, window $W): rms current | rms in-situ | rms difference [kg/m3] by depth")
for (lab, lo, hi) in (("0-300 m", -300, 0), ("300-1000 m", -1000, -300), ("1000-2000 m", -2000, -1000), ("> 2000 m", -1e4, -2000))
    m = (zr1 .> lo) .& (zr1 .<= hi)
    a = vcat([x_cur[:, :, :, n][m] for n in rs]...); b = vcat([x_is[:, :, :, n][m] for n in rs]...)
    @printf("  %-11s %.4f | %.4f | %.4f (%.1f%%)\n", lab, sqrt(mean(abs2, a)), sqrt(mean(abs2, b)), sqrt(mean(abs2, a .- b)), 100sqrt(mean(abs2, a .- b)) / sqrt(mean(abs2, a)))
end

# ---- p_wave, band, p_bc, p_n (window 1) for both; flux with the saved u_n, v_n
φ = NCDataset(d -> d["phi"].var[ix, jy, :, :], window_file(R * "modes10", "modes10", W, win.t0[W]))
nm = size(φ, 4)
fu = window_file(R * "u_n_$(band.name)", "u_n_$(band.name)", W, win.t0[W]); fv = replace(fu, "u_n_" => "v_n_")
Un = NCDataset(d -> d["u_n"].var[ix, jy, :, :], fu); Vn = NCDataset(d -> d["v_n"].var[ix, jy, :, :], fv)
Psaved = NCDataset(d -> d["p_n"].var[ix, jy, :, :], replace(fu, "u_n_" => "p_n_"))
function pn_series(x)
    p = band_filter(p_from_rho(x, ζt, h, dz0), band; order = 4)[:, :, :, rs]
    P = zeros(Float32, nx, ny, nm, length(rs)); F = similar(P)
    for (i, n) in enumerate(rs), jj in 1:ny, ii in 1:nx
        H = ζt[ii, jj, n] + h[ii, jj]; hz = H .* dz0[ii, jj, :]
        pc = p[ii, jj, :, i] .- sum(p[ii, jj, :, i] .* hz) / H                 # p_bc = p_wave - depth mean
        for m in 1:nm; P[ii, jj, m, i] = sum(pc .* hz .* φ[ii, jj, :, m]) / H; end
    end
    P
end
Pc = pn_series(x_cur); x_cur = nothing; GC.gc()
Pi = pn_series(x_is); x_is = nothing; GC.gc()
@printf("\nreproduces the saved p_n: max |p_n(current) - saved| %.2e Pa (rms p_1 %.1f Pa)\n", maximum(abs.(Pc .- Psaved)), sqrt(mean(abs2, Pc[:, :, 1, :])))
H = [ζt[i, j, n] + h[i, j] for i in 1:nx, j in 1:ny, n in rs]
println("mode | rms p_n current, in-situ, rel.diff | tile-sum |F_n| (H<p_n u_n>, H<p_n v_n>) current, in-situ [MW/m*pts], ratio, map corr")
for m in 1:nm
    a, b = Pc[:, :, m, :], Pi[:, :, m, :]
    Fc = (dropdims(mean(H .* a .* Un[:, :, m, :], dims = 3), dims = 3), dropdims(mean(H .* a .* Vn[:, :, m, :], dims = 3), dims = 3))
    Fi = (dropdims(mean(H .* b .* Un[:, :, m, :], dims = 3), dims = 3), dropdims(mean(H .* b .* Vn[:, :, m, :], dims = 3), dims = 3))
    mc, mi = sum(hypot.(Fc...)), sum(hypot.(Fi...))
    @printf("  %2d | %8.2f %8.2f %6.2f%% | %10.3f %10.3f  %.4f  %.5f\n", m, sqrt(mean(abs2, a)), sqrt(mean(abs2, b)),
            100sqrt(mean(abs2, a .- b)) / sqrt(mean(abs2, a)), mc / 1e6, mi / 1e6, mi / mc, cor(vcat(vec.(Fc)...), vcat(vec.(Fi)...)))
end
Ftot(P) = (s = (zeros(nx, ny), zeros(nx, ny)); for m in 1:nm
               s[1] .+= dropdims(mean(H .* P[:, :, m, :] .* Un[:, :, m, :], dims = 3), dims = 3)
               s[2] .+= dropdims(mean(H .* P[:, :, m, :] .* Vn[:, :, m, :], dims = 3), dims = 3); end; s)
Tc, Ti = Ftot(Pc), Ftot(Pi)
@printf("modes 1-%d total: tile-sum |F| in-situ / current = %.4f, map corr %.5f, rms|ΔF| / rms|F| = %.2f%%\n", nm,
        sum(hypot.(Ti...)) / sum(hypot.(Tc...)), cor(vcat(vec.(Tc)...), vcat(vec.(Ti)...)),
        100sqrt(mean(abs2, hypot.(Ti[1] .- Tc[1], Ti[2] .- Tc[2]))) / sqrt(mean(abs2, hypot.(Tc...))))
