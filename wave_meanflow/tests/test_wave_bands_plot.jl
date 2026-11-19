# Tile test 2 of steps 5-6 (wave_bands.jl): p_wave check + time-depth plots of rho_wave hp9 / bp9_15, both backgrounds.
# One tile of rho (tile 7, holds column (200, 250)), nothing written to meanflowdata. Batch job (~30 GB).
using NCDatasets, Printf, Statistics, Dates, DSP, CairoMakie
include("/home/hchen54/internalwave_julia/function/butter_filters.jl")
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
for f in ("meanflow_window", "wave_perturb", "mode_projection", "wave_bands")
    include("/home/hchen54/internalwave_julia/wave_meanflow/$f.jl")
end
D = "/expanse/lustre/projects/uso102/hchen54/amazon_2700m_cmems_0801/after_joint/"
R = "/expanse/lustre/projects/uso102/hchen54/amazon_2700m_cmems_0801/meanflowdata/L72h_ov0.00/"
grid_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/roms_cmems_grd_2700m.nc"
outdir = expanduser("~/figure_check"); mkpath(outdir)
win = window_setup(D, "rho"; tstart = DateTime(2022, 9, 15), tend = DateTime(2022, 10, 15), window_length = 72, verbose = false)
G = wave_grid(win, D, R, "rho", grid_file)
tiles, _ = var_tiles(joinpath(D, win.td.files[1]), "rho")
q = findfirst(((ix, jy),) -> 200 in ix && 250 in jy, tiles); ix, jy = tiles[q]
xbar, zbar = tile_means(win, R, "rho", G, ix, jy)
X = Dict(b => wave_series_tile("rho", ix, jy, win, D, G, xbar, zbar, bg_weights(win, b)) for b in ("window", "interp"))

# ---- p_wave check (interp background)
x = X["interp"]; ζt = view(G.ζ, ix, jy, :); ht = G.h[ix, jy]; dz0 = G.dz0[ix, jy, :]; z0 = G.z0[ix, jy, :]
t = @elapsed p = p_from_rho(x, ζt, ht, dz0)
@printf("p_wave: %.1f s, max |p| %.1f Pa, non-finite %d, surface level max |p| %.2f Pa\n", t, maximum(abs, p), count(!isfinite, p), maximum(abs, p[:, :, end, :]))
# hydrostatic balance between levels: (p[k+1] - p[k]) / (z[k+1] - z[k]) = -g (rho'[k] + rho'[k+1]) / 2
worst = 0.0; scale = 0.0
for n in 1:24:win.td.nt, j in 1:8:length(jy), i in 1:8:length(ix)
    H = ζt[i, j, n] + ht[i, j]; z = ζt[i, j, n] .+ H .* z0[i, j, :]
    lhs = diff(p[i, j, :, n]) ./ diff(z); rhs = -9.81 .* (x[i, j, 1:end-1, n] .+ x[i, j, 2:end, n]) ./ 2
    global worst = max(worst, maximum(abs.(lhs .- rhs))); global scale = max(scale, maximum(abs.(rhs)))
end
@printf("p_wave hydrostatic check: max |dp/dz + g rho'| %.2e of max |g rho'| %.2e (relative %.1e)\n", worst, scale, worst / scale)
vp = Statistics.var(p)
for b in BANDS
    y = band_filter(p, b; order = 4)
    @printf("p_wave %-7s var share %.1f%%, rms %.1f Pa\n", b.name, 100Statistics.var(y) / vp, sqrt(mean(abs2, y)))
end
p = nothing; GC.gc()

# ---- time-depth plots of rho_wave hp9 and bp9_15 at two columns, window vs interp background
h = G.h[ix, jy]
cols = [("deep", (200, 250)), ("shallow", Tuple(argmin(abs.(h .- 150)).I) .+ (first(ix) - 1, first(jy) - 1))]
days = Dates.value.(win.td.tdate .- win.td.tdate[1]) ./ 8.64e7
edges = [Dates.value(win.t0[w] - win.td.tdate[1]) / 8.64e7 for w in 2:win.nwin]
bsel = [b for b in BANDS if b.name in ("hp9", "bp9_15")]
F = Dict((bg, b.name) => band_filter(X[bg], b; order = 4) for bg in ("window", "interp"), b in bsel)
for (lab, (ig, jg)) in cols
    i, j = ig - first(ix) + 1, jg - first(jy) + 1
    zm = G.ζbar[1][ig, jg] .+ (G.ζbar[1][ig, jg] + G.h[ig, jg]) .* G.z0[ig, jg, :]     # level depths of window 1 (plot axis)
    fig = Figure(size = (1400, 900), fontsize = 14)
    Label(fig[0, 1:3], @sprintf("rho_wave bands at (xi %d, eta %d), h = %.0f m — window-mean vs time-interpolated background; dashed = window edges", ig, jg, G.h[ig, jg]),
          fontsize = 16)
    for (r, b) in enumerate(bsel)
        lim = maximum(abs, F[("interp", b.name)][i, j, :, :]) * 0.8
        local hm
        for (c, bg) in enumerate(("window", "interp"))
            ax = Axis(fig[r, c]; title = "rho_wave_$(b.name), background = $bg", xlabel = "days since 2022-09-15",
                      ylabel = "depth (m)")
            hm = heatmap!(ax, days, zm, permutedims(F[(bg, b.name)][i, j, :, :]); colormap = Reverse(:RdBu), colorrange = (-lim, lim))
            vlines!(ax, edges; color = (:black, 0.5), linestyle = :dash, linewidth = 1)
        end
        Colorbar(fig[r, 3], hm; label = "rho_wave_$(b.name) (kg m⁻³)")
    end
    fn = joinpath(outdir, "wmf_rho_wave_bands_$(lab)_xi$(ig)_eta$(jg).png")
    save(fn, fig); println("wrote ", fn)
end
