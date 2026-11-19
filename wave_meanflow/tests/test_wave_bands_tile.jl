# Test of steps 5-6 (wave_bands.jl) on one tile of rho, both backgrounds, nothing written. Run as a batch job (~20 GB).
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
var = "rho"
t = @elapsed G = wave_grid(win, D, R, var, grid_file); @printf("grid + zeta series: %.1f s\n", t)
tiles, ch = var_tiles(joinpath(D, win.td.files[1]), var)
q = findfirst(((ix, jy),) -> 200 in ix && 250 in jy, tiles); ix, jy = tiles[q]
@printf("tile %d of %d: xi %d:%d, eta %d:%d (chunks %s)\n", q, length(tiles), first(ix), last(ix), first(jy), last(jy), ch)
xbar, zbar = tile_means(win, R, var, G, ix, jy)
i0, j0 = 200 - first(ix) + 1, 250 - first(jy) + 1
X = Dict{String, Array{Float32, 4}}()
for bgm in ("window", "interp")
    t = @elapsed X[bgm] = wave_series_tile(var, ix, jy, win, D, G, xbar, zbar, bg_weights(win, bgm))
    @printf("step 5 (%s): %.1f s, %.2f GB, max |x| %.3f, non-finite %d\n", bgm, t, sizeof(X[bgm]) / 1e9, maximum(abs, X[bgm]), count(!isfinite, X[bgm]))
end
x = X["window"]
own = wave_owner(win)
m = maximum(abs(mean(x[i0, j0, k, own .== w])) for k in (1, 64, 128), w in 1:win.nwin)
@printf("check (200,250): rho_wave window means, max |.| over windows and 3 levels = %.4f (std of rho_wave at k=128 %.4f)\n", m, std(x[i0, j0, 128, :]))
vx = Statistics.var(x)
edge = [last(win.idx[w]) for w in 1:win.nwin-1]                                        # last record before each window edge
near = falses(win.td.nt); for e in edge, d in -12:12; 1 <= e + d <= win.td.nt && (near[e + d] = true); end
for b in BANDS
    t = @elapsed yw = band_filter(x, b; order = 4)
    yi = band_filter(X["interp"], b; order = 4)
    d = yw .- yi
    rms(a) = sqrt(mean(abs2, a))
    @printf("step 6 %-7s %.1f s | var share of rho_wave %.1f%% | window - interp: rms %.4f (within 12 h of window edges %.4f, elsewhere %.4f), band rms %.4f\n",
            b.name, t, 100Statistics.var(yw) / vx, rms(d), rms(d[:, :, :, near]), rms(d[:, :, :, .!near]), rms(yw))
end
