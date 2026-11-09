#= bcef_harmonic_plot.jl
Hsin-Yi Chen, USM DMS
Top-view maps of the harmonic fit of the modal pressure and velocity (mode_bcef_harmonic.jl,
<run>/ddf_bcef/bcef_modes<N>_harmonic.nc): one figure per mode and constituent,
    row 1: amplitude A' of pn, un, vn        row 2: phase g' (relative to the record centre tc) of pn, un, vn
un, vn are along the grid xi / eta axes (the mooring line is a grid column, so vn is along it, un across it).
The phase uses a cyclic colour map (0° = 360°); g' increases in the direction a wave travels (a crest passes later),
so the phase gradient shows the propagation direction and the spacing of a 360° cycle the wavelength.
Masked as the DFF plots: the shelf (h < h_deep) and the edge bands. Figures: figure/<run>/DFF_harmonic/
    harm_<con>_mode<n>.png
Usage:
  julia DFF/bcef_harmonic_plot.jl                                  # 3-month run, every constituent and mode in the file
  DFF_RUN=amazon_900m_pure_or HARM_PLOT_CONS=M2 julia DFF/bcef_harmonic_plot.jl
=#

using NCDatasets, Statistics, Printf, CairoMakie
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")                 # dff_run, run_dir, fig_dir (DFF_RUN)
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")       # topdown_axis, plot_curvilinear!
include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings

## --- settings
NMODE = parse(Int, get(ENV, "NMODE", "3"))
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, "ddf_bcef"))
harm_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_harmonic.nc")
isfile(harm_fname) || error("no $harm_fname: run mode_bcef_harmonic.jl first")
PLOT_CONS_ENV = get(ENV, "HARM_PLOT_CONS", "all")   # "all" = every constituent in the file, or e.g. M2,M4
figure_path = joinpath(dirname(fig_dir), "DFF_harmonic")
mkpath(figure_path)
run_mooring = Dict("amazon_2700m_cmems_0801" => "roms_cmems_grd_2700m_mor_edata.nc")
mooring_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/$(get(run_mooring, dff_run, "roms_grd_900m_mor_edata.nc"))"
edge = 60          # points masked along each edge, as the DFF plots
h_deep = 200.0     # [m] shelf masked
isobaths = [1000, 2000, 3000]   # [m]
stride = 2         # every 2nd grid point in the colour maps, as the DFF plots

## --- read the fit
ds = NCDataset(harm_fname)
cons = String.(ds["con"][:])
PLOT_CONS = PLOT_CONS_ENV == "all" ? cons : String.(split(PLOT_CONS_ENV, ","))
modes = parse.(Int, split(ds.attrib["modes"], ","))
lon, lat, h = ds["lon_rho"][:, :], ds["lat_rho"][:, :], Float64.(ds["h"][:, :])
nx, ny = size(h)
tcstr = ds.attrib["tc"]
recs = ds.attrib["records"]
deep = (h .>= h_deep) .& (edge .< (1:nx) .<= nx - edge) .& (edge .< (1:ny)' .<= ny - edge)
masked(a) = ifelse.(deep, a, NaN)
sub(a) = a[1:stride:end, 1:stride:end]
lon_s, lat_s, h_s = sub(lon), sub(lat), sub(h)
lims = (extrema(lon[deep]) .+ (-0.05, 0.05), extrema(lat[deep]) .+ (-0.05, 0.05))
println("$dff_run: $harm_fname; modes $(join(modes, ",")), constituents in the file $(join(cons, ",")); tc = $tcstr")

## --- moorings (the file's lon is 0..360, the mooring file's -180..180)
moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
lon360 = maximum(lon) > 180
moor_lon = lon360 ? mod.(moor_lon, 360) : moor_lon
ix_line = round(Int, 0.5 + nx / 2)   # the mooring line (middle grid column), as the DFF and mode_analysis plots

function decorate!(ax)
    contour!(ax, lon_s, lat_s, h_s; levels = isobaths, color = :gray40, linewidth = 0.7)
    lines!(ax, lon[ix_line, :], lat[ix_line, :]; color = :black, linestyle = :dash, linewidth = 0.8)
    scatter!(ax, moor_lon, moor_lat; marker = :circle, markersize = 6, color = :yellow, strokewidth = 0.6)
    limits!(ax, lims...)
end

## --- one figure per mode and constituent
# fields: (variable prefix, title, scale to the plotted unit, unit)
fields(n) = [("p$n", "p$(n) (pressure)", 1.0, "Pa"),
             ("u$n", "u$(n) (grid ξ, across the line)", 100.0, "cm s⁻¹"),
             ("v$n", "v$(n) (grid η, along the line)", 100.0, "cm s⁻¹")]
for c in PLOT_CONS, n in modes
    k = findfirst(==(c), cons)
    k === nothing && (println("  $c not in the file, skipped"); continue)
    fig = Figure(size = (1600, 1150))
    for (col, (var, title, sc, unit)) in enumerate(fields(n))
        A = masked(Float64.(ds["$(var)_amp"][:, :, k]) .* sc)
        g = masked(Float64.(ds["$(var)_pha"][:, :, k]))
        amax = quantile(filter(isfinite, A), 0.99)
        amax = round(amax; sigdigits = 2)
        # amplitude
        ax1 = topdown_axis(fig[1, col]; title = "A′ of $title", ylabel = col == 1 ? "Latitude" : "", xlabel = "",
                           backgroundcolor = :gray80)
        hm1 = plot_curvilinear!(ax1, lon_s, lat_s, sub(A); colormap = :thermal, colorrange = (0, amax))
        decorate!(ax1)
        Colorbar(fig[2, col], hm1; vertical = false, flipaxis = false, label = "A′ [$unit]")
        # phase
        ax2 = topdown_axis(fig[3, col]; title = "g′ of $title", ylabel = col == 1 ? "Latitude" : "", backgroundcolor = :gray80)
        hm2 = plot_curvilinear!(ax2, lon_s, lat_s, sub(g); colormap = :romaO, colorrange = (0, 360))
        decorate!(ax2)
        Colorbar(fig[4, col], hm2; vertical = false, flipaxis = false, label = "g′ [°]", ticks = 0:90:360)
        @printf("  %s mode %d %s: median A' %.3g %s (99th pct %.3g)\n", c, n, var, median(filter(isfinite, A)), unit, amax)
    end
    Label(fig[0, :], "$c harmonic fit of mode $n, $dff_run ($recs hourly records, phase relative to $tcstr)\n" *
          "masked: h < $(Int(h_deep)) m and $edge edge points; grey: $(join(isobaths, " / ")) m isobaths; dashed: mooring line; yellow: moorings";
          fontsize = 15)
    fname = joinpath(figure_path, "harm_$(c)_mode$(n).png")
    save(fname, fig; px_per_unit = 2)
    println("saved $fname")
end
close(ds)
