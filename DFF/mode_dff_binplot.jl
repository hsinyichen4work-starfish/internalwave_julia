#= mode_dff_binplot.jl
Hsin-Yi Chen, USM DMS
Line plots of the modal energy flux and related fields averaged over bins of a wider strip around the mooring line,
instead of the single grid column (xi = ix_line) used in mode_analysis.jl and bcef_harmonic_plot.jl: a single column
picks up whatever beam edge or interference pattern happens to sit on it, a bin average over a band of columns shows
the robust along-line structure.

Inputs (all from the 2022-08-24 -> 11-30 window of amazon_2700m_cmems_0801, BCEF_OUT = <run>/ddf_bcef_0824):
    bcef_modes3_structure.nc   h, mean zeta, c_n (mode speeds)
    bcef_dff_flux_modes3.nc    DFF time-mean flux maps <Fx|Fy>_<band>_<sector>_<mode1..3|modesum> (xi, eta, period)
                               on a grid cropped to odd sizes (377 x 509): every field here is cropped to it
    bcef_modes3_harmonic.nc    harmonic fit of pn, un, vn (modes 1, 2) and the constituent flux
    bcef_modes3_puv.nc         pn, un, vn time series (not read in step 1)
Figures: figure/amazon_2700m_cmems_0801/bin_avg_lineplot/

Steps: (1) read; (2) check map of the strip (~/figure_check); (3) harmonic flux F_x,n, F_y,n of every constituent,
strip-averaged (figure_path/binplot_harm_flux_<con>_mode<n>.png); (4) harmonic KE_n (figure_path/binplot_harm_KE_<con>.png);
(5) harmonic APE_n (figure_path/binplot_harm_APE_<con>.png); (6) total energy KE_n + APE_n and KE/APE (figure_path/binplot_harm_energy_<con>.png); (7) conversion conv_n
(W_BT_make.jl, bt_conv_harmonic.nc; figure_path/binplot_harm_conv_<con>.png); (8) flux divergence ∇·F_n with conv_n
(figure_path/binplot_harm_div_<con>.png); (9) local loss D_n = conv_n − ∇·F_n (figure_path/binplot_harm_D_<con>.png); (10) bathymetry along the mooring line and
across the strip (test, ~/figure_check); (11) DFF of the harmonic flux (DFF_on_harmonic.jl) by direction (figure_path/harm_dff/); (12) DFF of the time series
(DFF_on_bcef.jl): time-mean flux by direction across the strip (figure_path/dff/).
Usage:
  julia DFF/mode_dff_binplot.jl                                        # amazon_2700m_cmems_0801, 08-24 window, steps 2-4
  BIN_STEPS=3 HARM_BIN_CONS=M2:M4 julia DFF/mode_dff_binplot.jl        # step 3 for M2 and M4 only
  DFF_RUN=... BCEF_OUT=... julia DFF/mode_dff_binplot.jl               # another run / output folder
On the login node (8 GB limit) run step 3 a few constituents at a time.
=#

using NCDatasets, Statistics, Dates, Printf, CairoMakie
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
haskey(ENV, "DFF_RUN") || (ENV["DFF_RUN"] = "amazon_2700m_cmems_0801")   # default run of this script
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")                 # dff_run, run_dir, run_grid_fname, fig_dir
include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings

# ============================================================================
# (1) settings, paths, read the data
# ============================================================================
NMODE = parse(Int, get(ENV, "NMODE", "3"))
# default output folder of this run: the spin-up-free (2022-08-24 on) decomposition
default_bcef = Dict("amazon_2700m_cmems_0801" => "ddf_bcef_0824")
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, get(default_bcef, dff_run, "ddf_bcef")))
struct_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc")
dff_fname    = joinpath(bcef_dir, "bcef_dff_flux_modes$(NMODE).nc")
harm_fname   = joinpath(bcef_dir, "bcef_modes$(NMODE)_harmonic.nc")
puv_fname    = joinpath(bcef_dir, "bcef_modes$(NMODE)_puv.nc")
for f in (struct_fname, dff_fname, harm_fname, puv_fname)
    isfile(f) || error("missing $f")
end
figure_path = joinpath(dirname(fig_dir), "bin_avg_lineplot")
mkpath(figure_path)
run_mooring = Dict("amazon_2700m_cmems_0801" => "roms_cmems_grd_2700m_mor_edata.nc")
mooring_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/$(get(run_mooring, dff_run, "roms_grd_900m_mor_edata.nc"))"

h_mask  = 200.0                                           # [m] shelf not drawn
# steps to run after (1): BIN_STEPS=3 or 3:4 (default all: 2,3,4,5,6,7,8,9,10,11,12; ":" also works in sbatch --export)
bin_steps = parse.(Int, split(get(ENV, "BIN_STEPS", "2,3,4,5,6,7,8,9,10,11,12"), r"[,:; ]+"; keepempty = false))
edge_km = parse(Float64, get(ENV, "EDGE_KM", "54"))       # [km] edge bands (sponge / boundary), as mode_analysis.jl

# --- the DFF grid (cropped to odd sizes); every field below is cropped to it
dsd = NCDataset(dff_fname)
nx, ny = dsd.dim["xi_rho"], dsd.dim["eta_rho"]
periods = dsd.attrib["periods"]                           # period 1 = full record
dff_vars = filter(k -> startswith(k, "Fx_") || startswith(k, "Fy_"), collect(keys(dsd)))
dff_bands = unique([split(k, "_")[2] for k in dff_vars])
dff_secs  = unique([split(k, "_")[3] for k in dff_vars])

# --- grid, depth, mode speeds
lon, lat, pm, pn, angle = NCDataset(run_grid_fname) do g
    Float64.(g["lon_rho"][1:nx, 1:ny]), Float64.(g["lat_rho"][1:nx, 1:ny]),
    Float64.(g["pm"][1:nx, 1:ny]), Float64.(g["pn"][1:nx, 1:ny]), Float64.(g["angle"][1:nx, 1:ny])
end
h, zeta_mean, c_mode = NCDataset(struct_fname) do s   # c_mode: mode eigenspeeds (xi, eta, mode) [m s-1]
    Float64.(s["h"][1:nx, 1:ny]), Float64.(s["zeta_mean"][1:nx, 1:ny]), Float64.(s["c"][1:nx, 1:ny, :])
end
H = h .+ zeta_mean
dx_km = mean(1 ./ pm) / 1e3
edge = round(Int, edge_km / dx_km)
inside = (edge .< (1:nx) .<= nx - edge) .& (edge .< (1:ny)' .<= ny - edge)

# --- harmonic fit (opened, read field by field later)
dsh = NCDataset(harm_fname)
harm_cons  = String.(dsh["con"][:])
harm_modes = parse.(Int, split(dsh.attrib["modes"], ","))

# --- puv time series: only the record length here
nt, t1, t2 = NCDataset(puv_fname) do d
    d.dim["time"], d["ocean_time"][1], d["ocean_time"][end]
end

# --- moorings and the mooring line (the middle xi column, as in mode_analysis.jl)
moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
moor_lon = maximum(lon) > 180 ? mod.(moor_lon, 360) : moor_lon
moor_ij = NCDataset(mooring_file) do g
    [(round(Int, g["$(m)_mor"][1] + 1), round(Int, g["$(m)_mor"][2] + 1)) for m in moor_name]
end
ix_line = round(Int, 0.5 + (nx + 1) / 2)   # same column as mode_analysis.jl (its nx is the uncropped 378)
line_km = [0; cumsum(0.5 .* (1 ./ pn[ix_line, 1:end-1] .+ 1 ./ pn[ix_line, 2:end]))] ./ 1e3   # [km] from the grid-south end

println("$dff_run: $bcef_dir")
println("  grid (DFF crop) $nx x $ny, spacing $(round(dx_km, digits = 2)) km; h $(round.(extrema(h), digits = 0)) m; $NMODE modes")
println("  DFF flux: bands $(join(dff_bands, ", ")); sectors $(join(dff_secs, ", ")); periods: $periods")
println("  harmonic fit: modes $(join(harm_modes, ",")), constituents $(join(harm_cons, ","))")
println("  puv: $nt records, $t1 -> $t2")
println("  mooring line xi = $ix_line, $(round(line_km[end], digits = 0)) km long; edge bands $edge_km km = $edge points")
println("  $(length(moor_name)) moorings: ", join(moor_name, ", "))
println("figures -> $figure_path")

# ============================================================================
# (2) check map: the DFF |F| of the full record with the middle mooring line and the west / east lines
# ============================================================================
# The strip for the bin averages is bounded by two grid columns at side_km on each side of the mooring line
# (200 km = 74 points on the 2.7 km grid, 222 on the 900 m grid). Map: |F| of the time-mean DFF flux (period 1 = full record), the band /
# sector / mode set by BIN_BAND, BIN_SEC, BIN_MODE; arrows rotated from grid xi / eta to east / north.
check_path = expanduser("~/figure_check")
mkpath(check_path)
side_km   = parse(Float64, get(ENV, "SIDE_KM", "200"))   # [km] west / east lines from the mooring line (fixed distance; points from the grid spacing)
ix_side   = round(Int, side_km / dx_km)
ix_west, ix_east = ix_line - ix_side, ix_line + ix_side  # grid xi increases eastward (offshore-parallel)
# extra side lines for comparison on the check map (e.g. EXTRA_SIDE_KM=150:250, drawn green, red; default none)
extra_km = parse.(Float64, split(get(ENV, "EXTRA_SIDE_KM", ""), r"[,:; ]+"; keepempty = false))
extra_col = [:orange, :red, :magenta, :cyan]
band = get(ENV, "BIN_BAND", "M2"); sec = get(ENV, "BIN_SEC", "all"); mtag = get(ENV, "BIN_MODE", "modesum")
lon180 = ifelse.(lon .> 180, lon .- 360, lon)            # -180..180 for the map axis
moor_lon180 = ifelse.(moor_lon .> 180, moor_lon .- 360, moor_lon)
show = (h .>= h_mask) .& inside
masked(a) = ifelse.(show, a, NaN)
Fx = Float64.(nomissing(dsd["Fx_$(band)_$(sec)_$(mtag)"][:, :, 1], NaN)) ./ 1e3   # [kW m-1]
Fy = Float64.(nomissing(dsd["Fy_$(band)_$(sec)_$(mtag)"][:, :, 1], NaN)) ./ 1e3
Fe = Fx .* cos.(angle) .- Fy .* sin.(angle)               # grid -> east / north
Fn = Fx .* sin.(angle) .+ Fy .* cos.(angle)
Fmag = hypot.(Fx, Fy)
println("\ncheck map: $band $sec $mtag; lines xi = $ix_west (west), $ix_line (mooring), $ix_east (east): ±$ix_side points = ±$(round(ix_side * dx_km, digits = 1)) km")
if 2 in bin_steps   # check map
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # topdown_axis, plot_curvilinear!, quiver_curvilinear!
    fmax = round(quantile(filter(isfinite, masked(Fmag)), 0.995); sigdigits = 2)
    fig = Figure(size = (520, 640))
    ax = topdown_axis(fig[1, 1]; title = "|F $sec|  ($band, $mtag, full record)", backgroundcolor = :gray80)
    hm = plot_curvilinear!(ax, lon180, lat, masked(Fmag); colormap = :plasma, colorrange = (0, fmax))
    contour!(ax, lon180, lat, h; levels = [h_mask], color = :gray30, linewidth = 0.8)
    skip = max(1, round(Int, 36 / dx_km))                     # one arrow every ~36 km, as the DFF plots
    quiver_curvilinear!(ax, lon180, lat, masked(Fe), masked(Fn); skip = skip, lengthscale = 0.6 / fmax,
                        color = (:white, 0.8), shaftwidth = 0.8, tipwidth = 4, tiplength = 4)
    side_lines = [(ix_west, :lime, "W"), (ix_line, :white, "M"), (ix_east, :lime, "E")]
    for (k, km) in enumerate(extra_km)
        n = round(Int, km / dx_km)
        (ix_line - n >= 1 && ix_line + n <= nx) || (println("  $km km lines outside the grid, skipped"); continue)
        col = extra_col[mod1(k, length(extra_col))]
        append!(side_lines, [(ix_line - n, col, ""), (ix_line + n, col, "")])
        println("  extra lines ±$km km: xi = $(ix_line - n), $(ix_line + n) (±$n points)")
    end
    for (ix, col, lab) in side_lines
        lines!(ax, lon180[ix, :], lat[ix, :]; color = col, linestyle = :dash, linewidth = 1.5)
        isempty(lab) || text!(ax, lon180[ix, end - edge], lat[ix, end - edge]; text = lab, color = col, font = :bold,
                              align = (:center, :bottom), offset = (0, 4), fontsize = 13)
    end
    isempty(extra_km) || Label(fig[3, 1], "green: ±$(round(Int, side_km)) km" *
        join([";  $(extra_col[mod1(k, length(extra_col))]): ±$(round(Int, km)) km" for (k, km) in enumerate(extra_km)]);
        fontsize = 11, tellwidth = false)
    scatter!(ax, moor_lon180, moor_lat; marker = :circle, markersize = 6, color = :yellow, strokewidth = 0.6)
    Colorbar(fig[2, 1], hm; vertical = false, flipaxis = false, label = "|F| [kW/m]")
    fname = joinpath(check_path, "binplot_check_map_$(band)_$(sec)_$(mtag).png")
    save(fname, fig; px_per_unit = 2)
    println("saved $fname")
end

# ============================================================================
# shared by the strip-average figures (steps 3, 4)
# ============================================================================
# Line plots: for each grid row (fixed eta, south -> north) the mean over the strip xi = ix_west .. ix_east (shelf
# h < h_mask and the edge bands left out), against the distance along the mooring line; a row needs at least
# min_cover of the strip in deep water, or it is not drawn (near the coast the strip runs onto the shelf).
# Shading: ± one standard deviation across the strip (how much the row varies W - E). The plots end crop_km past
# crop_after (cyan dashed across the strip on the maps).
min_cover = 0.5
strip = ix_west:ix_east
# row statistics over the strip: mean, std and the fraction of deep points, per eta row
function strip_stats(a)
    m, s, f = fill(NaN, ny), fill(NaN, ny), zeros(ny)
    for j in 1:ny
        v = [a[i, j] for i in strip if show[i, j] && isfinite(a[i, j])]
        f[j] = length(v) / length(strip)
        f[j] >= min_cover || continue
        m[j], s[j] = mean(v), std(v)
    end
    m, s, f
end
rd_h(var, k) = Float64.(nomissing(dsh[var][1:nx, 1:ny, k], NaN))
Hh = Float64.(dsh["H"][1:nx, 1:ny])                       # h + mean zeta, as used in the fit's flux
j_node = argmin(abs.(line_km .- 200))
# north end of the line plots: CROP_KM past the last mooring of the line (CPIES7, ~632 km), the rest of the large
# 2700 m domain has little flux
crop_after = get(ENV, "CROP_AFTER", "CPIES7")
crop_km = parse(Float64, get(ENV, "CROP_KM", "250"))
j_crop = let k = findfirst(==(crop_after), moor_name)
    k === nothing && error("no mooring $crop_after")
    min(argmin(abs.(line_km .- (line_km[moor_ij[k][2]] + crop_km))), ny - edge)
end
println("line plots end $(round(Int, crop_km)) km past $crop_after: eta = $j_crop, $(round(Int, line_km[j_crop])) km along the line")
moor_in = [(nm, j, abs(i - ix_line) <= 1) for (nm, (i, j)) in zip(moor_name, moor_ij) if abs(i - ix_line) <= ix_side]   # moorings in the strip (true: on the line)
rows_ok(m) = isfinite.(m) .& ((1:ny) .<= j_crop)          # rows drawn: deep enough, up to the crop
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # topdown_axis, plot_curvilinear!
fig_w, paper_w_cm = 1300, 18.0
pt = fig_w / (paper_w_cm / 2.54 * 72)                     # pixels per point: 10 pt text on an 18 cm page
map_w = 400
map_lims = (extrema(lon180[show]) .+ (-0.1, 0.1), extrema(lat[show]) .+ (-0.1, 0.1))
map_h = round(Int, map_w * (map_lims[2][2] - map_lims[2][1]) / ((map_lims[1][2] - map_lims[1][1]) * cosd(mean(map_lims[2]))))
strip_note = "strip: ξ = $(ix_west)–$(ix_east) (green on the maps), one mean per grid row (η); shelf (h < $(Int(h_mask)) m) and $(round(Int, edge_km)) km edge bands left out;\n" *
             "rows with < $(round(Int, 100min_cover))% deep cover not drawn; cyan on the maps: north end of the line plots, $(round(Int, crop_km)) km past $crop_after;\n" *
             "red star / red dashed: 200 km node; grey lines: moorings (labelled: on the line); maps: one colour range per panel"

# map panel at fig[row, 1] (colour bar at fig[row, 2]) with the strip, crop, moorings and node
function strip_map!(fig, row, data, title; colormap, colorrange, last_row = false, cbticks = Makie.automatic)
    ax = topdown_axis(fig[row, 1]; title = title, xlabel = last_row ? "Longitude" : "", backgroundcolor = :gray80,
                      width = map_w, height = map_h, aspect = nothing)
    hm = plot_curvilinear!(ax, lon180, lat, masked(data); colormap = colormap, colorrange = colorrange)
    contour!(ax, lon180, lat, h; levels = [1000, 2000, 3000], color = :gray40, linewidth = 0.6)
    for (ix, col) in ((ix_west, :green), (ix_line, :black), (ix_east, :green))
        lines!(ax, lon180[ix, :], lat[ix, :]; color = col, linestyle = :dash, linewidth = 1.2)
    end
    lines!(ax, lon180[strip, j_crop], lat[strip, j_crop]; color = :cyan, linestyle = :dash, linewidth = 2)
    scatter!(ax, moor_lon180, moor_lat; marker = :circle, markersize = 5, color = :yellow, strokewidth = 0.5)
    scatter!(ax, [lon180[ix_line, j_node]], [lat[ix_line, j_node]]; marker = :star5, markersize = 14, color = :red,
             strokecolor = :black, strokewidth = 0.6)
    limits!(ax, map_lims...)
    Colorbar(fig[row, 2], hm; height = map_h, ticks = cbticks)
    ax
end

# line axis at fig[row, 3]; yvals: every value to be drawn (for the y range, with headroom for the mooring labels)
function strip_line!(fig, row, title, ylabel, yvals; last_row = false, ymin0 = false, ymax = nothing)
    al = Axis(fig[row, 3]; title = title, ylabel = ylabel,
              xlabel = last_row ? "distance along the mooring line from the south boundary [km]" : "",
              xticks = 0:100:ceil(Int, line_km[end]), xminorticks = IntervalsBetween(2), xminorticksvisible = true,
              xminorgridvisible = true, width = 2map_w + 60, height = map_h)
    v = filter(isfinite, yvals)
    ylo, yhi = extrema(vcat(v, 0.0))
    ymin0 && (ylo = 0.0)
    ymax === nothing || (yhi = ymax)   # fixed top (e.g. a ratio with outliers where the denominator is ~0)
    ytop = yhi + 0.4 * (yhi - ylo)
    ylims!(al, ymin0 ? 0.0 : ylo - 0.05 * (yhi - ylo), ytop)
    hlines!(al, [0]; color = :black, linewidth = 0.6)
    for (nm, j, on) in moor_in   # all moorings in the strip as thin lines; only those on the line labelled
        vlines!(al, [line_km[j]]; color = on ? :gray60 : :gray85, linewidth = on ? 0.8 : 0.5)
        on && text!(al, line_km[j], ytop; text = nm * " ", rotation = π / 2, align = (:right, :center), fontsize = 7pt, color = :gray35)
    end
    vlines!(al, [line_km[j_node]]; color = :red, linestyle = :dash, linewidth = 1.2)
    al
end
finish_line!(al, ok) = xlims!(al, line_km[findfirst(ok)], line_km[j_crop])

# ============================================================================
# (3) harmonic-fit energy flux, averaged across the ±side_km strip
# ============================================================================
# F_x,n = ½ H A_p A_u cos(g'_p - g'_u) and F_y,n = ½ H A_p A_v cos(g'_p - g'_v) (mode_bcef_harmonic.jl fit), per
# constituent (HARM_BIN_CONS, default every constituent in the file) and mode. Same curves as bcef_harmonic_plot.jl:
# F (the cos part), ½HA_pA_w (the most it could carry), the sin part. Figures: figure_path/binplot_harm_flux_<con>_mode<n>.png
harm_bin_env = get(ENV, "HARM_BIN_CONS", "all")
harm_bin_cons = harm_bin_env == "all" ? harm_cons : String.(split(harm_bin_env, r"[,:; ]+"; keepempty = false))
if 3 in bin_steps
    for c in harm_bin_cons, n in harm_modes
        k = findfirst(==(c), harm_cons)
        k === nothing && (println("  $c not in $harm_fname"); continue)
        Ap, gp = rd_h("p$(n)_amp", k), rd_h("p$(n)_pha", k)
        fig = Figure(size = (fig_w, 1500), fontsize = 10pt)
        axl = Axis[]
        for (row, (w, comp, dir)) in enumerate([("u", "x", "across the line"), ("v", "y", "along the line")])
            Aw, gw = rd_h("$(w)$(n)_amp", k), rd_h("$(w)$(n)_pha", k)
            P = 0.5 .* Hh .* Ap .* Aw ./ 1e3              # ½ H A_p A_w [kW m-1]
            dg = gp .- gw
            F, S = P .* cosd.(dg), P .* sind.(dg)
            lim = round(quantile(filter(isfinite, masked(P)), 0.99); sigdigits = 2)
            strip_map!(fig, row, F, "F_$comp = ½HA_pA_$w cos(g′_p − g′_$w)\n($dir)"; colormap = :balance,
                       colorrange = (-lim, lim), last_row = row == 2)
            Fm, Fs, _ = strip_stats(F)
            Pm, _, _ = strip_stats(P)
            Sm, _, _ = strip_stats(S)
            ok = rows_ok(Fm)
            al = strip_line!(fig, row, "F_$comp averaged across the strip (±$(round(Int, side_km)) km, ξ = $(ix_west)–$(ix_east))",
                             "F_$comp [kW m⁻¹]", vcat((Fm .- Fs)[ok], (Fm .+ Fs)[ok], Pm[ok], Sm[ok]); last_row = row == 2)
            push!(axl, al)
            band!(al, line_km, Fm .- Fs, Fm .+ Fs; color = (:royalblue, 0.18), label = "F ± 1 std across the strip")
            lines!(al, line_km, Pm; color = :gray40, linestyle = :dash, linewidth = 1.5, label = "½HA_pA_w (most it could carry)")
            lines!(al, line_km, Sm; color = :darkorange, linestyle = :dot, linewidth = 2, label = "½HA_pA_w sin(Δg′) (out of phase)")
            lines!(al, line_km, Fm; color = :royalblue, linewidth = 2.5, label = "F = ½HA_pA_w cos(Δg′), strip mean")
            finish_line!(al, ok)
            @printf("  %s mode %d F%s strip mean: %.0f W/m (½HA_pA %.0f); rows drawn %d of %d\n",
                    c, n, comp, 1e3mean(Fm[ok]), 1e3mean(Pm[ok]), count(ok), ny)
        end
        Label(fig[0, :], "$c mode-$n energy flux (harmonic fit), mean across ±$(round(Int, side_km)) km of the mooring line, $dff_run"; fontsize = 11pt, font = :bold)
        Legend(fig[3, :], axl[2]; orientation = :horizontal, framevisible = false, labelsize = 9pt, nbanks = 2)
        Label(fig[4, :], strip_note; fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(figure_path, "binplot_harm_flux_$(c)_mode$(n).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()   # free the figure (8 GB limit on the login node)
        println("saved $fname")
    end
end

# ============================================================================
# (4) harmonic-fit kinetic energy, averaged across the ±side_km strip
# ============================================================================
# KE_n = ¼ ρ0 H (A_u² + A_v²) [J m-2]: the time mean of ½ ρ0 H (un² + vn²) for one sinusoid of amplitudes A_u, A_v
# (mode n, constituent HARM_KE_CONS, default every constituent in the file). One row per mode: map of KE_n, and its
# strip mean with the parts ¼ρ0HA_u² (across the line) and ¼ρ0HA_v² (along it).
# Figures: figure_path/binplot_harm_KE_<con>.png
harm_ke_env = get(ENV, "HARM_KE_CONS", "all")
harm_ke_cons = harm_ke_env == "all" ? harm_cons : String.(split(harm_ke_env, r"[,:; ]+"; keepempty = false))
his0 = first(filter(f -> startswith(f, "roms_his") && endswith(f, ".nc"), readdir(joinpath(run_dir, "after_joint"))))
rho0 = NCDataset(d -> Float64(d.attrib["rho0"]), joinpath(run_dir, "after_joint", his0))   # [kg m-3]
if 4 in bin_steps
    println("\nKE: rho0 = $rho0 kg m-3 (his file attribute)")
    for c in harm_ke_cons
        k = findfirst(==(c), harm_cons)
        k === nothing && (println("  $c not in $harm_fname"); continue)
        nm = length(harm_modes)
        fig = Figure(size = (fig_w, 800nm), fontsize = 10pt)
        axl = Axis[]
        for (row, n) in enumerate(harm_modes)
            Au, Av = rd_h("u$(n)_amp", k), rd_h("v$(n)_amp", k)
            KEu = 0.25 .* rho0 .* Hh .* Au .^ 2 ./ 1e3    # [kJ m-2]
            KEv = 0.25 .* rho0 .* Hh .* Av .^ 2 ./ 1e3
            KE = KEu .+ KEv
            lim = round(quantile(filter(isfinite, masked(KE)), 0.99); sigdigits = 2)
            strip_map!(fig, row, KE, "KE_$n = ¼ρ₀H(A_u² + A_v²), mode $n"; colormap = :thermal, colorrange = (0, lim),
                       last_row = row == nm)
            Km, Ks, _ = strip_stats(KE)
            Kum, _, _ = strip_stats(KEu)
            Kvm, _, _ = strip_stats(KEv)
            ok = rows_ok(Km)
            al = strip_line!(fig, row, "KE_$n averaged across the strip (±$(round(Int, side_km)) km, ξ = $(ix_west)–$(ix_east))",
                             "KE_$n [kJ m⁻²]", vcat((Km .+ Ks)[ok], Kum[ok], Kvm[ok]); last_row = row == nm, ymin0 = true)
            push!(axl, al)
            band!(al, line_km, max.(Km .- Ks, 0), Km .+ Ks; color = (:royalblue, 0.18), label = "KE ± 1 std across the strip")
            lines!(al, line_km, Kum; color = :darkorange, linestyle = :dot, linewidth = 2, label = "¼ρ₀HA_u² (across the line)")
            lines!(al, line_km, Kvm; color = :seagreen, linestyle = :dash, linewidth = 2, label = "¼ρ₀HA_v² (along the line)")
            lines!(al, line_km, Km; color = :royalblue, linewidth = 2.5, label = "KE = ¼ρ₀H(A_u² + A_v²), strip mean")
            finish_line!(al, ok)
            @printf("  %s mode %d KE strip mean: %.2f kJ/m² (u part %.2f, v part %.2f); at the node %.2f\n",
                    c, n, mean(Km[ok]), mean(Kum[ok]), mean(Kvm[ok]), Km[j_node])
        end
        Label(fig[0, :], "$c modal KE (harmonic fit), mean across ±$(round(Int, side_km)) km of the mooring line, $dff_run"; fontsize = 11pt, font = :bold)
        Legend(fig[nm + 1, :], axl[end]; orientation = :horizontal, framevisible = false, labelsize = 9pt, nbanks = 2)
        Label(fig[nm + 2, :], strip_note; fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(figure_path, "binplot_harm_KE_$(c).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("saved $fname")
    end
end

# ============================================================================
# (5) harmonic-fit available potential energy, averaged across the ±side_km strip
# ============================================================================
# APE_n = H A_p² / (4 ρ0 c_n²) [J m-2]: the time mean of H pn² / (2 ρ0 c_n²) for one sinusoid of amplitude A_p
# (c_n: mode-n eigenspeed at each point, bcef_modes<N>_structure.nc). One row per mode: map of APE_n, and its strip
# mean ± 1 std, with the strip-mean KE_n of step 4 for comparison (free progressive wave: KE/APE = (ω²+f²)/(ω²-f²)).
# Constituents HARM_APE_CONS (default every constituent in the file). Figures: figure_path/binplot_harm_APE_<con>.png
harm_ape_env = get(ENV, "HARM_APE_CONS", "all")
harm_ape_cons = harm_ape_env == "all" ? harm_cons : String.(split(harm_ape_env, r"[,:; ]+"; keepempty = false))
if 5 in bin_steps
    println("\nAPE: rho0 = $rho0 kg m-3, c_n from $struct_fname")
    for c in harm_ape_cons
        k = findfirst(==(c), harm_cons)
        k === nothing && (println("  $c not in $harm_fname"); continue)
        nm = length(harm_modes)
        fig = Figure(size = (fig_w, 800nm), fontsize = 10pt)
        axl = Axis[]
        for (row, n) in enumerate(harm_modes)
            Ap, Au, Av = rd_h("p$(n)_amp", k), rd_h("u$(n)_amp", k), rd_h("v$(n)_amp", k)
            cn = c_mode[:, :, n]
            APE = Hh .* Ap .^ 2 ./ (4 .* rho0 .* cn .^ 2) ./ 1e3                 # [kJ m-2]
            KE = 0.25 .* rho0 .* Hh .* (Au .^ 2 .+ Av .^ 2) ./ 1e3
            lim = round(quantile(filter(isfinite, masked(APE)), 0.99); sigdigits = 2)
            strip_map!(fig, row, APE, "APE_$n = HA_p² / (4ρ₀c_$(n)²), mode $n"; colormap = :thermal, colorrange = (0, lim),
                       last_row = row == nm)
            Am, As, _ = strip_stats(APE)
            Km, _, _ = strip_stats(KE)
            ok = rows_ok(Am)
            al = strip_line!(fig, row, "APE_$n averaged across the strip (±$(round(Int, side_km)) km, ξ = $(ix_west)–$(ix_east))",
                             "energy [kJ m⁻²]", vcat((Am .+ As)[ok], Km[ok]); last_row = row == nm, ymin0 = true)
            push!(axl, al)
            band!(al, line_km, max.(Am .- As, 0), Am .+ As; color = (:crimson, 0.15), label = "APE ± 1 std across the strip")
            lines!(al, line_km, Km; color = :royalblue, linestyle = :dash, linewidth = 2, label = "KE = ¼ρ₀H(A_u² + A_v²), strip mean")
            lines!(al, line_km, Am; color = :crimson, linewidth = 2.5, label = "APE = HA_p² / (4ρ₀c²), strip mean")
            finish_line!(al, ok)
            @printf("  %s mode %d APE strip mean: %.2f kJ/m² (KE %.2f, KE/APE %.2f); at the node APE %.2f, KE %.2f\n",
                    c, n, mean(Am[ok]), mean(Km[ok]), mean(Km[ok]) / mean(Am[ok]), Am[j_node], Km[j_node])
        end
        Label(fig[0, :], "$c modal APE (harmonic fit), mean across ±$(round(Int, side_km)) km of the mooring line, $dff_run"; fontsize = 11pt, font = :bold)
        Legend(fig[nm + 1, :], axl[end]; orientation = :horizontal, framevisible = false, labelsize = 9pt, nbanks = 2)
        Label(fig[nm + 2, :], strip_note; fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(figure_path, "binplot_harm_APE_$(c).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("saved $fname")
    end
end

# ============================================================================
# (6) total energy E_n = KE_n + APE_n and the ratio KE_n / APE_n, averaged across the ±side_km strip
# ============================================================================
# Two rows per mode: (a) map of E_n, and the strip means of E_n, KE_n, APE_n; (b) map of KE_n / APE_n at each point,
# and the ratio of the strip means (KE strip mean / APE strip mean) with the 25-75 % range of the point ratios across
# the strip. Reference: a free progressive wave has KE/APE = (ω² + f²) / (ω² - f²) (f at the mooring line, row by row);
# a pressure node of a partly standing wave has KE > APE, a velocity node KE < APE.
# Constituents HARM_E_CONS (default every constituent in the file). Figures: figure_path/binplot_harm_energy_<con>.png
harm_e_env = get(ENV, "HARM_E_CONS", "all")
harm_e_cons = harm_e_env == "all" ? harm_cons : String.(split(harm_e_env, r"[,:; ]+"; keepempty = false))
# 25th and 75th percentiles of a field across the strip, per eta row (rows with too little deep water: NaN)
function strip_quartiles(a)
    q1, q3 = fill(NaN, ny), fill(NaN, ny)
    for j in 1:ny
        v = [a[i, j] for i in strip if show[i, j] && isfinite(a[i, j])]
        length(v) / length(strip) >= min_cover || continue
        q1[j], q3[j] = quantile(v, 0.25), quantile(v, 0.75)
    end
    q1, q3
end
if 6 in bin_steps
    println("\ntotal energy and KE/APE: rho0 = $rho0 kg m-3")
    harm_freq = Float64.(dsh["freq"][:])                  # [deg/hour]
    for c in harm_e_cons
        k = findfirst(==(c), harm_cons)
        k === nothing && (println("  $c not in $harm_fname"); continue)
        ω = deg2rad(harm_freq[k]) / 3600                  # [rad/s]
        f_line = 2 * 7.2921e-5 .* sind.(lat[ix_line, :])  # Coriolis along the mooring line [rad/s]
        r_free = (ω^2 .+ f_line .^ 2) ./ (ω^2 .- f_line .^ 2)   # free-wave KE/APE (ω > f; NaN-free at these latitudes)
        nm = length(harm_modes)
        fig = Figure(size = (fig_w, 1600nm), fontsize = 10pt)
        axE, axR = Axis[], Axis[]
        for (im, n) in enumerate(harm_modes)
            rE, rR = 2im - 1, 2im                         # figure rows of this mode
            Ap, Au, Av = rd_h("p$(n)_amp", k), rd_h("u$(n)_amp", k), rd_h("v$(n)_amp", k)
            APE = Hh .* Ap .^ 2 ./ (4 .* rho0 .* c_mode[:, :, n] .^ 2) ./ 1e3   # [kJ m-2]
            KE = 0.25 .* rho0 .* Hh .* (Au .^ 2 .+ Av .^ 2) ./ 1e3
            E = KE .+ APE
            R = KE ./ APE
            # (a) total energy
            lim = round(quantile(filter(isfinite, masked(E)), 0.99); sigdigits = 2)
            strip_map!(fig, rE, E, "E_$n = KE_$n + APE_$n, mode $n"; colormap = :thermal, colorrange = (0, lim))
            Em, Es, _ = strip_stats(E)
            Km, _, _ = strip_stats(KE)
            Am, _, _ = strip_stats(APE)
            ok = rows_ok(Em)
            al = strip_line!(fig, rE, "E_$n averaged across the strip (±$(round(Int, side_km)) km, ξ = $(ix_west)–$(ix_east))",
                             "energy [kJ m⁻²]", vcat((Em .+ Es)[ok], Km[ok], Am[ok]); ymin0 = true)
            push!(axE, al)
            band!(al, line_km, max.(Em .- Es, 0), Em .+ Es; color = (:gray50, 0.18), label = "E ± 1 std")
            lines!(al, line_km, Km; color = :royalblue, linestyle = :dash, linewidth = 2, label = "KE strip mean")
            lines!(al, line_km, Am; color = :crimson, linestyle = :dash, linewidth = 2, label = "APE strip mean")
            lines!(al, line_km, Em; color = :black, linewidth = 2.5, label = "E = KE + APE")
            finish_line!(al, ok)
            # (b) KE / APE
            strip_map!(fig, rR, log2.(R), "KE_$n / APE_$n, mode $n (log₂ scale)"; colormap = :balance, colorrange = (-1, 1),
                       last_row = im == nm, cbticks = ([-1, -0.5, 0, 0.5, 1], ["0.5", "0.71", "1", "1.41", "2"]))
            Rm = Km ./ Am                                 # ratio of the strip means
            q1, q3 = strip_quartiles(R)
            okr = ok .& isfinite.(Rm)
            ar = strip_line!(fig, rR, "KE_$n / APE_$n across the strip", "KE / APE",
                             Rm[okr]; last_row = im == nm, ymin0 = true, ymax = 2.5)
            push!(axR, ar)
            band!(ar, line_km, q1, q3; color = (:purple, 0.15), label = "KE/APE 25–75 % across the strip")
            lines!(ar, line_km, r_free; color = :gray30, linestyle = :dash, linewidth = 1.5, label = "free wave (ω²+f²)/(ω²−f²)")
            lines!(ar, line_km, Rm; color = :purple, linewidth = 2.5, label = "KE / APE of the strip means")
            finish_line!(ar, okr)
            @printf("  %s mode %d: E strip mean %.2f kJ/m² (KE %.2f, APE %.2f); KE/APE %.2f (free wave %.3f); at the node E %.2f, KE/APE %.2f\n",
                    c, n, mean(Em[ok]), mean(Km[ok]), mean(Am[ok]), mean(Km[ok]) / mean(Am[ok]), mean(r_free[ok]),
                    Em[j_node], Rm[j_node])
        end
        Label(fig[0, :], "$c modal energy (harmonic fit), mean across ±$(round(Int, side_km)) km of the mooring line, $dff_run"; fontsize = 11pt, font = :bold)
        leg = fig[2nm + 1, :] = GridLayout()
        Legend(leg[1, 1], axE[end], "energy"; orientation = :horizontal, framevisible = false, labelsize = 9pt, nbanks = 2, titlesize = 9pt)
        Legend(leg[1, 2], axR[end], "ratio"; orientation = :horizontal, framevisible = false, labelsize = 9pt, nbanks = 2, titlesize = 9pt)
        Label(fig[2nm + 2, :], strip_note * "\nKE = ¼ρ₀H(A_u² + A_v²), APE = HA_p²/(4ρ₀cₙ²); ratio map (log₂): white = equipartition, red: KE > APE, blue: KE < APE; ratio panels clipped at 2.5"; fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(figure_path, "binplot_harm_energy_$(c).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("saved $fname")
    end
end

# ============================================================================
# (7) barotropic-to-baroclinic conversion conv_n, averaged across the ±side_km strip
# ============================================================================
# conv_n = ½ A_W A_p,n φ_n(-H) cos(g_W - g_p,n) [W m-2] from W_BT_make.jl (bt_conv_harmonic.nc; W_BT = -(ubar dh/dx +
# vbar dh/dy) from the barotropic fit, p_n from the modal fit, same tc). Named conv_n: c_n is the mode speed.
# One row per mode: map of conv_n (red: barotropic -> mode n, blue: back to the barotropic tide), and its strip mean
# ± 1 std. Printed: the integral over the strip up to the crop [MW]. Constituents HARM_CONV_CONS (default every one).
# Figures: figure_path/binplot_harm_conv_<con>.png
conv_fname = joinpath(bcef_dir, "bt_conv_harmonic.nc")
harm_conv_env = get(ENV, "HARM_CONV_CONS", "all")
harm_conv_cons = harm_conv_env == "all" ? harm_cons : String.(split(harm_conv_env, r"[,:; ]+"; keepempty = false))
if 7 in bin_steps
    isfile(conv_fname) || error("no $conv_fname: run W_BT_make.jl (steps 1-5) first")
    dsc = NCDataset(conv_fname)
    area = 1 ./ (pm .* pn)                                # [m2] per cell (DFF crop)
    in_strip = falses(nx, ny); in_strip[strip, 1:j_crop] .= true
    println("\nconversion conv_n from $conv_fname")
    for c in harm_conv_cons
        nm = length(harm_modes)
        all(n -> haskey(dsc, "conv_$(c)_mode$n"), harm_modes) || (println("  $c not in $conv_fname"); continue)
        fig = Figure(size = (fig_w, 800nm), fontsize = 10pt)
        axl = Axis[]
        for (row, n) in enumerate(harm_modes)
            cv = Float64.(nomissing(dsc["conv_$(c)_mode$n"][1:nx, 1:ny], NaN)) .* 1e3   # [mW m-2]
            lim = round(quantile(abs.(filter(isfinite, masked(cv))), 0.99); sigdigits = 2)
            strip_map!(fig, row, cv, "conv_$n, mode $n\n½A_W A_p φ_$(n)(−H) cos(g_W − g_p)"; colormap = :balance,
                       colorrange = (-lim, lim), last_row = row == nm)
            Cm, Cs, _ = strip_stats(cv)
            ok = rows_ok(Cm)
            al = strip_line!(fig, row, "conv_$n averaged across the strip (±$(round(Int, side_km)) km, ξ = $(ix_west)–$(ix_east))",
                             "conv_$n [mW m⁻²]", vcat((Cm .- Cs)[ok], (Cm .+ Cs)[ok]); last_row = row == nm)
            push!(axl, al)
            band!(al, line_km, Cm .- Cs, Cm .+ Cs; color = (:crimson, 0.15), label = "conv ± 1 std across the strip")
            lines!(al, line_km, Cm; color = :crimson, linewidth = 2.5, label = "conv_n strip mean (+: barotropic → mode n)")
            finish_line!(al, ok)
            sel = in_strip .& show .& isfinite.(cv)
            I = sum(cv[sel] .* area[sel]) / 1e3 / 1e6     # [MW]
            Ipos = sum(max.(cv[sel], 0) .* area[sel]) / 1e3 / 1e6
            @printf("  %s mode %d: strip integral (to %.0f km) %.1f MW (positive part %.1f MW); strip-mean peak %.2f mW/m² at %.0f km\n",
                    c, n, line_km[j_crop], I, Ipos, maximum(Cm[ok]), line_km[findall(ok)[argmax(Cm[ok])]])
        end
        Label(fig[0, :], "$c conversion conv_n (harmonic fit), mean across ±$(round(Int, side_km)) km of the mooring line, $dff_run"; fontsize = 11pt, font = :bold)
        Legend(fig[nm + 1, :], axl[end]; orientation = :horizontal, framevisible = false, labelsize = 9pt)
        Label(fig[nm + 2, :], strip_note * "\nconv_n = <W_BT p_n,bot>, W_BT = −(ubar ∂h/∂x + vbar ∂h/∂y), p_n,bot = p_n φ_n(−H); red: barotropic → mode n, blue: mode n → barotropic"; fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(figure_path, "binplot_harm_conv_$(c).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("saved $fname")
    end
    close(dsc)
end

# ============================================================================
# (8) divergence of the harmonic-fit flux, ∇·F_n, with the conversion conv_n
# ============================================================================
# F_n = (Fx_<con>_mode<n>, Fy_<con>_mode<n>) of bcef_modes<N>_harmonic.nc (½ H A_p A_w cos Δg, grid xi / eta) on the
# curvilinear grid: ∇·F = pm pn [∂ξ(Fx / pn) + ∂η(Fy / pm)] (centred differences; NaN where a difference reaches the
# shelf or an edge band). Steady energy balance of mode n: conv_n = ∇·F_n + D_n (D_n: dissipation and transfer to
# other modes / frequencies), so conv_n − ∇·F_n is what is lost locally. One row per mode: map of ∇·F_n, and the strip
# means of ∇·F_n, conv_n and conv_n − ∇·F_n. Printed: strip integrals up to the crop [MW].
# Constituents HARM_DIV_CONS (default every one). Figures: figure_path/binplot_harm_div_<con>.png
harm_div_env = get(ENV, "HARM_DIV_CONS", "all")
harm_div_cons = harm_div_env == "all" ? harm_cons : String.(split(harm_div_env, r"[,:; ]+"; keepempty = false))
# centred difference per grid step along dim (NaN at the ends)
function cdiff(a, dim)
    d = fill(NaN, size(a))
    if dim == 1
        d[2:end-1, :] .= (a[3:end, :] .- a[1:end-2, :]) ./ 2
    else
        d[:, 2:end-1] .= (a[:, 3:end] .- a[:, 1:end-2]) ./ 2
    end
    d
end
flux_div(Fx, Fy) = pm .* pn .* (cdiff(Fx ./ pn, 1) .+ cdiff(Fy ./ pm, 2))
rd2(d, v) = Float64.(nomissing(d[v][1:nx, 1:ny], NaN))
# ∇·F_n and conv_n of constituent c, mode n [mW m-2] (∇·F NaN off the deep, inside-the-edges region)
function div_conv(dsc, c, n)
    dv = flux_div(rd2(dsh, "Fx_$(c)_mode$n"), rd2(dsh, "Fy_$(c)_mode$n")) .* 1e3
    dv[.!show] .= NaN
    dv, rd2(dsc, "conv_$(c)_mode$n") .* 1e3
end
if 8 in bin_steps
    isfile(conv_fname) || error("no $conv_fname: run W_BT_make.jl (steps 1-5) first")
    dsc = NCDataset(conv_fname)
    area = 1 ./ (pm .* pn)
    in_strip = falses(nx, ny); in_strip[strip, 1:j_crop] .= true
    println("\nflux divergence ∇·F_n (harmonic fit) and conv_n")
    for c in harm_div_cons
        nm = length(harm_modes)
        haskey(dsh, "Fx_$(c)_mode$(harm_modes[1])") || (println("  $c: no flux in $harm_fname"); continue)
        fig = Figure(size = (fig_w, 800nm), fontsize = 10pt)
        axl = Axis[]
        for (row, n) in enumerate(harm_modes)
            dv, cv = div_conv(dsc, c, n)                                         # [mW m-2]
            lim = round(quantile(abs.(filter(isfinite, masked(cv))), 0.99); sigdigits = 2)   # the conv_n map range
            strip_map!(fig, row, dv, "∇·F_$n, mode $n"; colormap = :balance, colorrange = (-lim, lim), last_row = row == nm)
            Dm, Ds, _ = strip_stats(dv)
            Cm, _, _ = strip_stats(cv)
            Rm = Cm .- Dm
            ok = rows_ok(Dm) .& isfinite.(Cm)
            al = strip_line!(fig, row, "∇·F_$n and conv_$n averaged across the strip (±$(round(Int, side_km)) km)",
                             "[mW m⁻²]", vcat((Dm .- Ds)[ok], (Dm .+ Ds)[ok], Cm[ok], Rm[ok]); last_row = row == nm)
            push!(axl, al)
            band!(al, line_km, Dm .- Ds, Dm .+ Ds; color = (:royalblue, 0.15), label = "∇·F ± 1 std across the strip")
            lines!(al, line_km, Rm; color = :gray40, linestyle = :dot, linewidth = 2, label = "conv − ∇·F (local loss D)")
            lines!(al, line_km, Cm; color = :crimson, linewidth = 2, label = "conv_n strip mean")
            lines!(al, line_km, Dm; color = :royalblue, linewidth = 2.5, label = "∇·F_n strip mean")
            finish_line!(al, ok)
            sel = in_strip .& isfinite.(dv) .& isfinite.(cv)
            Iv, Ic = sum(dv[sel] .* area[sel]) / 1e9, sum(cv[sel] .* area[sel]) / 1e9   # [MW]
            @printf("  %s mode %d: strip integrals (to %.0f km): ∇·F %.1f MW, conv %.1f MW, conv − ∇·F %.1f MW\n",
                    c, n, line_km[j_crop], Iv, Ic, Ic - Iv)
        end
        Label(fig[0, :], "$c flux divergence ∇·F_n and conversion (harmonic fit), mean across ±$(round(Int, side_km)) km, $dff_run"; fontsize = 11pt, font = :bold)
        Legend(fig[nm + 1, :], axl[end]; orientation = :horizontal, framevisible = false, labelsize = 9pt, nbanks = 2)
        Label(fig[nm + 2, :], strip_note * "\n∇·F = pm pn [∂ξ(Fx/pn) + ∂η(Fy/pm)], F = ½HA_pA_w cos Δg; red on the map: flux leaving (source), blue: converging (sink); map range as conv_n";
              fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(figure_path, "binplot_harm_div_$(c).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("saved $fname")
    end
    close(dsc)
end

# ============================================================================
# (9) local loss D_n = conv_n − ∇·F_n
# ============================================================================
# The steady energy balance of mode n, conv_n = ∇·F_n + D_n, gives D_n: energy taken out of the coherent mode-n tide
# locally (dissipation, transfer to other modes, to the overtides / other frequencies, or to the incoherent part).
# Positive: loss; negative: energy gained from other modes / frequencies. One row per mode: map of D_n, and its strip
# mean ± 1 std with the 200 km node; printed: strip integral to the crop and its positive part [MW].
# Constituents HARM_D_CONS (default every one). Figures: figure_path/binplot_harm_D_<con>.png
harm_d_env = get(ENV, "HARM_D_CONS", "all")
harm_d_cons = harm_d_env == "all" ? harm_cons : String.(split(harm_d_env, r"[,:; ]+"; keepempty = false))
if 9 in bin_steps
    isfile(conv_fname) || error("no $conv_fname: run W_BT_make.jl (steps 1-5) first")
    dsc = NCDataset(conv_fname)
    area = 1 ./ (pm .* pn)
    in_strip = falses(nx, ny); in_strip[strip, 1:j_crop] .= true
    println("\nlocal loss D_n = conv_n − ∇·F_n (harmonic fit)")
    for c in harm_d_cons
        nm = length(harm_modes)
        haskey(dsh, "Fx_$(c)_mode$(harm_modes[1])") || (println("  $c: no flux in $harm_fname"); continue)
        fig = Figure(size = (fig_w, 800nm), fontsize = 10pt)
        axl = Axis[]
        for (row, n) in enumerate(harm_modes)
            dv, cv = div_conv(dsc, c, n)
            D = cv .- dv                                                         # [mW m-2]
            lim = round(quantile(abs.(filter(isfinite, masked(D))), 0.99); sigdigits = 2)
            strip_map!(fig, row, D, "D_$n = conv_$n − ∇·F_$n, mode $n"; colormap = :balance, colorrange = (-lim, lim),
                       last_row = row == nm)
            Dm, Ds, _ = strip_stats(D)
            ok = rows_ok(Dm)
            al = strip_line!(fig, row, "D_$n averaged across the strip (±$(round(Int, side_km)) km, ξ = $(ix_west)–$(ix_east))",
                             "D_$n [mW m⁻²]", vcat((Dm .- Ds)[ok], (Dm .+ Ds)[ok]); last_row = row == nm)
            push!(axl, al)
            band!(al, line_km, Dm .- Ds, Dm .+ Ds; color = (:darkgreen, 0.15), label = "D ± 1 std across the strip")
            lines!(al, line_km, Dm; color = :darkgreen, linewidth = 2.5, label = "D_n = conv_n − ∇·F_n, strip mean (+: loss)")
            finish_line!(al, ok)
            sel = in_strip .& isfinite.(D)
            I, Ip = sum(D[sel] .* area[sel]) / 1e9, sum(max.(D[sel], 0) .* area[sel]) / 1e9   # [MW]
            seg(a, b) = (s_ = sel .& (a .<= line_km' .< b); sum(D[s_] .* area[s_]) / 1e9)
            @printf("  %s mode %d: D strip integral %.1f MW (positive part %.1f); slope (< 200 km) %.1f, 200-400 km %.1f, 400 km - crop %.1f MW\n",
                    c, n, I, Ip, seg(0, 200), seg(200, 400), seg(400, line_km[j_crop] + 1))
        end
        Label(fig[0, :], "$c local loss D_n = conv_n − ∇·F_n (harmonic fit), mean across ±$(round(Int, side_km)) km, $dff_run"; fontsize = 11pt, font = :bold)
        Legend(fig[nm + 1, :], axl[end]; orientation = :horizontal, framevisible = false, labelsize = 9pt)
        Label(fig[nm + 2, :], strip_note * "\nsteady balance conv_n = ∇·F_n + D_n; red: energy leaves the coherent mode-n tide here (dissipation, other modes / frequencies, incoherent), blue: gained";
              fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(figure_path, "binplot_harm_D_$(c).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("saved $fname")
    end
    close(dsc)
end

# ============================================================================
# (10) bathymetry: the mooring-line transect and the strip statistics, per grid row
# ============================================================================
# Depth along the mooring line (xi = ix_line) and, per grid row (eta), over the ±side_km strip: mean ± 1 std and the
# shallowest / deepest point (all water points of the strip, the shelf included, so the coast shows where the strip
# reaches it); the share of the strip shallower than h_mask marks where the strip rows are cut from the line plots.
# Test figure: ~/figure_check/binplot_bathy.png
# per grid row over the strip's water points (the shelf included): mean, std, shallowest, deepest depth, the share of the
# strip shallower than h_mask; and the depth along the mooring line (NaN on land)
function strip_bathy()
    wet = NCDataset(g -> Float64.(g["mask_rho"][1:nx, 1:ny]), run_grid_fname) .> 0
    hm_, hs_, hmin, hmax, shelf = fill(NaN, ny), fill(NaN, ny), fill(NaN, ny), fill(NaN, ny), fill(NaN, ny)
    for j in 1:ny
        v = [h[i, j] for i in strip if wet[i, j]]
        isempty(v) && continue
        hm_[j], hs_[j], hmin[j], hmax[j] = mean(v), length(v) > 1 ? std(v) : 0.0, minimum(v), maximum(v)
        shelf[j] = count(<(h_mask), v) / length(strip)
    end
    hm_, hs_, hmin, hmax, shelf, ifelse.(wet[ix_line, :], h[ix_line, :], NaN)
end
if 10 in bin_steps
    hm_, hs_, hmin, hmax, shelf, line_h = strip_bathy()
    fig = Figure(size = (fig_w, 760), fontsize = 10pt)
    ax = Axis(fig[1, 1]; title = "bathymetry along the mooring line (ξ = $ix_line) and across the ±$(round(Int, side_km)) km strip (ξ = $(ix_west)–$(ix_east))",
              xlabel = "distance along the mooring line from the south boundary [km]", ylabel = "depth [m]",
              xticks = 0:50:ceil(Int, line_km[end]), width = 1150, height = 520)
    jw = findall(isfinite.(hm_) .& ((1:ny) .<= j_crop))
    band!(ax, line_km, -hmax, -hmin; color = (:gray60, 0.25), label = "strip: shallowest – deepest point")
    band!(ax, line_km, -(hm_ .+ hs_), -(hm_ .- hs_); color = (:royalblue, 0.25), label = "strip: mean ± 1 std")
    lines!(ax, line_km, -hm_; color = :royalblue, linewidth = 2.5, label = "strip mean")
    lines!(ax, line_km, -line_h; color = :black, linewidth = 2, label = "mooring line (transect)")
    hlines!(ax, [-h_mask]; color = :gray40, linestyle = :dot, linewidth = 1)
    text!(ax, line_km[jw[end]], -h_mask; text = "$(Int(h_mask)) m (shelf cut) ", align = (:right, :bottom), fontsize = 8pt, color = :gray30)
    ylo = -maximum(filter(isfinite, hmax[jw])) * 1.05
    ytop = 0.12 * -ylo
    for (nm, j, on) in moor_in
        on || continue
        vlines!(ax, [line_km[j]]; color = :gray60, linewidth = 0.8)
        text!(ax, line_km[j], ytop; text = nm * " ", rotation = π / 2, align = (:right, :center), fontsize = 7pt, color = :gray35)
    end
    vlines!(ax, [line_km[j_node]]; color = :red, linestyle = :dash, linewidth = 1.2)
    text!(ax, line_km[j_node], ylo; text = " 200 km node", align = (:left, :bottom), fontsize = 8pt, color = :red)
    j0 = findfirst(r -> isfinite(hm_[r]), 1:ny)
    xlims!(ax, line_km[j0], line_km[j_crop]); ylims!(ax, ylo, ytop)
    # first row where the strip passes min_cover deep water (rows before it are not drawn in steps 3-9)
    jcut = findfirst(j -> isfinite(shelf[j]) && 1 - shelf[j] >= min_cover, 1:ny)
    jcut === nothing || (vspan!(ax, [line_km[j0]], [line_km[jcut]]; color = (:orange, 0.12));
                         text!(ax, line_km[j0], ylo; text = " rows < $(round(Int, 100min_cover))% deep", align = (:left, :bottom), fontsize = 8pt, color = :darkorange))
    axislegend(ax; position = :rc, framevisible = true, labelsize = 9pt, backgroundcolor = (:white, 0.8))
    Label(fig[2, 1], "strip statistics over the water points of ξ = $(ix_west)–$(ix_east) in each grid row (the shelf included);\n" *
          "orange: rows not drawn in the strip-average figures (< $(round(Int, 100min_cover))% of the strip deeper than $(Int(h_mask)) m); $dff_run grid"; fontsize = 8pt, tellwidth = false)
    resize_to_layout!(fig)
    fname = joinpath(check_path, "binplot_bathy.png")
    save(fname, fig; px_per_unit = 2)
    @printf("\nbathymetry: strip rows drawn from %.0f km (first row with >= %.0f%% deep water); line depth at the node %.0f m, strip mean %.0f m\n",
            jcut === nothing ? NaN : line_km[jcut], 100min_cover, line_h[j_node], hm_[j_node])
    println("saved $fname")
end

# ============================================================================
# (11) DFF of the harmonic flux (DFF_on_harmonic.jl, harm_dff_flux.nc), averaged across the strip
# ============================================================================
# Per constituent, mode and sector scheme (half: north / south; quad: north / west / south / east; grid coordinates),
# three panels along the mooring line: (1) the total flux (all directions) Fy (along the line, ± 1 std across the strip)
# and Fx (across it); (2) Fy of every sector and the cross term (all − Σ sectors); (3) bathymetry (mooring line, strip
# mean ± 1 std). Constituents HARM_HDFF_CONS (default every one). Figures: figure_path/harm_dff/binplot_hdff_<scheme>_<con>_mode<n>.png
hdff_fname = joinpath(bcef_dir, "harm_dff_flux.nc")
harm_hdff_env = get(ENV, "HARM_HDFF_CONS", "all")
hdff_path = joinpath(figure_path, "harm_dff")
harm_hdff_cons = harm_hdff_env == "all" ? harm_cons : String.(split(harm_hdff_env, r"[,:; ]+"; keepempty = false))
hdff_schemes = [("half", ["north", "south"]), ("quad", ["north", "west", "south", "east"])]
sec_col = Dict("north" => :crimson, "south" => :royalblue, "west" => :darkorange, "east" => :seagreen, "cross" => :gray40)
sec_rng = Dict(("half", "north") => "0–180°", ("half", "south") => "180–360°", ("quad", "north") => "45–135°",
               ("quad", "west") => "135–225°", ("quad", "south") => "225–315°", ("quad", "east") => "315–45°")
if 11 in bin_steps
    isfile(hdff_fname) || error("no $hdff_fname: run DFF_on_harmonic.jl first")
    mkpath(hdff_path)
    dsf = NCDataset(hdff_fname)
    hm_, hs_, hmin, hmax, shelf, line_h = strip_bathy()
    println("\nDFF of the harmonic flux from $hdff_fname")
    for c in harm_hdff_cons, n in harm_modes, (sch, secs) in hdff_schemes
        haskey(dsf, "Fy_$(c)_$(sch)_all_mode$n") || (println("  $c $sch mode $n not in the file"); continue)
        rdf(v) = Float64.(nomissing(dsf[v][1:nx, 1:ny], NaN)) ./ 1e3        # [kW m-1]
        Fya, Fya_s, _ = strip_stats(rdf("Fy_$(c)_$(sch)_all_mode$n"))
        Fxa, _, _ = strip_stats(rdf("Fx_$(c)_$(sch)_all_mode$n"))
        Fst = Dict(s_ => strip_stats(rdf("Fy_$(c)_$(sch)_$(s_)_mode$n")) for s_ in [secs; "cross"])
        Fys = Dict(s_ => v[1] for (s_, v) in Fst)          # strip means
        Fn_s = Fst["north"][2]                             # std across the strip of the north sector (dominant)
        ok = rows_ok(Fya)
        x0, x1 = line_km[findfirst(ok)], line_km[j_crop]
        fig = Figure(size = (fig_w, 1300), fontsize = 10pt)
        W = 930
        # (1) total
        a1 = Axis(fig[1, 1]; title = "(1) total flux (all directions), mean across the strip", ylabel = "[kW m⁻¹]",
                  xticks = 0:50:ceil(Int, line_km[end]), width = W, height = 300)
        band!(a1, line_km, Fya .- Fya_s, Fya .+ Fya_s; color = (:black, 0.10), label = "F_y ± 1 std across the strip")
        lines!(a1, line_km, Fxa; color = :gray40, linestyle = :dash, linewidth = 2, label = "F_x (across the line)")
        lines!(a1, line_km, Fya; color = :black, linewidth = 2.5, label = "F_y (along the line)")
        # (2) sectors
        a2 = Axis(fig[2, 1]; title = "(2) F_y by direction ($sch scheme, θ from grid ξ, grid η = 90°)", ylabel = "F_y [kW m⁻¹]",
                  xticks = 0:50:ceil(Int, line_km[end]), width = W, height = 300)
        band!(a2, line_km, Fys["north"] .- Fn_s, Fys["north"] .+ Fn_s; color = (sec_col["north"], 0.15), label = "north ± 1 std across the strip")
        for s_ in secs
            lines!(a2, line_km, Fys[s_]; color = sec_col[s_], linewidth = 2.2, label = "$s_ ($(sec_rng[(sch, s_)]))")
        end
        lines!(a2, line_km, Fys["cross"]; color = sec_col["cross"], linestyle = :dot, linewidth = 2, label = "cross (all − Σ sectors)")
        lines!(a2, line_km, Fya; color = :black, linewidth = 1, linestyle = :dash, label = "all")
        # (3) bathymetry
        a3 = Axis(fig[3, 1]; title = "(3) bathymetry", ylabel = "depth [m]", xlabel = "distance along the mooring line from the south boundary [km]",
                  xticks = 0:50:ceil(Int, line_km[end]), width = W, height = 200)
        band!(a3, line_km, -(hm_ .+ hs_), -(hm_ .- hs_); color = (:royalblue, 0.25), label = "strip mean ± 1 std")
        lines!(a3, line_km, -hm_; color = :royalblue, linewidth = 2, label = "strip mean")
        lines!(a3, line_km, -line_h; color = :black, linewidth = 1.8, label = "mooring line")
        ylims!(a3, -maximum(filter(isfinite, hmax[ok])) * 1.05, 0)
        # y ranges (with headroom for the mooring labels), moorings, node, crop
        for (ax, vals) in ((a1, vcat((Fya .- Fya_s)[ok], (Fya .+ Fya_s)[ok], Fxa[ok])),
                           (a2, vcat([Fys[s_][ok] for s_ in [secs; "cross"]]..., (Fys["north"] .+ Fn_s)[ok], (Fys["north"] .- Fn_s)[ok])), (a3, nothing))
            if vals !== nothing
                ylo, yhi = extrema(vcat(filter(isfinite, vals), 0.0))
                ylims!(ax, ylo - 0.05 * (yhi - ylo), yhi + 0.35 * (yhi - ylo))
                hlines!(ax, [0]; color = :black, linewidth = 0.6)
            end
            ytop = ax.finallimits[].origin[2] + ax.finallimits[].widths[2]
            for (nm, j, on) in moor_in
                on || continue
                vlines!(ax, [line_km[j]]; color = :gray70, linewidth = 0.6)
                ax === a1 && text!(ax, line_km[j], ytop; text = nm * " ", rotation = π / 2, align = (:right, :center), fontsize = 7pt, color = :gray35)
            end
            vlines!(ax, [line_km[j_node]]; color = :red, linestyle = :dash, linewidth = 1.2)
            xlims!(ax, x0, x1)
        end
        hidexdecorations!(a1; grid = false, ticks = false); hidexdecorations!(a2; grid = false, ticks = false)
        for (r, ax) in enumerate((a1, a2, a3))   # legends to the right of the panels, clear of the curves
            Legend(fig[r, 2], ax; framevisible = false, labelsize = 8pt, halign = :left, tellheight = false)
        end
        Label(fig[0, 1:2], "$c mode-$n flux by direction (harmonic DFF, $sch), mean across ±$(round(Int, side_km)) km\n$dff_run"; fontsize = 11pt, font = :bold)
        Label(fig[4, 1:2], strip_note; fontsize = 8pt)
        rowgap!(fig.layout, 8)
        resize_to_layout!(fig)
        fname = joinpath(hdff_path, "binplot_hdff_$(sch)_$(c)_mode$(n).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        @printf("  %s mode %d %s: strip-mean F_y (to the crop) all %.2f, %s, cross %.2f kW/m\n", c, n, sch, mean(Fya[ok]),
                join([@sprintf("%s %.2f", s_, mean(filter(isfinite, Fys[s_][ok]))) for s_ in secs], ", "), mean(filter(isfinite, Fys["cross"][ok])))
        println("saved $fname")
    end
    close(dsf)
end

# ============================================================================
# (12) DFF of the time series (DFF_on_bcef.jl): time-mean flux by direction, averaged across the strip
# ============================================================================
# The time-mean maps of bcef_dff_flux_modes<N>.nc (<Fx|Fy>_<band>_<sector>_<mode>, period 1 = full record; sectors
# all / north 0-180° / south 180-360° / cross = all - north - south, grid coordinates), averaged per grid row across
# the ±side_km strip (as steps 3-11). Three panels along the mooring line:
#   (1) |F| of the strip-mean flux vector (Fx, Fy) of every term, north ± 1 std of |F_N| across the strip
#   (2) the same on a log scale (the weak south-going flux and cross terms)
#   (3) bathymetry (mooring line, strip mean ± 1 std)
# Check: the strip-mean Fy of the hourly strip file (bcef_dff_flux_modes<N>_transect_strip_*.nc, DFF_on_bcef.jl)
# averaged over the record must equal the strip mean of the time-mean map (same rows, same deep-water points).
# Bands DFF_BIN_BANDS, modes DFF_BIN_MODES (default: every band and mode in the file, modesum included).
# Figures: figure_path/dff/binplot_dff_<band>_<mode>.png
dff_modes_all = unique([split(k, "_")[4] for k in dff_vars])
dff_bin_env, dff_mode_env = get(ENV, "DFF_BIN_BANDS", "all"), get(ENV, "DFF_BIN_MODES", "all")
dff_bin_bands = dff_bin_env == "all" ? String.(dff_bands) : String.(split(dff_bin_env, r"[,:; ]+"; keepempty = false))
dff_bin_modes = dff_mode_env == "all" ? String.(dff_modes_all) : String.(split(dff_mode_env, r"[,:; ]+"; keepempty = false))
dff_path = joinpath(figure_path, "dff")
band_desc = Dict("M2" => "M2 band, periods 9-15 h", "hp24" => "periods < 24 h", "hp9" => "periods < 9 h")
dff_col = Dict("all" => :black, "north" => :crimson, "south" => :royalblue, "cross" => :gray40)
if 12 in bin_steps
    hm_, hs_, hmin, hmax, shelf, line_h = strip_bathy()
    mkpath(dff_path)
    sfile = filter(f -> occursin(Regex("bcef_dff_flux_modes$(NMODE)_transect_strip_xi$(ix_west)-$(ix_east)\\.nc"), f), readdir(bcef_dir))
    println("\nDFF of the time series, time-mean flux across the strip ($(dff_fname))")
    for b in dff_bin_bands, m in dff_bin_modes
        haskey(dsd, "Fy_$(b)_all_$(m)") || (println("  $b $m not in $dff_fname"); continue)
        rdm(c, sec) = Float64.(nomissing(dsd["$(c)_$(b)_$(sec)_$(m)"][:, :, 1], NaN)) ./ 1e3   # full record [kW m-1]
        secs = ["all", "north", "south", "cross"]
        Fx = Dict(sec => strip_stats(rdm("Fx", sec))[1] for sec in secs)
        Fy = Dict(sec => strip_stats(rdm("Fy", sec))[1] for sec in secs)
        Fmag = Dict(sec => hypot.(Fx[sec], Fy[sec]) for sec in secs)     # |F| of the strip-mean vector
        _, magN_s, _ = strip_stats(hypot.(rdm("Fx", "north"), rdm("Fy", "north")))   # spread of |F_N| across the strip
        ratio = Fmag["south"] ./ Fmag["north"]
        ok = rows_ok(Fy["all"])
        # check against the hourly strip file
        if !isempty(sfile)
            Fh = NCDataset(d -> vec(mean(Float64.(d["Fy_$(b)_all_$(m)"][:, :]); dims = 2)), joinpath(bcef_dir, sfile[1])) ./ 1e3
            cover = NCDataset(d -> Float64.(d["strip_cover"][:]), joinpath(bcef_dir, sfile[1]))
            ck = ok .& (cover .> 0)
            @printf("  %s %s check: strip-mean Fy, time mean of the hourly strip file vs strip mean of the map: max |diff| %.2e kW/m (max %.2f)\n",
                    b, m, maximum(abs.(Fh[ck] .- Fy["all"][ck])), maximum(abs.(Fy["all"][ck])))
        end
        x0, x1 = line_km[findfirst(ok)], line_km[j_crop]
        fig = Figure(size = (fig_w, 1300), fontsize = 10pt)
        W = 930
        pos(a) = ifelse.(ok .& (a .> 0), a, NaN)          # log axis: positive values only
        yhi = maximum(filter(isfinite, (Fmag["north"] .+ magN_s)[ok]))
        # |F| of every term on one axis; logy: log10 y axis (positive values only)
        function flux_terms!(ax, logy)
            f(a) = logy ? pos(a) : ifelse.(ok, a, NaN)
            band!(ax, line_km, f(max.(Fmag["north"] .- magN_s, logy ? 1e-6 : 0.0)), f(Fmag["north"] .+ magN_s);
                  color = (dff_col["north"], 0.15), label = "north ± 1 std of |F_N| across the strip")
            lines!(ax, line_km, f(Fmag["cross"]); color = dff_col["cross"], linestyle = :dash, linewidth = 1.8, label = "cross terms")
            lines!(ax, line_km, f(Fmag["south"]); color = dff_col["south"], linewidth = 2.2, label = "south going (180–360°)")
            lines!(ax, line_km, f(Fmag["north"]); color = dff_col["north"], linewidth = 2.2, label = "north going (0–180°)")
            lines!(ax, line_km, f(Fmag["all"]); color = dff_col["all"], linewidth = 2.5, label = "all directions")
        end
        a1 = Axis(fig[1, 1]; title = "(1) time-mean energy flux magnitude |F|, mean across the strip", ylabel = "|F| [kW m⁻¹]",
                  xticks = 0:50:ceil(Int, line_km[end]), width = W, height = 330)
        flux_terms!(a1, false)
        a2 = Axis(fig[2, 1]; title = "(2) the same, log scale", ylabel = "|F| [kW m⁻¹]", yscale = log10,
                  xticks = 0:50:ceil(Int, line_km[end]), width = W, height = 330)
        flux_terms!(a2, true)
        a3 = Axis(fig[3, 1]; title = "(3) bathymetry", ylabel = "depth [m]", xlabel = "distance along the mooring line from the south boundary [km]",
                  xticks = 0:50:ceil(Int, line_km[end]), width = W, height = 200)
        band!(a3, line_km, -(hm_ .+ hs_), -(hm_ .- hs_); color = (:royalblue, 0.25), label = "strip mean ± 1 std")
        lines!(a3, line_km, -hm_; color = :royalblue, linewidth = 2, label = "strip mean")
        lines!(a3, line_km, -line_h; color = :black, linewidth = 1.8, label = "mooring line")
        ylims!(a3, -maximum(filter(isfinite, hmax[ok])) * 1.05, 0)
        ytop1 = 1.3yhi
        ylims!(a1, 0, ytop1)
        # log range: from the smallest south / cross value (cut at 1e-3 of the maximum) to above the north band
        ylo = max(minimum(filter(isfinite, vcat(pos(Fmag["south"]), pos(Fmag["cross"])))), 1e-3yhi)
        ylims!(a2, ylo / 1.5, yhi * 2)
        for ax in (a1, a2, a3)
            for (nm, j, on) in moor_in
                on || continue
                vlines!(ax, [line_km[j]]; color = :gray70, linewidth = 0.6)
                ax === a1 && text!(ax, line_km[j], ytop1; text = nm * " ", rotation = π / 2, align = (:right, :center), fontsize = 7pt, color = :gray35)
            end
            vlines!(ax, [line_km[j_node]]; color = :red, linestyle = :dash, linewidth = 1.2)
            xlims!(ax, x0, x1)
        end
        hidexdecorations!(a1; grid = false, ticks = false); hidexdecorations!(a2; grid = false, ticks = false)
        Legend(fig[1:2, 2], a1; framevisible = false, labelsize = 8pt, halign = :left, tellheight = false)
        Legend(fig[3, 2], a3; framevisible = false, labelsize = 8pt, halign = :left, tellheight = false)
        mlabel = m == "modesum" ? "modes 1-$(NMODE) (sum)" : "mode-$(replace(m, "mode" => ""))"
        Label(fig[0, 1:2], "$mlabel energy flux, $(get(band_desc, b, b)), mean across ±$(round(Int, side_km)) km of the mooring line\n" *
              "DFF north 0–180° / south 180–360° (grid); $dff_run, $(replace(split(split(periods, ";")[1], "= ")[end], "T00:00:00" => ""))"; fontsize = 11pt, font = :bold)
        Label(fig[4, 1:2], strip_note; fontsize = 8pt)
        rowgap!(fig.layout, 8)
        resize_to_layout!(fig)
        fname = joinpath(dff_path, "binplot_dff_$(b)_$(m).png")
        save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        @printf("  %s %s: strip-mean |F| (to the crop) all %.2f, north %.2f, south %.2f, cross %.2f kW/m; median |F_S|/|F_N| %.3f\n",
                b, m, mean(Fmag["all"][ok]), mean(Fmag["north"][ok]), mean(Fmag["south"][ok]), mean(Fmag["cross"][ok]), median(filter(isfinite, ratio[ok])))
        println("saved $fname")
    end
end
