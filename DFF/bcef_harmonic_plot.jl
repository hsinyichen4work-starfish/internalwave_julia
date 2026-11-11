#= bcef_harmonic_plot.jl
Hsin-Yi Chen, USM DMS
Top-view maps of the harmonic fit of the modal pressure and velocity (mode_bcef_harmonic.jl,
<run>/ddf_bcef/bcef_modes<N>_harmonic.nc): one figure per mode and constituent,
    row 1: amplitude A' of pn, un, vn        row 2: phase g' (relative to the record centre tc) of pn, un, vn
un, vn are along the grid xi / eta axes (the mooring line is a grid column, so vn is along it, un across it).
The phase uses a cyclic colour map (0° = 360°); g' increases in the direction a wave travels (a crest passes later),
so the phase gradient shows the propagation direction and the spacing of a 360° cycle the wavelength.
Masked as the DFF plots: the shelf (h < h_deep) and the edge bands. Figures: figure/<run>/DFF_harmonic/
    harm_<con>_mode<n>.png        A' and g' of pn, un, vn
    harm_fluxline_<con>_mode<n>.png  F_x, F_y maps with their sections along the mooring line below
    harm_flux_<con>_mode<n>.png   the energy flux of the constituent from the fit, rows Fx (w = u) and Fy (w = v):
        ½ H A_p A_w  |  ½ H A_p A_w cos(g'_p - g'_w) = F (the net flux)  |  ½ H A_p A_w sin(g'_p - g'_w)
      cos ≈ the amplitude product: p and w in phase, a progressive wave carrying all it could; a large sin part:
      p and w ~90° apart, as in a standing pattern (energy sloshing back and forth, no net flux).
The 200 km node of the mooring line (as mode_analysis.jl: distance along the line from its grid-south end) is marked.
Usage:
  julia DFF/bcef_harmonic_plot.jl                                  # 3-month run, every constituent and mode in the file
  DFF_RUN=amazon_900m_pure_or HARM_PLOT_CONS=M2 julia DFF/bcef_harmonic_plot.jl
  FLUX_CONS=M2 NODE_KM=178 julia DFF/bcef_harmonic_plot.jl     # flux figures for M2 only; node marker at 178 km
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
edge_km = parse(Float64, get(ENV, "EDGE_KM", "54"))   # [km] masked along each edge (sponge / boundary): 60 points on the 900 m grid
dx_km = NCDataset(g -> mean(1 ./ Float64.(g["pm"][:, :])) / 1e3, run_grid_fname)   # mean grid spacing [km]
edge = round(Int, edge_km / dx_km)                     # the same distance in grid points (900 m: 60, 2700 m: 20)
println("edge bands: $edge_km km = $edge points (grid spacing $(round(dx_km, digits = 2)) km)")
h_deep = 200.0     # [m] shelf masked
isobaths = [1000, 2000, 3000]   # [m]
stride = 2         # every 2nd grid point in the colour maps, as the DFF plots
FLUX_CONS_ENV = get(ENV, "FLUX_CONS", "all")   # constituents of the flux figures: "all" (as HARM_PLOT_CONS) or e.g. M2,M4
node_km = parse(Float64, get(ENV, "NODE_KM", "200"))            # [km] node marker along the mooring line

## --- read the fit
ds = NCDataset(harm_fname)
cons = String.(ds["con"][:])
PLOT_CONS = PLOT_CONS_ENV == "all" ? cons : String.(split(PLOT_CONS_ENV, r"[,:; ]+"; keepempty = false))
FLUX_CONS = FLUX_CONS_ENV == "all" ? cons : String.(split(FLUX_CONS_ENV, r"[,:; ]+"; keepempty = false))
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

# one constituent of a fitted field, reopening the file and retrying on a Lustre read error (HDF -101 / EIO)
function rd(var, k; tries = 5)
    for t in 1:tries
        try
            return NCDataset(g -> Float64.(g[var][:, :, k]), harm_fname)
        catch e
            t == tries && rethrow()
            wait_s = 60 * 2^(t - 1)
            println("  read $var [$k] failed ($(sprint(showerror, e)[1:min(end, 80)])), retry $t in $(wait_s ÷ 60) min"); flush(stdout)
            sleep(wait_s)
        end
    end
end

## --- moorings (the file's lon is 0..360, the mooring file's -180..180)
moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
lon360 = maximum(lon) > 180
moor_lon = lon360 ? mod.(moor_lon, 360) : moor_lon
ix_line = round(Int, 0.5 + nx / 2)   # the mooring line (middle grid column), as the DFF and mode_analysis plots
pn_line = NCDataset(g -> Float64.(g["pn"][ix_line, 1:ny]), run_grid_fname)
line_km = [0; cumsum(0.5 .* (1 ./ pn_line[1:end-1] .+ 1 ./ pn_line[2:end]))] ./ 1e3   # [km] from the grid-south end
j_node = argmin(abs.(line_km .- node_km))
node_lon, node_lat = lon[ix_line, j_node], lat[ix_line, j_node]
@printf("node marker: %.0f km along the line, (i, j) = (%d, %d), h = %.0f m\n", line_km[j_node], ix_line, j_node, h[ix_line, j_node])

function decorate!(ax; node_label = true)
    contour!(ax, lon_s, lat_s, h_s; levels = isobaths, color = :gray40, linewidth = 0.7)
    lines!(ax, lon[ix_line, :], lat[ix_line, :]; color = :black, linestyle = :dash, linewidth = 0.8)
    scatter!(ax, moor_lon, moor_lat; marker = :circle, markersize = 6, color = :yellow, strokewidth = 0.6)
    scatter!(ax, [node_lon], [node_lat]; marker = :star5, markersize = 16, color = :red, strokecolor = :black, strokewidth = 0.8)
    node_label && text!(ax, node_lon, node_lat; text = "$(round(Int, node_km)) km node", align = (:left, :center), offset = (10, 0),
          fontsize = 11, color = :red, font = :bold, strokecolor = :white, strokewidth = 2)
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
        A = masked(rd("$(var)_amp", k) .* sc)
        g = masked(rd("$(var)_pha", k))
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
          "masked: h < $(Int(h_deep)) m and $(round(Int, edge_km)) km ($edge points) at the edges; grey: $(join(isobaths, " / ")) m isobaths; dashed: mooring line; yellow: moorings; red star: $(round(Int, node_km)) km node";
          fontsize = 15)
    fname = joinpath(figure_path, "harm_$(c)_mode$(n).png")
    save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()   # free the figure (8 GB limit on the login node)
    println("saved $fname")
end

## --- energy flux of a constituent from the fit: ½ H A_p A_w, its cos and sin parts (w = u: Fx, w = v: Fy)
Hfit = Float64.(ds["H"][:, :])
for c in FLUX_CONS, n in modes
    k = findfirst(==(c), cons)
    k === nothing && (println("  $c not in the file, skipped"); continue)
    Ap, gp = rd("p$(n)_amp", k), rd("p$(n)_pha", k)
    fig = Figure(size = (1600, 1150))
    for (row, (w, comp, dir)) in enumerate([("u", "x", "grid ξ, across the line"), ("v", "y", "grid η, along the line")])
        Aw, gw = rd("$(w)$(n)_amp", k), rd("$(w)$(n)_pha", k)
        P = 0.5 .* Hfit .* Ap .* Aw ./ 1e3          # ½ H A_p A_w [kW m-1]
        dg = gp .- gw
        panels = [(masked(P), "½ H A_p A_$w", "½HA_pA_$w [kW m⁻¹]", :thermal),
                  (masked(P .* cosd.(dg)), "F_$comp = ½ H A_p A_$w cos(g′_p − g′_$w)", "F_$comp [kW m⁻¹]", :balance),
                  (masked(P .* sind.(dg)), "½ H A_p A_$w sin(g′_p − g′_$w)", "[kW m⁻¹]", :balance)]
        lim = round(quantile(filter(isfinite, panels[1][1]), 0.99); sigdigits = 2)   # one range per row
        for (col, (data, title, clabel, cmap)) in enumerate(panels)
            ax = topdown_axis(fig[2row - 1, col]; title = "$title  ($dir)", ylabel = col == 1 ? "Latitude" : "",
                              xlabel = row == 2 ? "Longitude" : "", backgroundcolor = :gray80, titlesize = 13)
            hm = plot_curvilinear!(ax, lon_s, lat_s, sub(data); colormap = cmap, colorrange = col == 1 ? (0, lim) : (-lim, lim))
            decorate!(ax)
            Colorbar(fig[2row, col], hm; vertical = false, flipaxis = false, label = clabel)
        end
        ok = deep .& (h .>= 1000)
        @printf("  %s mode %d F%s: mean over h >= 1000 m: ½HA_pA_%s %.0f, cos part %.0f, sin part %.0f W/m; |sin| / (½HA_pA) median %.2f\n",
                c, n, comp, w, 1e3mean(P[ok]), 1e3mean((P .* cosd.(dg))[ok]), 1e3mean((P .* sind.(dg))[ok]),
                median(abs.(sind.(dg[ok]))))
    end
    Label(fig[0, :], "$c energy flux of mode $n from the harmonic fit, $dff_run ($recs hourly records)\n" *
          "middle: the net flux F = ½HA_pA_w cos(Δg′); right: the part out of phase (standing); one colour range per row; red star: $(round(Int, node_km)) km node";
          fontsize = 15)
    fname = joinpath(figure_path, "harm_flux_$(c)_mode$(n).png")
    save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()   # free the figure (8 GB limit on the login node)
    println("saved $fname")
end
## --- F_x and F_y maps with the sections along the mooring line next to them
# rows F_x, F_y; column 1: the map (the cos part above); columns 2-3: the same along the mooring line (x = distance
# from its grid-south end), with ½HA_pA_w (dashed, the most the wave could carry) and the sin part (dotted, out of phase).
# Text 10 pt on an 18 cm wide page, as the mode_analysis line sections.
moor_ij = NCDataset(mooring_file) do g
    [(round(Int, g["$(m)_mor"][1] + 1), round(Int, g["$(m)_mor"][2] + 1)) for m in moor_name]
end
fig_w, paper_w_cm = 1300, 18.0
pt = fig_w / (paper_w_cm / 2.54 * 72)   # pixels per point
on_line = h[ix_line, :] .>= h_deep      # shelf not drawn along the line
for c in FLUX_CONS, n in modes
    k = findfirst(==(c), cons)
    k === nothing && continue
    Ap, gp = rd("p$(n)_amp", k), rd("p$(n)_pha", k)
    fig = Figure(size = (fig_w, 1500), fontsize = 10pt)
    map_w = 400                                              # fixed panel sizes, then the figure fits the layout
    map_h = round(Int, map_w * (lims[2][2] - lims[2][1]) / ((lims[1][2] - lims[1][1]) * cosd(mean(lims[2]))))
    axl = Axis[]
    line_w = 2map_w + 60                                     # the line plot about twice the map
    for (row, (w, comp, dir)) in enumerate([("u", "x", "across the line"), ("v", "y", "along the line")])
        Aw, gw = rd("$(w)$(n)_amp", k), rd("$(w)$(n)_pha", k)
        P = 0.5 .* Hfit .* Ap .* Aw ./ 1e3          # [kW m-1]
        dg = gp .- gw
        F = P .* cosd.(dg)
        lim = round(quantile(filter(isfinite, masked(P)), 0.99); sigdigits = 2)
        ax = topdown_axis(fig[row, 1]; title = "F_$comp = ½HA_pA_$w cos(g′_p − g′_$w)\n($dir)",
                          xlabel = row == 2 ? "Longitude" : "", backgroundcolor = :gray80,
                          width = map_w, height = map_h, aspect = nothing)
        hm = plot_curvilinear!(ax, lon_s, lat_s, sub(masked(F)); colormap = :balance, colorrange = (-lim, lim))
        decorate!(ax; node_label = false)
        text!(ax, node_lon, node_lat; text = "$(round(Int, node_km)) km node", align = (:left, :center), offset = (10, 0),
              fontsize = 9pt, color = :red, font = :bold, strokecolor = :white, strokewidth = 1)
        Colorbar(fig[row, 2], hm; height = map_h)   # same unit as the line plot next to it
        # section along the line
        sec(a) = ifelse.(on_line, a[ix_line, :], NaN)
        al = Axis(fig[row, 3]; title = "F_$comp along the mooring line (ξ = $ix_line)",
                  xlabel = row == 2 ? "distance from the south boundary [km]" : "",
                  ylabel = "F_$comp [kW m⁻¹]", xticks = 0:50:ceil(Int, line_km[end]),
                  width = line_w, height = map_h)
        push!(axl, al)
        hlines!(al, [0]; color = :black, linewidth = 0.6)
        # y range from the drawn curves, with headroom at the top for the mooring labels
        ok = on_line .& (edge .< (1:ny) .<= ny - edge)
        ylo, yhi = extrema(vcat(F[ix_line, ok], P[ix_line, ok], (P .* sind.(dg))[ix_line, ok], 0.0))
        ytop = yhi + 0.45 * (yhi - ylo)
        ylims!(al, ylo - 0.05 * (yhi - ylo), ytop)
        for (nm, (i, j)) in zip(moor_name, moor_ij)
            abs(i - ix_line) <= 1 || continue
            vlines!(al, [line_km[j]]; color = :gray60, linewidth = 0.8)
            text!(al, line_km[j], ytop; text = nm * " ", rotation = π / 2, align = (:right, :center), fontsize = 7pt, color = :gray35)
        end
        vlines!(al, [line_km[j_node]]; color = :red, linestyle = :dash, linewidth = 1.5)
        lines!(al, line_km, sec(P); color = :gray40, linestyle = :dash, linewidth = 1.5, label = "½HA_pA_w (most it could carry)")
        lines!(al, line_km, sec(P .* sind.(dg)); color = :darkorange, linestyle = :dot, linewidth = 2, label = "½HA_pA_w sin(Δg′) (out of phase)")
        lines!(al, line_km, sec(F); color = :royalblue, linewidth = 2.5, label = "F = ½HA_pA_w cos(Δg′)")
        jj = findall(on_line)
        xlims!(al, line_km[first(jj)], line_km[end])
        vspan!(al, [line_km[1], line_km[end - edge + 1]], [line_km[edge], line_km[end]]; color = (:gray, 0.2))
        @printf("  %s mode %d F%s along the line: mean %.0f W/m (½HA_pA %.0f), at the node %.0f W/m (½HA_pA %.0f)\n",
                c, n, comp, 1e3mean(F[ix_line, ok]), 1e3mean(P[ix_line, ok]), 1e3F[ix_line, j_node], 1e3P[ix_line, j_node])
    end
    Label(fig[0, :], "$c energy flux of mode $n from the harmonic fit, $dff_run"; fontsize = 11pt, font = :bold)
    Legend(fig[3, :], axl[2]; orientation = :horizontal, framevisible = false, labelsize = 9pt)
    Label(fig[4, :], "maps: one colour range per panel (99th pct of ½HA_pA_w); red star / red dashed: $(round(Int, node_km)) km node; " *
          "grey lines: moorings;\ngrey shading: $(round(Int, edge_km)) km ($edge-point) edge bands; shelf (h < $(Int(h_deep)) m) not drawn"; fontsize = 8pt)
    resize_to_layout!(fig)
    fname = joinpath(figure_path, "harm_fluxline_$(c)_mode$(n).png")
    save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()   # free the figure (8 GB limit on the login node)
    println("saved $fname")
end
close(ds)
