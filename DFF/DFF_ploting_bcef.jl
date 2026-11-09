using NCDatasets, Dates, Statistics, CairoMakie

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")   # dff_run, run_dir, ddf_dir, fig_dir (set by DFF_RUN)
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # topdown_axis, plot_curvilinear!, quiver_curvilinear!
include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings

# Figures of the north / south energy flux of modes 1..NMODE from DFF_on_bcef.jl (bcef_dff_flux_modes<NMODE>.nc), the flux
# version of DFF_ploting.jl, one folder per mode: DFF_energyflux/modes<NMODE>/<mode1|mode2|...|modesum>/
# (modesum = modes 1..NMODE together; the mode-1-only version, for bcef_dff_flux.nc, is DFF_ploting_bcef_mode1.jl):
#   flux_map_<group>_<period>.png             |F| north, |F| south (with arrows), and log10(|F_S| / |F_N|)
#   flux_map_<group>_fixedclim_<period>.png   the same, with one colour range and arrow scale for north and south
#   flux_cross_<band>_<period>.png            |F all|, |F_N + F_S|, |cross|
#   hovmoller_dir_<group>_<line>_<period>.png Fy along the line, north and south going
#   hovmoller_all_<band>_<line>_<period>.png  full Fy and cross terms along the line
#   transect/flux_line_<group>_<line>_<period>.png  time-mean |F| of each term along the line, and the south / north ratio
# The flux is saved as period means and along the three lines only, so there are no hourly map frames as for the ssh.
# The small-sector figures are drawn only when the DFF was run with them (BCEF_SMALLSEC=true in DFF_on_bcef.jl).
# Shelf (h < h_deep) and edge bands are masked: the modes mean nothing on the shelf (cn -> 0), and the DFF leaks at the edges.
NMODE = parse(Int, get(ENV, "NMODE", "3"))
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, "ddf_bcef"))
flux_fname = joinpath(bcef_dir, "bcef_dff_flux_modes$(NMODE).nc")
transect_fname(lname, ix) = joinpath(bcef_dir, "bcef_dff_flux_modes$(NMODE)_transect_$(lname)_xi$(ix).nc")
# which modes to draw: BCEF_MODES=mode1,modesum (default: every mode and their sum)
mode_tags = split(get(ENV, "BCEF_MODES", join([["mode$n" for n in 1:NMODE]; "modesum"], ",")), ",")
mode_desc(m) = m == "modesum" ? "modes 1-$NMODE (sum)" : replace(m, "mode" => "mode-")
fig_root = joinpath(dirname(fig_dir), "DFF_energyflux", "modes$(NMODE)")
figure_path = fig_root   # set per mode in the loop below
grid_fname = run_grid_fname   # from dff_run.jl
run_mooring = Dict("amazon_2700m_cmems_0801" => "roms_cmems_grd_2700m_mor_edata.nc")
mooring_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/$(get(run_mooring, dff_run, "roms_grd_900m_mor_edata.nc"))"
edge = 60        # points masked along each edge, as in DFF_transect_analysis.jl
h_deep = 200     # [m]
arrow_skip = 40  # one arrow every 40 grid points (~36 km)
only_cross = get(ENV, "BCEF_ONLY_CROSS", "false") == "true"   # true: redraw only the flux_cross figures (a few minutes)
only_fixedclim = get(ENV, "BCEF_ONLY_FIXEDCLIM", "false") == "true"   # true: draw only the flux_map fixedclim figures
only_lines = get(ENV, "BCEF_ONLY_LINES", "false") == "true"   # true: draw only the transect line plots
draw_all = !(only_cross || only_fixedclim || only_lines)

# ============================================================================
# (1) grid, moorings, flux maps
# ============================================================================
ds = NCDataset(flux_fname)
nx_f, ny_f, nt = ds.dim["xi_rho"], ds.dim["eta_rho"], ds.dim["time"]
lon_rho, lat_rho = ds["lon_rho"][:, :], ds["lat_rho"][:, :]
date = ds["ocean_time"][:]
h, pn, grid_angle = NCDataset(grid_fname) do g
    g["h"][1:nx_f, 1:ny_f], g["pn"][1:nx_f, 1:ny_f], g["angle"][1:nx_f, 1:ny_f]
end
# period names and dates, as saved by DFF_on_bcef.jl (index = third dimension of the flux maps)
periods = [(strip(first(split(s, "="))), strip(last(split(s, "=")))) for s in split(ds.attrib["periods"], ";")]

deep = (h .>= h_deep) .& (edge .< (1:nx_f) .<= nx_f - edge) .& (edge .< (1:ny_f)' .<= ny_f - edge)
masked(a) = ifelse.(deep, a, NaN32)
fit_lims = (extrema(lon_rho[deep]) .+ (-0.05, 0.05), extrema(lat_rho[deep]) .+ (-0.05, 0.05))   # map limits around the unmasked data

moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
moor_ij = NCDataset(mooring_file) do g
    [(g["$(n)_mor"][1] + 1, g["$(n)_mor"][2] + 1) for n in moor_name]   # fractional Julia (xi, eta)
end
moor_group = [replace(n, r"\d+$" => "") for n in moor_name]
moor_style = Dict("french" => (marker = :star5,   markersize = 11, color = :magenta),
                  "M"      => (marker = :diamond, markersize = 7,  color = :cyan),
                  "CPIES"  => (marker = :circle,  markersize = 6,  color = :yellow))
tr_lines = [("west", round(Int, 0.5 + nx_f / 4)), ("mooring", round(Int, 0.5 + nx_f / 2)), ("east", round(Int, 0.5 + 3nx_f / 4))]

function add_mooring!(ax)
    for (_, ix) in tr_lines
        lines!(ax, Point3f.(lon_rho[ix, :], lat_rho[ix, :], 0); color = :black, linestyle = :dash, linewidth = 0.8, overdraw = true)
    end
    for g in unique(moor_group)
        k = findall(==(g), moor_group)
        st = get(moor_style, g, (marker = :rect, markersize = 6, color = :white))
        scatter!(ax, Point3f.(moor_lon[k], moor_lat[k], 0); marker = st.marker, markersize = st.markersize,
                 color = st.color, strokecolor = :black, strokewidth = 0.75, overdraw = true)
    end
end

bands = [("M2", "M2 band, periods 9–15 h"), ("hp24", "periods shorter than 24 h"), ("hp9", "periods shorter than 9 h")]
# (group suffix, north sector, south sector, description)
has_small = haskey(ds, "Fx_M2_north_smallsec_mode1")
sector_sets = [("", "north", "south", "north 0–180°, south 180–360°")]
has_small && push!(sector_sets, ("_smallsec", "north_smallsec", "south_smallsec", "small sectors: north 60–120°, south 240–300°"))
println("$dff_run: $flux_fname; modes ", join(mode_tags, ", "), has_small ? "" : "; no small sectors in the file")
line_secs = has_small ? ("all", "north", "south", "north_smallsec", "south_smallsec") : ("all", "north", "south")
cur_mode = mode_tags[1]   # the mode being drawn, set in the loop below
read_flux(c, b, sec, ip) = nomissing(ds["$(c)_$(b)_$(sec)_$(cur_mode)"][:, :, ip], NaN32)
flux_vec(b, sec, ip) = (read_flux("Fx", b, sec, ip), read_flux("Fy", b, sec, ip))
mag(F) = sqrt.(F[1] .^ 2 .+ F[2] .^ 2)
# Fx, Fy are along the grid axes; the arrows are drawn on a lon/lat map, so rotate them to east, north
to_en(F) = (F[1] .* cos.(grid_angle) .- F[2] .* sin.(grid_angle), F[1] .* sin.(grid_angle) .+ F[2] .* cos.(grid_angle))

stride = 2   # every 2nd grid point in the colour maps, as in DFF_ploting.jl
sub(a) = a[1:stride:end, 1:stride:end]
lon_s, lat_s, h_s = sub(lon_rho), sub(lat_rho), sub(h)

# isobaths: depths to contour; lims: (xlims, ylims) of the map; ref: (arrow size [kW/m], its length [°]) of a reference
# arrow drawn in the top-left corner, which is outside the data
function flux_panel!(pos, title, data, cmap, crange; ylabel = "", arrows = nothing, ascale = 0.0,
                     isobaths = [h_deep], lims = nothing, ref = nothing)
    ax = topdown_axis(pos; title = title, ylabel = ylabel, backgroundcolor = :gray80)
    hm = plot_curvilinear!(ax, lon_s, lat_s, sub(masked(data)); colormap = cmap, colorrange = crange)
    contour!(ax, lon_s, lat_s, h_s; levels = isobaths, color = :gray85, linewidth = 1, labels = length(isobaths) > 1,
             labelsize = 9, labelcolor = :gray85)
    if arrows !== nothing
        Fe, Fn = to_en(arrows)
        quiver_curvilinear!(ax, lon_rho, lat_rho, masked(Fe), masked(Fn); skip = arrow_skip, lengthscale = ascale,
                            color = :white, shaftwidth = 1, tipwidth = 5, tiplength = 5)
    end
    add_mooring!(ax)
    lims === nothing || limits!(ax, lims...)
    if ref !== nothing
        x0, y0 = lims[1][1] + 0.05 * (lims[1][2] - lims[1][1]), lims[2][2] - 0.06 * (lims[2][2] - lims[2][1])
        arrows2d!(ax, [x0], [y0], [0.0], [ref[2]], [0.0], [0.0]; color = :black, shaftwidth = 1.5, tipwidth = 6, tiplength = 6)
        text!(ax, x0, y0; text = "$(ref[1]) kW/m", align = (:left, :top), offset = (0, -5), fontsize = 11)
    end
    hm
end

# ============================================================================
# (2) period-mean flux maps, and the deep-ocean numbers
# ============================================================================
# The ratio is |F_S| / |F_N| on a log10 scale, red = south dominates. Arrows: an arrow of the panel's colour-bar maximum
# is 0.4° long. The cross terms belong to the half-plane split only (all = north + south + cross).
function plot_flux_map(b, bdesc, suffix, nsec, ssec, sdesc, ip, p, pdates; fixedclim = false)
    Fn, Fs = flux_vec(b, nsec, ip), flux_vec(b, ssec, ip)
    mn, ms = mag(Fn), mag(Fs)
    # by default each direction has its own colour range (the south flux is far weaker), to compare with the ratio
    # panel; fixedclim = true puts south on the north colour range and arrow scale, so the two compare by eye
    lim(m) = round(quantile(filter(!isnan, masked(m)), 0.99) / 1e3, RoundUp; sigdigits = 2)   # kW/m
    nmax = lim(mn)
    smax = fixedclim ? nmax : lim(ms)
    fig = Figure(size = (1500, 650))
    panels = [("|F| north going", mn ./ 1e3, :thermal, (0, nmax), "|F| [kW/m]", Fn, 0.4 / (nmax * 1e3)),
              ("|F| south going", ms ./ 1e3, :thermal, (0, smax), "|F| [kW/m]", Fs, 0.4 / (smax * 1e3)),
              ("flux ratio south / north", log10.(ms ./ mn), :balance, (-1, 1), "log₁₀(|F_S| / |F_N|)", nothing, 0.0)]
    for (k, (title, data, cmap, crange, clabel, arrows, ascale)) in enumerate(panels)
        hm = flux_panel!(fig[1, k], title, data, cmap, crange; ylabel = k == 1 ? "Latitude" : "", arrows = arrows, ascale = ascale)
        Colorbar(fig[2, k], hm; vertical = false, label = clabel, flipaxis = false)
    end
    Label(fig[0, :], "$(mode_desc(cur_mode)) energy flux, $bdesc, $sdesc\n$p: $pdates", fontsize = 15)
    fname = joinpath(figure_path, "flux_map_$(b)$(suffix)$(fixedclim ? "_fixedclim" : "")_$p.png")
    save(fname, fig)
    println("saved $fname")
end

function plot_flux_cross(b, bdesc, ip, p, pdates)
    Fa, Fn, Fs = flux_vec(b, "all", ip), flux_vec(b, "north", ip), flux_vec(b, "south", ip)
    Fc = flux_vec(b, "cross", ip)
    Fsum = (Fn[1] .+ Fs[1], Fn[2] .+ Fs[2])
    amax = round(quantile(filter(!isnan, masked(mag(Fa))), 0.99) / 1e3, RoundUp; sigdigits = 2)
    ascale = 0.4 / (amax * 1e3)
    aref = round(amax / 3; sigdigits = 1)   # reference arrow [kW/m]
    fig = Figure(size = (1500, 700))
    for (k, (title, F)) in enumerate([("|F all|", Fa), ("|F_N + F_S|", Fsum), ("|cross terms|", Fc)])
        hm = flux_panel!(fig[1, k], title, mag(F) ./ 1e3, :thermal, (0, amax); ylabel = k == 1 ? "Latitude" : "", arrows = F, ascale = ascale,
                         isobaths = [h_deep, 500, 1000, 2000], lims = fit_lims, ref = (aref, aref * 1e3 * ascale))
        Colorbar(fig[2, k], hm; vertical = false, label = "|F| [kW/m]", flipaxis = false)
    end
    Label(fig[0, :], "$(mode_desc(cur_mode)) energy flux, $bdesc: all = north + south + cross\n$p: $pdates", fontsize = 15)
    fname = joinpath(figure_path, "flux_cross_$(b)_$p.png")
    save(fname, fig)
    println("saved $fname")
end

function draw_maps()
    println("$dff_run $(mode_desc(cur_mode)): deep-ocean means (h >= $h_deep m, $edge edge points masked) of Fy [W/m]; |cross| / |all| from point-by-point values")
    for (b, bdesc) in bands, (ip, (p, pdates)) in enumerate(periods)
        fy(sec) = read_flux("Fy", b, sec, ip)[deep]
        small = has_small ? "north_smallsec $(round(mean(fy("north_smallsec")), sigdigits = 3)), south_smallsec $(round(mean(fy("south_smallsec")), sigdigits = 3)), " : ""
        println("  $b $p: all $(round(mean(fy("all")), sigdigits = 3)), north $(round(mean(fy("north")), sigdigits = 3)), ",
                "south $(round(mean(fy("south")), sigdigits = 3)), $(small)cross $(round(mean(fy("cross")), sigdigits = 3)); ",
                "|cross| / |all| = $(round(mean(abs, fy("cross")) / mean(abs, fy("all")), digits = 2))")
        for (suffix, nsec, ssec, sdesc) in sector_sets
            draw_all && plot_flux_map(b, bdesc, suffix, nsec, ssec, sdesc, ip, p, pdates)
            (draw_all || only_fixedclim) && plot_flux_map(b, bdesc, suffix, nsec, ssec, sdesc, ip, p, pdates; fixedclim = true)
        end
        (draw_all || only_cross) && plot_flux_cross(b, bdesc, ip, p, pdates)
    end
end

# ============================================================================
# (3) Hovmöller of Fy along the three grid columns, each period and the one-day zoom
# ============================================================================
# y = distance along the line from its grid-south (shelf) end, x = time. Fy > 0 (red) is energy going grid north, along
# the line. The flux of one wave pulses at twice its frequency about a mean of its own sign. Colour ranges are shared by
# all panels of a band and all three lines (per period); the shelf and edge bands are masked.
check_day = Date(2022, 9, 4)   # same one-day zoom as DFF_ploting.jl
chunk(d1, d2) = (a = findfirst(>=(DateTime(d1)), date); b = findlast(<(DateTime(d2)), date);
                 a === nothing || b === nothing || b <= a ? nothing : a:b)
hov_periods = filter(x -> x[2] !== nothing,
                     [("full", 1:nt), ("augsep", chunk(Date(2022, 8, 1), Date(2022, 10, 1))),
                      ("oct", chunk(Date(2022, 10, 1), Date(2022, 11, 1))), ("nov", chunk(Date(2022, 11, 1), Date(2022, 12, 1))),
                      ("day", chunk(check_day, check_day + Day(1)))])
pstr(tidx) = "$(Dates.format(date[tidx[1]], "yyyy-mm-dd HH:MM")) – $(Dates.format(date[tidx[end]], "yyyy-mm-dd HH:MM"))"

hov = Dict{String, Any}()
# Fy of the current mode along the three lines, every hour (one mode at a time: ~0.6 GB per line and mode)
function load_hov!()
    empty!(hov)
    for (lname, ix) in tr_lines
        dist = [0; cumsum(0.5 .* (1 ./ pn[ix, 1:end-1] .+ 1 ./ pn[ix, 2:end]))] ./ 1e3   # [km] along the line
        dist_at(j) = dist[floor(Int, j)] + (j - floor(j)) * (dist[floor(Int, j)+1] - dist[floor(Int, j)])
        k_on = findall(k -> abs(moor_ij[k][1] - ix) < 1, eachindex(moor_name))
        f = NCDataset(transect_fname(lname, ix)) do g
            Dict((b, sec) => ifelse.(deep[ix, :], nomissing(g["Fy_$(b)_$(sec)_$(cur_mode)"][:, :], NaN32), NaN32)
                 for (b, _) in bands for sec in line_secs)
        end
        for (b, _) in bands
            f[(b, "cross")] = f[(b, "all")] .- f[(b, "north")] .- f[(b, "south")]
        end
        hov[lname] = (ix = ix, dist = dist, moor = moor_name[k_on], moor_dist = [dist_at(moor_ij[k][2]) for k in k_on], f = f)
    end
end

function hov_panel!(ax, L, t_ax, data, cmax; label = false)
    hm = heatmap!(ax, t_ax, L.dist, permutedims(data ./ 1e3); colormap = :balance, colorrange = (-cmax, cmax), nan_color = :gray80)
    isempty(L.moor_dist) || hlines!(ax, L.moor_dist; color = :magenta, linestyle = :dash, linewidth = 1.5)
    label && !isempty(L.moor) && text!(ax, fill(t_ax[end], length(L.moor)), L.moor_dist; text = L.moor, align = (:right, :bottom),
                                       fontsize = 10, color = :magenta, font = :bold, offset = (-3, 1))
    hm
end

function plot_hov_period(p, ptidx)
    span_h = length(ptidx) - 1
    tunit, tdiv = span_h <= 72 ? ("hours", 1) : ("days", 24)
    t_ax = (0:span_h) ./ tdiv
    xlab = "$tunit since $(Dates.format(date[ptidx[1]], "yyyy-mm-dd HH:MM"))"
    ylab = "distance along line [km]\n(grid south → north)"
    width = clamp(span_h + 200, 900, 2400)
    for (b, bdesc) in bands
        # colour range [kW/m]: 99th percentile of |Fy| over the three lines, north and south going
        vals = reduce(vcat, [vec(abs.(hov[l].f[(b, sec)][:, ptidx])) for (l, _) in tr_lines for sec in ("north", "south")])
        cmax = round(quantile(filter(!isnan, vals), 0.99) / 1e3, RoundUp; sigdigits = 2)
        for (lname, ix) in tr_lines
            L = hov[lname]
            tag = "$(lname)_xi$(ix)_$p"
            # (file name, figure title, panels)
            figs = [("hovmoller_dir_$(b)", "north 0–180° / south 180–360°", [("north going", "north"), ("south going", "south")]),
                    ("hovmoller_all_$(b)", "all = north + south + cross", [("all directions", "all"), ("cross terms", "cross")])]
            has_small && push!(figs, ("hovmoller_dir_$(b)_smallsec", "small sectors", [("north 60–120°", "north_smallsec"), ("south 240–300°", "south_smallsec")]))
            for (fname, ftitle, panels) in figs
                fig = Figure(size = (width, 950))
                hm = nothing
                for (i, (title, sec)) in enumerate(panels)
                    ax = Axis(fig[i, 1]; title = title, xlabel = i == 2 ? xlab : "", ylabel = ylab)
                    hm = hov_panel!(ax, L, t_ax, L.f[(b, sec)][:, ptidx], cmax; label = true)
                end
                Colorbar(fig[3, 1], hm; vertical = false, label = "Fy [kW/m]", flipaxis = false, width = Relative(0.4))
                Label(fig[0, 1], "$(mode_desc(cur_mode)) Fy, $bdesc, $ftitle, $lname line (xi = $ix), $(pstr(ptidx))", fontsize = 13, tellwidth = false)
                save(joinpath(figure_path, "$(fname)_$tag.png"), fig)
            end
            println("saved hovmoller_{dir,all}_$(b)*_$tag.png (± $cmax kW/m)")
        end
    end
end


# ============================================================================
# (4) time-mean flux magnitude of each term along the three lines, as the transect/ figures of the ssh
# ============================================================================
# |F| = |(Fx, Fy)| of the period-mean flux, so it includes the flux across the line too. Top: the terms; bottom: the
# south / north ratio. Half-plane figure: all, north, south, cross; small-sector figure: the small sectors, with the
# half-plane north and south as thin lines. Only the deep interior is drawn.

function plot_flux_line(b, bdesc, suffix, nsec, ssec, sdesc, ip, p, pdates)
    mags = Dict(sec => mag(flux_vec(b, sec, ip)) ./ 1e3 for sec in unique(["all", "north", "south", "cross", nsec, ssec]))   # kW/m
    # (term, label, colour, line width, line style)
    terms = suffix == "" ?
            [("all", "all directions", :black, 2.5, :solid), ("north", "north going", :firebrick, 2, :solid),
             ("south", "south going", :royalblue, 2, :solid), ("cross", "cross terms", :gray50, 2, :dash)] :
            [("north", "north 0–180°", :firebrick, 1, :solid), ("south", "south 180–360°", :royalblue, 1, :solid),
             (nsec, "north 60–120°", :firebrick, 2.5, :solid), (ssec, "south 240–300°", :royalblue, 2.5, :solid)]
    for (lname, ix) in tr_lines
        Ll = hov[lname]
        keep = deep[ix, :]
        jj = findall(keep)
        line(sec) = ifelse.(keep, mags[sec][ix, :], NaN32)   # NaN outside the deep interior: lines! leaves a gap there
        top = 1.15 * maximum(maximum(line(t[1])[jj]) for t in terms)

        fig = Figure(size = (1500, 800))
        ax1 = Axis(fig[1, 1]; title = "time-mean energy flux magnitude |F|", ylabel = "|F| [kW/m]")
        for (sec, label, color, lw, ls) in terms
            lines!(ax1, Ll.dist, line(sec); color = color, linewidth = lw, linestyle = ls, label = label)
        end
        ylims!(ax1, 0, top)
        Legend(fig[1, 2], ax1; framevisible = false)
        ax2 = Axis(fig[2, 1]; title = "flux ratio south / north", ylabel = "|F_S| / |F_N|", yscale = log10,
                   xlabel = "distance along line [km] (grid south → north)")
        lines!(ax2, Ll.dist, line(ssec) ./ line(nsec); color = :black, linewidth = 2)
        hlines!(ax2, [1]; color = :gray50, linestyle = :dot)
        for ax in (ax1, ax2)
            xlims!(ax, Ll.dist[jj[1]], Ll.dist[jj[end]])
            isempty(Ll.moor_dist) || vlines!(ax, Ll.moor_dist; color = :magenta, linestyle = :dash, linewidth = 1.5)
        end
        isempty(Ll.moor) || text!(ax1, Ll.moor_dist, fill(top, length(Ll.moor)); text = Ll.moor, rotation = π / 2,
                                  align = (:right, :bottom), fontsize = 9, color = :magenta, font = :bold, offset = (-2, -3))
        linkxaxes!(ax1, ax2)
        Label(fig[0, :], "$(mode_desc(cur_mode)) energy flux, $bdesc, $sdesc, $lname line (xi = $ix)\n$p: $pdates; deep interior (h ≥ $h_deep m)", fontsize = 15, tellwidth = false)
        save(joinpath(figure_path, "transect", "flux_line_$(b)$(suffix)_$(lname)_xi$(ix)_$p.png"), fig)
    end
    println("saved transect/flux_line_$(b)$(suffix)_*_$p.png")
end

# ============================================================================
# (5) every mode: maps, Hovmöllers, line plots, each in DFF_energyflux/modes<NMODE>/<mode>/
# ============================================================================
for m in mode_tags
    global cur_mode = m
    global figure_path = joinpath(fig_root, m)
    mkpath(joinpath(figure_path, "transect"))
    println("\n==== $(mode_desc(m)) -> $figure_path")
    draw_maps()
    (draw_all || only_lines) && load_hov!()
    for (p, ptidx) in (draw_all ? hov_periods : [])
        plot_hov_period(p, ptidx)
        GC.gc()
    end
    if draw_all || only_lines
        for (b, bdesc) in bands, (ip, (p, pdates)) in enumerate(periods), (suffix, nsec, ssec, sdesc) in sector_sets
            plot_flux_line(b, bdesc, suffix, nsec, ssec, sdesc, ip, p, pdates)
        end
    end
    empty!(hov)
    GC.gc()
end
close(ds)
println("figures in $fig_root/<mode>")
