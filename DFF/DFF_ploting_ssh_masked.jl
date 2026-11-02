using NCDatasets, Dates, Statistics, CairoMakie

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")   # dff_run, run_dir, ddf_dir, fig_dir (set by DFF_RUN)
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # topdown_axis, plot_curvilinear!
include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings

# The DFF ssh figures in the style of DFF_ploting_bcef.jl, with the shelf and edge bands masked; DFF_ploting.jl's own
# figures are not changed. Needs ssh_ddf_north_south.nc (DDF_on_ssh.jl) and the ssh_ddf_transect files (DFF_ploting.jl).
#   ssh_map_<group>_<period>.png               amplitude north, south, and log10(E_S / E_N); own colour range each
#   ssh_map_<group>_fixedclim_<period>.png     the same, with one colour range for north and south
#   hovmoller_dir_<group>_<line>_<period>.png  ssh along the line, north and south going
#   transect/ssh_line_<group>_<line>_<period>.png  ssh variance of each term along the line, and the south / north ratio
ddf_fname = joinpath(ddf_dir, "ssh_ddf_north_south.nc")
figure_path = joinpath(dirname(fig_dir), "DFF_ssh")
line_path = joinpath(figure_path, "transect")
mkpath(line_path)
grid_fname = "/expanse/lustre/projects/uso102/hchen54/input/grid/roms_grd_900m.nc"
mooring_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/roms_grd_900m_mor_edata.nc"
edge = 60      # points masked along each edge, as in DFF_transect_analysis.jl
h_deep = 200   # [m]

# ============================================================================
# (1) grid, moorings, periods, mask
# ============================================================================
ds = NCDataset(ddf_fname)
nx_f, ny_f, nt = ds.dim["xi_rho"], ds.dim["eta_rho"], ds.dim["time"]
lon_rho, lat_rho = ds["lon_rho"][:, :], ds["lat_rho"][:, :]
date = ds["ocean_time"][:]
h, pn = NCDataset(g -> (g["h"][1:nx_f, 1:ny_f], g["pn"][1:nx_f, 1:ny_f]), grid_fname)
λcut_km = round(Int, ds.attrib["M2_kcut_wavelength_m"] / 1e3)

deep = (h .>= h_deep) .& (edge .< (1:nx_f) .<= nx_f - edge) .& (edge .< (1:ny_f)' .<= ny_f - edge)
masked(a) = ifelse.(deep, a, NaN)

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

# (group, field suffix, description): ssh_north<suffix>, ssh_south<suffix>
groups = [("total", "", "all frequencies"), ("M2", "_M2", "M2 band 9–15 h, λ < $λcut_km km"), ("bc", "_bc", "all frequencies, λ < $λcut_km km"),
          ("total_smallsec", "_smallsec", "small sectors, all frequencies"),
          ("M2_smallsec", "_smallsec_M2", "small sectors, M2 band 9–15 h, λ < $λcut_km km"),
          ("bc_smallsec", "_smallsec_bc", "small sectors, all frequencies, λ < $λcut_km km")]
is_small(suffix) = occursin("smallsec", suffix)
half(suffix) = replace(suffix, "_smallsec" => "")   # the half-plane field of a small-sector group
sec_str(suffix) = is_small(suffix) ? "north 60–120°, south 240–300°" : "north 0–180°, south 180–360°"

chunk(d1, d2) = (a = findfirst(>=(DateTime(d1)), date); b = findlast(<(DateTime(d2)), date);
                 a === nothing || b === nothing || b <= a ? nothing : a:b)
check_day = Date(2022, 9, 4)   # same one-day zoom as DFF_ploting.jl
periods = filter(x -> x[2] !== nothing,
                 [("full", 1:nt), ("augsep", chunk(Date(2022, 8, 1), Date(2022, 10, 1))),
                  ("oct", chunk(Date(2022, 10, 1), Date(2022, 11, 1))), ("nov", chunk(Date(2022, 11, 1), Date(2022, 12, 1)))])
day_period = chunk(check_day, check_day + Day(1))
hov_periods = day_period === nothing ? periods : [periods; ("day", day_period)]
pstr(tidx) = "$(Dates.format(date[tidx[1]], "yyyy-mm-dd HH:MM")) – $(Dates.format(date[tidx[end]], "yyyy-mm-dd HH:MM"))"

# ============================================================================
# (2) amplitude maps: one pass over the hours, one field at a time (~5 min for the 3-month run)
# ============================================================================
# amplitude = √2 × std over time, as in DFF_ploting.jl, so the time mean is not counted. Energy ∝ amplitude², with the
# same factor for both directions at one place, so (A_S / A_N)² is the south / north energy ratio there.
function amp_field(name, stride_t = 1)
    S1 = [zeros(nx_f, ny_f) for _ in periods]
    S2 = [zeros(nx_f, ny_f) for _ in periods]
    n = zeros(Int, length(periods))
    for t in 1:stride_t:nt
        a = nomissing(ds[name][:, :, t], NaN32)
        for (ip, (_, tidx)) in enumerate(periods)
            t in tidx || continue
            S1[ip] .+= a
            S2[ip] .+= Float64.(a) .^ 2
            n[ip] += 1
        end
    end
    [sqrt.(2 .* max.(S2[ip] ./ n[ip] .- (S1[ip] ./ n[ip]) .^ 2, 0)) for ip in eachindex(periods)]
end

stride = 2   # every 2nd grid point in the colour maps, as in DFF_ploting.jl
sub(a) = a[1:stride:end, 1:stride:end]
lon_s, lat_s, h_s = sub(lon_rho), sub(lat_rho), sub(h)

function plot_ssh_map(group, suffix, desc, A_n, A_s, p, tidx; fixedclim = false)
    # by default each direction has its own colour range; fixedclim = true puts south on the north colour range
    lim(a) = round(quantile(filter(!isnan, masked(a)), 0.99) * 100, RoundUp; sigdigits = 2)   # cm
    nmax = lim(A_n)
    smax = fixedclim ? nmax : lim(A_s)
    alabel = startswith(group, "M2") ? "M2 amplitude [cm]" : "√2 × ssh std [cm]"
    fig = Figure(size = (1500, 650))
    panels = [("A north going", A_n .* 100, :thermal, (0, nmax), alabel),
              ("A south going", A_s .* 100, :thermal, (0, smax), alabel),
              ("energy ratio south / north", 2 .* log10.(A_s ./ A_n), :balance, (-1, 1), "log₁₀(E_S / E_N)")]
    for (k, (title, data, cmap, crange, clabel)) in enumerate(panels)
        ax = topdown_axis(fig[1, k]; title = title, ylabel = k == 1 ? "Latitude" : "", backgroundcolor = :gray80)
        hm = plot_curvilinear!(ax, lon_s, lat_s, sub(masked(data)); colormap = cmap, colorrange = crange)
        contour!(ax, lon_s, lat_s, h_s; levels = [h_deep], color = :gray85, linewidth = 1)
        add_mooring!(ax)
        Colorbar(fig[2, k], hm; vertical = false, label = clabel, flipaxis = false)
    end
    Label(fig[0, :], "ssh, $desc, $(sec_str(suffix))\n$p: $(pstr(tidx))", fontsize = 15)
    fname = joinpath(figure_path, "ssh_map_$(group)$(fixedclim ? "_fixedclim" : "")_$p.png")
    save(fname, fig)
    println("saved $fname")
end

function median_str(a)
    v = filter(!isnan, a[deep]) .* 100
    round(median(v), digits = 2)
end
println("$dff_run: median amplitude over the deep ocean (h >= $h_deep m, $edge edge points masked) [cm]")
for (group, suffix, desc) in groups
    A_n, A_s = amp_field("ssh_north$suffix"), amp_field("ssh_south$suffix")
    for (ip, (p, tidx)) in enumerate(periods)
        println("  $group $p: north $(median_str(A_n[ip])), south $(median_str(A_s[ip]))")
        plot_ssh_map(group, suffix, desc, A_n[ip], A_s[ip], p, tidx)
        plot_ssh_map(group, suffix, desc, A_n[ip], A_s[ip], p, tidx; fixedclim = true)
    end
    GC.gc()
end

# ============================================================================
# (3) the three lines, from the transect files of DFF_ploting.jl
# ============================================================================
hov = Dict{String, Any}()
for (lname, ix) in tr_lines
    tag = "$(Dates.format(date[1], "yyyymmddHH"))-$(Dates.format(date[end], "yyyymmddHH"))"
    fname = joinpath(ddf_dir, "ssh_ddf_transect_$(lname)_xi$(ix)_$tag.nc")
    dist = [0; cumsum(0.5 .* (1 ./ pn[ix, 1:end-1] .+ 1 ./ pn[ix, 2:end]))] ./ 1e3   # [km] along the line
    dist_at(j) = dist[floor(Int, j)] + (j - floor(j)) * (dist[floor(Int, j)+1] - dist[floor(Int, j)])
    k_on = findall(k -> abs(moor_ij[k][1] - ix) < 1, eachindex(moor_name))
    f = NCDataset(fname) do g
        Dict(n => ifelse.(deep[ix, :], nomissing(g[n][:, :], NaN32), NaN32)
             for (_, sfx, _) in groups for n in ("ssh_north$sfx", "ssh_south$sfx"))
    end
    hov[lname] = (ix = ix, dist = dist, moor = moor_name[k_on], moor_dist = [dist_at(moor_ij[k][2]) for k in k_on], f = f)
end

# ============================================================================
# (4) Hovmöller of the ssh, north and south going
# ============================================================================
# "north" waves tilt up to the right, "south" waves down. Colour ranges are shared by a group's panels and its three
# lines (per period); the shelf and edge bands are masked.
function hov_panel!(ax, L, t_ax, data, cmax)
    hm = heatmap!(ax, t_ax, L.dist, permutedims(data .* 100); colormap = :balance, colorrange = (-cmax, cmax), nan_color = :gray80)
    if !isempty(L.moor_dist)
        hlines!(ax, L.moor_dist; color = :magenta, linestyle = :dash, linewidth = 1.5)
        text!(ax, fill(t_ax[end], length(L.moor)), L.moor_dist; text = L.moor, align = (:right, :bottom),
              fontsize = 10, color = :magenta, font = :bold, offset = (-3, 1))
    end
    hm
end

function plot_hov_period(p, ptidx)
    span_h = length(ptidx) - 1
    tunit, tdiv = span_h <= 72 ? ("hours", 1) : ("days", 24)
    t_ax = (0:span_h) ./ tdiv
    xlab = "$tunit since $(Dates.format(date[ptidx[1]], "yyyy-mm-dd HH:MM"))"
    ylab = "distance along line [km]\n(grid south → north)"
    width = clamp(span_h + 200, 900, 2400)
    for (group, suffix, desc) in groups
        vals = reduce(vcat, [vec(abs.(hov[l].f["ssh_$(d)$suffix"][:, ptidx])) for (l, _) in tr_lines for d in ("north", "south")])
        cmax = round(quantile(filter(!isnan, vals), 0.99) * 100, RoundUp; sigdigits = 2)
        for (lname, ix) in tr_lines
            L = hov[lname]
            fig = Figure(size = (width, 950))
            hm = nothing
            for (i, d) in enumerate(("north", "south"))
                ax = Axis(fig[i, 1]; title = "$d going", xlabel = i == 2 ? xlab : "", ylabel = ylab)
                hm = hov_panel!(ax, L, t_ax, L.f["ssh_$(d)$suffix"][:, ptidx], cmax)
            end
            Colorbar(fig[3, 1], hm; vertical = false, label = "ssh [cm]", flipaxis = false, width = Relative(0.4))
            Label(fig[0, 1], "ssh, $desc, $lname line (xi = $ix), $(pstr(ptidx))", fontsize = 13, tellwidth = false)
            save(joinpath(figure_path, "hovmoller_dir_$(group)_$(lname)_xi$(ix)_$p.png"), fig)
        end
        println("saved hovmoller_dir_$(group)_*_$p.png (± $cmax cm)")
    end
end
for (p, ptidx) in hov_periods
    plot_hov_period(p, ptidx)
    GC.gc()
end

# ============================================================================
# (5) ssh variance of each term along the three lines
# ============================================================================
# The variance about the period mean, ∝ energy, adds like the flux: var(N + S) = var(N) + var(S) + 2 cov(N, S), so the
# cross term is 2 cov(N, S) (its size is drawn). Half-plane figure: all (N + S), north, south, cross; small-sector
# figure: the small sectors, with the half-plane north and south as thin lines.
line_var(a) = vec(mean((a .- mean(a; dims = 2)) .^ 2; dims = 2)) .* 1e4   # [cm²]

function plot_ssh_line(group, suffix, desc, p, ptidx)
    for (lname, ix) in tr_lines
        L = hov[lname]
        jj = findall(deep[ix, :])
        n, s = L.f["ssh_north$suffix"][:, ptidx], L.f["ssh_south$suffix"][:, ptidx]
        nh, sh = L.f["ssh_north$(half(suffix))"][:, ptidx], L.f["ssh_south$(half(suffix))"][:, ptidx]
        vn, vs = line_var(n), line_var(s)
        terms = is_small(suffix) ?
                [(line_var(nh), "north 0–180°", :firebrick, 1, :solid), (line_var(sh), "south 180–360°", :royalblue, 1, :solid),
                 (vn, "north 60–120°", :firebrick, 2.5, :solid), (vs, "south 240–300°", :royalblue, 2.5, :solid)] :
                [(line_var(n .+ s), "north + south", :black, 2.5, :solid), (vn, "north going", :firebrick, 2, :solid),
                 (vs, "south going", :royalblue, 2, :solid), (abs.(line_var(n .+ s) .- vn .- vs), "cross terms", :gray50, 2, :dash)]
        top = 1.15 * maximum(maximum(t[1][jj]) for t in terms)

        fig = Figure(size = (1500, 800))
        ax1 = Axis(fig[1, 1]; title = "ssh variance about the period mean", ylabel = "⟨ssh′²⟩ [cm²]")
        for (v, label, color, lw, ls) in terms
            lines!(ax1, L.dist, v; color = color, linewidth = lw, linestyle = ls, label = label)
        end
        ylims!(ax1, 0, top)
        Legend(fig[1, 2], ax1; framevisible = false)
        ax2 = Axis(fig[2, 1]; title = "energy ratio south / north", ylabel = "E_S / E_N", yscale = log10,
                   xlabel = "distance along line [km] (grid south → north)")
        lines!(ax2, L.dist, vs ./ vn; color = :black, linewidth = 2)
        hlines!(ax2, [1]; color = :gray50, linestyle = :dot)
        for ax in (ax1, ax2)
            xlims!(ax, L.dist[jj[1]], L.dist[jj[end]])
            isempty(L.moor_dist) || vlines!(ax, L.moor_dist; color = :magenta, linestyle = :dash, linewidth = 1.5)
        end
        isempty(L.moor) || text!(ax1, L.moor_dist, fill(top, length(L.moor)); text = L.moor, rotation = π / 2,
                                 align = (:right, :bottom), fontsize = 9, color = :magenta, font = :bold, offset = (-2, -3))
        linkxaxes!(ax1, ax2)
        Label(fig[0, :], "ssh, $desc, $(sec_str(suffix)), $lname line (xi = $ix)\n$p: $(pstr(ptidx)); deep interior (h ≥ $h_deep m)",
              fontsize = 15, tellwidth = false)
        save(joinpath(line_path, "ssh_line_$(group)_$(lname)_xi$(ix)_$p.png"), fig)
    end
    println("saved transect/ssh_line_$(group)_*_$p.png")
end
for (group, suffix, desc) in groups, (p, ptidx) in periods
    plot_ssh_line(group, suffix, desc, p, ptidx)
end
close(ds)
println("figures in $figure_path")
