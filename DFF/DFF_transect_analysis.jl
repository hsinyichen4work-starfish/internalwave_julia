using NCDatasets, Dates, Statistics, FFTW, CairoMakie

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading

# Envelope (energy) and phase speed of the grid-north and grid-south going ssh along the three Hovmöller lines, for the
# three DFF groups (total, M2, bc). Works only on the transect files written by DFF_ploting.jl (section 5, ~56 MB each),
# so it runs in a few minutes — no need for the 33 GB DDF file or a batch job.
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")   # dff_run, run_dir, ddf_dir, fig_dir (set by DFF_RUN)
tr_dir = ddf_dir
figure_path = joinpath(fig_dir, "transect")
mkpath(figure_path)

# ============================================================================
# (1) read the north / south ssh of every group along each line
# ============================================================================
# [eta, time] in m. The file of each line covering the most hours is used (the full record; DFF_ploting.jl also writes
# one-day files).
#   total  ssh_north / ssh_south         all frequencies
#   M2     ssh_north_M2 / ssh_south_M2   M2 band (9–15 h), wavelengths < 360 km
#   bc     ssh_north_bc / ssh_south_bc   all frequencies, wavelengths < 360 km (approximate baroclinic)
# and the same three for the small sectors (<group>_smallsec: ssh_north_smallsec<_M2/_bc>, north 60–120°, south
# 240–300°), which keep only waves travelling within ±30° of the grid-north / grid-south axis — closer to along the lines
groups = [("total", "", "all frequencies"),
          ("M2", "_M2", "M2 band (9–15 h), λ < 360 km"),
          ("bc", "_bc", "all frequencies, λ < 360 km"),
          ("total_smallsec", "_smallsec", "small sector, all frequencies"),
          ("M2_smallsec", "_smallsec_M2", "small sector, M2 band (9–15 h), λ < 360 km"),
          ("bc_smallsec", "_smallsec_bc", "small sector, all frequencies, λ < 360 km")]
dirs = ("north", "south")
line_names = ["west", "mooring", "east"]
edge = 60        # points at both ends of a line inside the FFT edge-leakage bands: shaded in plots, left out of stats
edge_hours = 24  # hours at both ends of the record: the analytic signal wraps around there, left out of stats
h_deep = 200     # [m] shallower is shelf: the envelope there is mostly barotropic leakage, left out of the line plots/stats

function read_line(lname)
    files = filter(f -> startswith(f, "ssh_ddf_transect_$(lname)_xi") && endswith(f, ".nc"), readdir(tr_dir))
    isempty(files) && error("no transect file for the $lname line in $tr_dir — run DFF_ploting.jl first")
    fname = joinpath(tr_dir, files[argmax([NCDataset(ds -> ds.dim["time"], joinpath(tr_dir, f)) for f in files])])
    NCDataset(fname) do ds
        missing_vars = ["ssh_$(dir)$(suf)" for (_, suf, _) in groups for dir in dirs if !haskey(ds, "ssh_$(dir)$(suf)")]
        isempty(missing_vars) || error("$(basename(fname)) has no $(join(missing_vars, ", ")) — re-run DDF_on_ssh.jl " *
                                       "and DFF_ploting.jl (section 5) to add them")
        # moorings_on_line = "M1 at 302.1 km; M2 at 342.0 km; ..." or "none"
        moor = ds.attrib["moorings_on_line"] == "none" ? Tuple{String, Float64}[] :
               [(m[1], parse(Float64, m[2])) for m in eachmatch(r"(\w+) at ([\d.]+) km", ds.attrib["moorings_on_line"])]
        sort!(moor, by = last)
        dist, h = ds["dist"][:], ds["h"][:]
        j200 = findfirst(>=(h_deep), h)
        ny = length(dist)
        (name = lname, fname = fname, ix = parse(Int, match(r"_xi(\d+)_", basename(fname))[1]),
         dist = dist, h = h, d200 = j200 === nothing ? NaN : dist[j200], date = ds["ocean_time"][:],
         moor = first.(moor), moor_dist = last.(moor),
         # deep interior: outside the edge-leakage bands and deeper than h_deep — the points the line plots and stats use
         deep = [edge < j <= ny - edge && h[j] >= h_deep for j in 1:ny],
         ssh = Dict((g, dir) => Float32.(nomissing(ds["ssh_$(dir)$(suf)"][:, :], NaN)) for (g, suf, _) in groups for dir in dirs))
    end
end
L = Dict(l => read_line(l) for l in line_names)
date = L["mooring"].date
nt = length(date)
dt_s = Dates.value(date[2] - date[1]) / 1000   # [s]
tstat = edge_hours+1:nt-edge_hours             # hours used in the statistics
for l in line_names
    jd = findall(L[l].deep)
    println("$l line: xi = $(L[l].ix), $(length(L[l].dist)) points over $(round(L[l].dist[end], digits = 1)) km, $nt hours, ",
            "deep interior $(round(L[l].dist[jd[1]], digits = 1))–$(round(L[l].dist[jd[end]], digits = 1)) km, ",
            "moorings: ", isempty(L[l].moor) ? "none" : join(L[l].moor, ", "), "   ($(basename(L[l].fname)))")
end

# ============================================================================
# (2) analytic signal: envelope and phase
# ============================================================================
# z = x + i·Hilbert(x), built in the frequency domain along time: keep frequency 0, double the positive frequencies,
# zero the negative ones (the same h as an envelope(x) function per time series, done for every point at once).
#   envelope |z|   amplitude at each point and hour; |z|² ∝ energy, with the same factor for north and south at one
#                  place (it depends on depth and stratification, not direction), so they compare directly
#   phase angle(z) for cos(ωt − k·y) it is ωt − k·y, which gives the phase speed in (4)
# Clean for the band-limited M2 data. For total and bc (all frequencies) the envelope is the amplitude of everything,
# and the phase follows the dominant frequency at each point (tides, but also sub-tidal signals where they are strong).
# The FFT wraps around in time, so the first and last ~day are less reliable.
function analytic(x::AbstractMatrix)   # x [space, time]
    n = size(x, 2)
    hw = zeros(Float32, n); hw[1] = 1
    if iseven(n)
        hw[2:n÷2] .= 2; hw[n÷2+1] = 1
    else
        hw[2:(n+1)÷2] .= 2
    end
    ifft(fft(x, 2) .* permutedims(hw), 2)
end

# ============================================================================
# (3) envelope Hovmöllers: north, south and their energy ratio, full record and each month
# ============================================================================
hov_chunk(d1, d2) = (a = findfirst(>=(DateTime(d1)), date); b = findlast(<(DateTime(d2)), date);
                     a === nothing || b === nothing || b <= a ? nothing : a:b)
periods = filter(x -> x[2] !== nothing,
                 [("full", 1:nt),
                  ("augsep", hov_chunk(Date(2022, 8, 1), Date(2022, 10, 1))),
                  ("oct", hov_chunk(Date(2022, 10, 1), Date(2022, 11, 1))),
                  ("nov", hov_chunk(Date(2022, 11, 1), Date(2022, 12, 1)))])
plabel(tidx) = "$(Dates.format(date[tidx[1]], "yyyymmddHH"))-$(Dates.format(date[tidx[end]], "yyyymmddHH"))"
pstr(tidx) = "$(Dates.format(date[tidx[1]], "yyyy-mm-dd HH:MM")) – $(Dates.format(date[tidx[end]], "yyyy-mm-dd HH:MM"))"

# moorings (magenta dashed), the 200 m shelf break (grey) and the edge-leakage bands (grey shading) on a distance axis.
# horizontal = true: distance is the y axis (Hovmöller), names at x = label_at (the right end of the time axis);
# horizontal = false: distance is the x axis, names written upward from y = label_at (nothing: no names)
function add_marks!(ax, Ll, horizontal; label_at = nothing)
    lines_f!, span_f! = horizontal ? (hlines!, hspan!) : (vlines!, vspan!)
    span_f!(ax, [Ll.dist[1], Ll.dist[end-edge]], [Ll.dist[edge], Ll.dist[end]]; color = (:gray, 0.25))
    isempty(Ll.moor_dist) || lines_f!(ax, Ll.moor_dist; color = :magenta, linestyle = :dash, linewidth = 1.5)
    isnan(Ll.d200) || lines_f!(ax, [Ll.d200]; color = :gray40, linewidth = 1)
    (label_at === nothing || isempty(Ll.moor)) && return
    if horizontal
        text!(ax, fill(label_at, length(Ll.moor)), Ll.moor_dist; text = Ll.moor, align = (:right, :bottom),
              fontsize = 10, color = :magenta, font = :bold, offset = (-3, 1))
    else
        text!(ax, Ll.moor_dist, fill(label_at, length(Ll.moor)); text = Ll.moor, rotation = π / 2,
              align = (:right, :bottom), fontsize = 9, color = :magenta, font = :bold, offset = (-2, -3))
    end
end
deep_idx(Ll) = findall(Ll.deep)
deep_only(Ll, v) = ifelse.(Ll.deep, v, NaN)   # NaN outside the deep interior: lines! leaves a gap there

# save a figure and free it right away: CairoMakie keeps each figure's buffers until collected, and on the login node
# (8 GB address-space cap per process) a few large figures are enough to run out
function save_free(fname, fig)
    save(fname, fig)
    println("saved $(basename(fname))")
    empty!(fig)
    GC.gc()
    Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)
end

function plot_env_period(g, gdesc, env, ptidx)
    span_h = length(ptidx) - 1
    tunit, tdiv = span_h <= 72 ? ("hours", 1) : ("days", 24)
    t_ax = (0:span_h) ./ tdiv
    width = clamp(span_h + 200, 900, 2400)
    # shared colour range [cm] for this group and period: 99th percentile of the envelope over the deep interior of all
    # lines (the shelf, where total is dominated by the barotropic tide, saturates)
    emax = round(quantile(reduce(vcat, [vec(env[(l, d)][deep_idx(L[l]), ptidx]) for l in line_names for d in dirs]),
                          0.99) * 100, RoundUp; sigdigits = 2)
    for l in line_names
        Ll = L[l]
        fig = Figure(size = (width, 1350))
        panels = [("north going: $g envelope", env[(l, "north")][:, ptidx] .* 100, :thermal, (0, emax), "envelope [cm]"),
                  ("south going: $g envelope", env[(l, "south")][:, ptidx] .* 100, :thermal, (0, emax), "envelope [cm]"),
                  ("energy ratio south / north", log10.((env[(l, "south")][:, ptidx] ./ env[(l, "north")][:, ptidx]) .^ 2),
                   :balance, (-1, 1), "log₁₀(E_south / E_north)")]
        for (i, (title, data, cmap, crange, clabel)) in enumerate(panels)
            ax = Axis(fig[i, 1]; title = title, ylabel = "distance along line [km]\n(grid south → north)",
                      xlabel = i == 3 ? "$tunit since $(Dates.format(date[ptidx[1]], "yyyy-mm-dd HH:MM"))" : "")
            hm = heatmap!(ax, t_ax, Ll.dist, permutedims(data); colormap = cmap, colorrange = crange)
            ylims!(ax, Ll.dist[1], Ll.dist[end])
            add_marks!(ax, Ll, true; label_at = t_ax[end])
            Colorbar(fig[i, 2], hm; label = clabel)
        end
        Label(fig[0, :], "$g ssh envelope ($gdesc) along the $l line (xi = $(Ll.ix)), $(pstr(ptidx))", fontsize = 16)
        save_free(joinpath(figure_path, "envelope_$(g)_$(l)_xi$(Ll.ix)_$(plabel(ptidx)).png"), fig)
    end
end

# ============================================================================
# (4) phase speed along each line, from the phase of the analytic signal
# ============================================================================
# For z = A·e^{i(ωt − k·y)}:  ω = ∂φ/∂t,  k = −∂φ/∂y,  c = ω / k  (north going: k > 0, c > 0; south going: c < 0).
# The phase differences are taken as angle(z₂ · conj(z₁)) — no unwrapping needed — and summed over time before taking
# the angle, which weights every hour by its amplitude (strong, clean waves count most):
#   ω̄(y) = angle(Σ_t z(y, t+1)·conj z(y, t)) / Δt
#   k̄(y) = −angle(Σ_t z(y+s, t)·conj z(y−s, t)) / (dist(y+s) − dist(y−s))   over ±s points (s = 5: ~9 km)
# For total and bc, ω̄ is the energy-weighted mean frequency of everything at that point, so c is the speed of the
# dominant signal (usually the semidiurnal tide); where sub-tidal signals dominate, ω̄ and c drop.
# Cross-check on the mooring line: the same k̄ between neighbouring moorings (~40 km apart, less than half a
# wavelength, so the phase difference is unambiguous).
s_pts = 5

function phase_speed(z, dist, tidx)
    ny = size(z, 1)
    ω = [angle(sum(ComplexF64.(z[j, tidx[2:end]] .* conj.(z[j, tidx[1:end-1]])))) / dt_s for j in 1:ny]   # [rad/s]
    k = fill(NaN, ny)
    for j in 1+s_pts:ny-s_pts
        k[j] = -angle(sum(ComplexF64.(z[j+s_pts, tidx] .* conj.(z[j-s_pts, tidx])))) / ((dist[j+s_pts] - dist[j-s_pts]) * 1e3)   # [rad/m]
    end
    c = ω ./ k   # [m/s]
    c[abs.(k) .< 2π / 2000e3] .= NaN   # wavelength > 2000 km: no clear propagation, c meaningless
    (; ω, k, c)
end

# speed between neighbouring moorings (mooring line only): k from the phase difference between the two moorings,
# ω from the mean of their ω̄
function pair_speed(z, Ll, tidx, ω)
    j_of(d) = argmin(abs.(Ll.dist .- d))
    out = NamedTuple[]
    for m in 1:length(Ll.moor)-1
        j1, j2 = j_of(Ll.moor_dist[m]), j_of(Ll.moor_dist[m+1])
        k = -angle(sum(ComplexF64.(z[j2, tidx] .* conj.(z[j1, tidx])))) / ((Ll.dist[j2] - Ll.dist[j1]) * 1e3)
        push!(out, (pair = "$(Ll.moor[m])–$(Ll.moor[m+1])", dmid = (Ll.dist[j1] + Ll.dist[j2]) / 2,
                    c = (ω[j1] + ω[j2]) / 2 / k))
    end
    out
end

# ============================================================================
# (5) summary per line: time-mean energy (envelope²) and phase speed against distance, deep interior only
# ============================================================================
# energy proxy: ⟨envelope²⟩ over the record (cm²), north and south, and their ratio; phase speed |c| (north solid,
# south dashed) for the full record and each month (thin), and the mooring-pair speeds on the mooring line (markers).
# Everything is masked (NaN) in the edge-leakage bands and on the shelf (h < h_deep), and the x axis covers the deep
# interior only.
month_periods = filter(x -> x[1] != "full", periods)
month_colors = Dict("augsep" => :darkorange, "oct" => :seagreen, "nov" => :purple)

function plot_summary(g, gdesc, Z, env, ps)
    for l in line_names
        Ll = L[l]
        jj = deep_idx(Ll)
        xr = (Ll.dist[jj[1]], Ll.dist[jj[end]])
        E_n = deep_only(Ll, vec(mean(env[(l, "north")][:, tstat] .^ 2; dims = 2)) .* 1e4)   # [cm²]
        E_s = deep_only(Ll, vec(mean(env[(l, "south")][:, tstat] .^ 2; dims = 2)) .* 1e4)
        Etop = 1.15 * maximum([E_n[jj]; E_s[jj]])

        fig = Figure(size = (1500, 1150))
        ax1 = Axis(fig[1, 1]; title = "time-mean $g energy proxy ⟨envelope²⟩", ylabel = "⟨envelope²⟩ [cm²]")
        lines!(ax1, Ll.dist, E_n; color = :firebrick, linewidth = 2, label = "north going")
        lines!(ax1, Ll.dist, E_s; color = :royalblue, linewidth = 2, label = "south going")
        ylims!(ax1, 0, Etop)
        axislegend(ax1; position = :rt)

        ax2 = Axis(fig[2, 1]; title = "energy ratio south / north", ylabel = "E_south / E_north", yscale = log10)
        lines!(ax2, Ll.dist, E_s ./ E_n; color = :black, linewidth = 2)
        hlines!(ax2, [1]; color = :gray50, linestyle = :dot)

        ax3 = Axis(fig[3, 1]; title = "$g phase speed |c| (north solid, south dashed; thick = full record, thin = month)",
                   ylabel = "|c| [m/s]", xlabel = "distance along line [km] (grid south → north)")
        for (dir, ls) in (("north", :solid), ("south", :dash))
            for (p, _) in month_periods
                lines!(ax3, Ll.dist, deep_only(Ll, abs.(ps[(l, dir, p)].c)); color = (month_colors[p], 0.8), linewidth = 1,
                       linestyle = ls, label = dir == "north" ? p : nothing)
            end
            lines!(ax3, Ll.dist, deep_only(Ll, abs.(ps[(l, dir, "full")].c)); color = :black, linewidth = 2.5, linestyle = ls,
                   label = dir == "north" ? "full record" : nothing)
            if length(Ll.moor) >= 2
                pr = filter(q -> xr[1] <= q.dmid <= xr[2], pair_speed(Z[(l, dir)], Ll, tstat, ps[(l, dir, "full")].ω))
                isempty(pr) || scatter!(ax3, [q.dmid for q in pr], abs.([q.c for q in pr]); color = :magenta,
                                        strokecolor = :black, strokewidth = 0.75, markersize = 11,
                                        marker = dir == "north" ? :circle : :utriangle,
                                        label = dir == "north" ? "mooring pairs (● north, ▲ south)" : nothing)
            end
        end
        cvals = filter(isfinite, abs.(ps[(l, "north", "full")].c[jj]))
        ylims!(ax3, 0, isempty(cvals) ? 5 : 2 * quantile(cvals, 0.9))
        axislegend(ax3; position = :rt, nbanks = 2)

        for ax in (ax1, ax2, ax3)
            xlims!(ax, xr...)
            add_marks!(ax, Ll, false; label_at = ax === ax1 ? Etop : nothing)
        end
        linkxaxes!(ax1, ax2, ax3)
        Label(fig[0, :], "$g ($gdesc) energy and phase speed along the $l line (xi = $(Ll.ix)), $(pstr(tstat)); " *
                         "deep interior (h ≥ $(h_deep) m)", fontsize = 16)
        save_free(joinpath(figure_path, "energy_phase_speed_$(g)_$(l)_xi$(Ll.ix)_$(plabel(1:nt)).png"), fig)
    end
end

# ============================================================================
# (6) save envelope and phase speed for other analysis
# ============================================================================
function save_group(g, gdesc, Z, env, ps)
    for l in line_names
        Ll = L[l]
        out = joinpath(tr_dir, "ssh_$(g)_envelope_phase_speed_$(l)_xi$(Ll.ix)_$(plabel(1:nt)).nc")
        isfile(out) && rm(out)
        NCDataset(out, "c") do ds
            defDim(ds, "eta_rho", length(Ll.dist))
            defDim(ds, "time", nt)
            defVar(ds, "ocean_time", date, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
            defVar(ds, "dist", Ll.dist, ("eta_rho",); attrib = ["long_name" => "distance along the line from its grid-south end", "units" => "km"])
            defVar(ds, "h", Ll.h, ("eta_rho",); attrib = ["units" => "meter"])
            defVar(ds, "deep", Int8.(Ll.deep), ("eta_rho",);
                   attrib = ["long_name" => "1: outside the $edge-point edge-leakage bands and h >= $h_deep m (used in the line plots)"])
            for dir in dirs
                defVar(ds, "env_$dir", Float32.(env[(l, dir)]), ("eta_rho", "time");
                       attrib = ["long_name" => "envelope |analytic signal| of grid-$dir-going $g ssh ($gdesc)", "units" => "meter"])
                defVar(ds, "phase_$dir", Float32.(angle.(Z[(l, dir)])), ("eta_rho", "time");
                       attrib = ["long_name" => "phase of the analytic signal of grid-$dir-going $g ssh", "units" => "radian"])
                r = ps[(l, dir, "full")]
                defVar(ds, "c_$dir", r.c, ("eta_rho",);
                       attrib = ["long_name" => "phase speed along the line (+ grid north), amplitude-weighted over the record", "units" => "m s-1"])
                defVar(ds, "k_$dir", r.k, ("eta_rho",); attrib = ["long_name" => "wavenumber along the line (+ grid north)", "units" => "rad m-1"])
                defVar(ds, "omega_$dir", r.ω, ("eta_rho",); attrib = ["long_name" => "mean frequency", "units" => "rad s-1"])
            end
            ds.attrib["source"] = Ll.fname
            ds.attrib["group"] = "$g: $gdesc"
            ds.attrib["note"] = "statistics (c, k, omega) leave out the first and last $edge_hours h; the $edge points at each " *
                                "end of the line are inside the FFT edge-leakage bands"
            ds.attrib["moorings_on_line"] = isempty(Ll.moor) ? "none" :
                join(["$m at $(round(d, digits = 1)) km" for (m, d) in zip(Ll.moor, Ll.moor_dist)], "; ")
        end
        println("wrote $out")
    end
end

# ============================================================================
# run (2)–(6) one group at a time, so only one group's analytic signals are in memory
# ============================================================================
function analyse_group(g, gdesc)
    println("\n===== $g: $gdesc =====")
    Z = Dict((l, dir) => analytic(L[l].ssh[(g, dir)]) for l in line_names for dir in dirs)
    env = Dict(k => abs.(z) for (k, z) in Z)   # [m]
    # check: the real part of the analytic signal is the ssh itself
    err = maximum(abs, real.(Z[("mooring", "north")]) .- L["mooring"].ssh[(g, "north")])
    println("analytic-signal check: max |real(z) − ssh| = $(round(err, sigdigits = 2)) m")

    for (p, ptidx) in periods
        plot_env_period(g, gdesc, env, ptidx)
    end

    # per line, direction and period (the record ends trimmed by edge_hours)
    ps = Dict((l, dir, p) => phase_speed(Z[(l, dir)], L[l].dist, intersect(ptidx, tstat))
              for l in line_names for dir in dirs for (p, ptidx) in periods)
    for l in line_names, dir in dirs
        r = ps[(l, dir, "full")]
        jj = intersect(deep_idx(L[l]), findall(isfinite, r.c))
        isempty(jj) && (println("$l $dir: no finite c in the deep interior"); continue)
        println("$l $dir (deep interior): median ω̄ → period $(round(2π / median(r.ω[jj]) / 3600, digits = 2)) h, ",
                "median c = $(round(median(r.c[jj]), digits = 2)) m/s ",
                "(interquartile $(round.(quantile(r.c[jj], [0.25, 0.75]), digits = 2)) m/s), ",
                "median wavelength $(round(2π / median(abs.(r.k[jj])) / 1e3, digits = 1)) km")
    end

    plot_summary(g, gdesc, Z, env, ps)

    # mooring-pair speeds on the mooring line, printed
    for dir in dirs
        Ll = L["mooring"]
        pr = pair_speed(Z[("mooring", dir)], Ll, tstat, ps[("mooring", dir, "full")].ω)
        println("mooring line, $dir going, phase speed between neighbouring moorings [m/s]: ",
                join(["$(q.pair) $(round(q.c, digits = 2))" for q in pr], ", "))
    end

    save_group(g, gdesc, Z, env, ps)
    write_stats(g, env, ps)
end

# ============================================================================
# (7) spatial means of phase speed and energy, per group, line, period and direction, into one text table
# ============================================================================
# two regions on every line:
#   deep        the masked deep interior (outside the edge bands, h ≥ h_deep)
#   M1-CPIES7   the part of the deep interior between M1 and CPIES7 along the mooring line; the west and east lines use
#               the same along-line distances
# c is averaged over the points where it is finite (sign kept: north > 0, south < 0); the median is given too because a
# few near-zero k points can pull the mean. env and E are the time means of envelope and envelope² over the period
# (minus the first/last edge_hours of the record), then averaged in space.
stats_file = joinpath(tr_dir, "ssh_envelope_phase_speed_stats_$(plabel(1:nt)).txt")
moor_range = (Lm = L["mooring"]; (Lm.moor_dist[findfirst(==("M1"), Lm.moor)], Lm.moor_dist[findfirst(==("CPIES7"), Lm.moor)]))
regions = [("deep", Ll -> Ll.deep),
           ("M1-CPIES7", Ll -> Ll.deep .& (moor_range[1] .<= Ll.dist .<= moor_range[2]))]

open(stats_file, "w") do io
    println(io, "# spatial means of the DFF ssh envelope and phase speed along the transect lines")
    println(io, "# source: $tr_dir/ssh_ddf_transect_<line>_xi<ix>_$(plabel(1:nt)).nc")
    println(io, "# groups: " * join(["$g = $d" for (g, _, d) in groups], "; "))
    println(io, "# regions: deep = h >= $h_deep m and outside the $edge-point edge bands; M1-CPIES7 = deep and " *
                "$(round(moor_range[1], digits = 1))-$(round(moor_range[2], digits = 1)) km along the line")
    println(io, "# periods: " * join(["$p = $(pstr(intersect(t, tstat)))" for (p, t) in periods], "; "))
    println(io, "# c_mean, c_median [m/s]: phase speed along the line (+ grid north); env_mean [cm]: mean envelope; " *
                "E_mean [cm^2]: mean envelope^2; E_ratio_s_n: E_mean(south) / E_mean(north) (same on both direction rows)")
    println(io, rpad("group", 16), rpad("line", 9), rpad("period", 8), rpad("region", 11), rpad("dir", 7), rpad("npts", 6),
            lpad("c_mean", 9), lpad("c_median", 10), lpad("env_mean", 10), lpad("E_mean", 10), lpad("E_ratio_s_n", 13))
end

function write_stats(g, env, ps)
    fmt(x, d = 3) = isfinite(x) ? string(round(x, digits = d)) : "NaN"
    open(stats_file, "a") do io
        for l in line_names, (p, ptidx) in periods, (rname, rmask) in regions
            Ll = L[l]
            jr = findall(rmask(Ll))
            tidx = intersect(ptidx, tstat)
            E = Dict(dir => vec(mean(Float64.(env[(l, dir)][jr, tidx]) .^ 2; dims = 2)) .* 1e4 for dir in dirs)   # [cm²]
            ratio = mean(E["south"]) / mean(E["north"])
            for dir in dirs
                c = filter(isfinite, ps[(l, dir, p)].c[jr])
                println(io, rpad(g, 16), rpad(l, 9), rpad(p, 8), rpad(rname, 11), rpad(dir, 7), rpad(length(jr), 6),
                        lpad(fmt(isempty(c) ? NaN : mean(c)), 9), lpad(fmt(isempty(c) ? NaN : median(c)), 10),
                        lpad(fmt(mean(Float64.(env[(l, dir)][jr, tidx])) * 100), 10), lpad(fmt(mean(E[dir])), 10),
                        lpad(fmt(ratio), 13))
            end
        end
    end
    println("appended $g statistics to $stats_file")
end

for (g, _, gdesc) in groups
    analyse_group(g, gdesc)
    GC.gc()
    Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)
end
