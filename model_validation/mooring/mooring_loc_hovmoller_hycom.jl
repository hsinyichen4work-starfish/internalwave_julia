# mooring_loc_hovmoller_hycom.jl
#
# Two sets of figures comparing GLORYS | NCOM | HYCOM, all 28 h low-passed (switch each on/off below):
#  (1) do_hovmoller: Hovmöller diagrams along the M1 -> CPIES7 transect (x = longitude along the line, y = time,
#      time running down), one panel per model. Saved in figure_path.
#  (2) do_station  : time-depth sections (x = time, y = depth 0-400 m) at M1, M3, CPIES7: rows = models,
#      columns = stations. Saved in figure_path/station. See the section at the end.
#
# (1) Hovmöller
#   depths    : surface, 100 m, 200 m
#   variables : temp, salt, u, v, speed, direction (speed / direction from the low-passed u, v)
# Same look as the Hovmöller figures of mooring_loc_model_comp.jl (20261019_mooring_loc_comp), but with the new
# 3-hourly HYCOM GOFS 3.1 transect (extract_mooring_HYCOM.jl) as a third model. The cyan HYCOM-snapshot lines of the
# old figures are gone: HYCOM is now a continuous record.
#
# HYCOM water_temp is IN-SITU temperature: it is converted to potential temperature (GibbsSeaWater) before the
# low-pass, like the other models (the difference is ~0.03 °C at 200 m).
#
# Every transect file stores 3-D vars as (z, x, time), z ASCENDING (bottom -> surface, negative down), on the same
# 330 points (1 km apart). "surface" = each model's top level (GLORYS 0.5 m, NCOM / HYCOM 0 m); 100 / 200 m are
# linear in z between the two levels around that depth (the moving surface is ignored).
# Each point's time series is low-passed over the whole record, then cut to [t0, t1].
#
# Usage: julia mooring_loc_hovmoller_hycom.jl

using NCDatasets, CairoMakie, Dates, Statistics, DSP, Printf, GibbsSeaWater
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
figure_path   = "/home/hsinyi/figure/20261027_meanflow_comp/comp_for_hycom/mooring_loc_comp"
do_hovmoller  = true          # (1) transect Hovmöller figures
do_station    = true          # (2) station time-depth figures (figure_path/station)
transect_name = "M1_CPIES7"
models = [   # (panel title, transect file); left to right
    ("GLORYS", "/home/hsinyi/roms_data/GLOBAL_mooring/GLOBAL_$transect_name.nc"),
    ("NCOM",   "/home/hsinyi/roms_data/NCOM_mooring/NCOM_$transect_name.nc"),       # NCOM 900m (NCOM_mooring_2700 for 2700m)
    ("HYCOM",  "/home/hsinyi/roms_data/HYCOM_mooring/HYCOM_$transect_name.nc"),
]
hov_depths = [0.0, 100.0, 200.0]          # 0 = top level of each model ("surf")
Tcut       = 28.0                          # low-pass cutoff period (hours)
filt_order = 4
t0, t1     = DateTime(2022, 8, 22), DateTime(2022, 11, 30)   # period shown
min_uv_clim = 1.0                          # u, v colour range is at least ±this (m/s) below the surface
speed_max = Dict(0.0 => 1.5, 100.0 => 1.25, 200.0 => 1.25)   # fixed speed colour range 0..this (m/s) per depth
fig_height_cm = 16.0                       # figure height (width is 18 cm, 3 panels side by side)

mkpath(figure_path)
tday(t) = Dates.value.(t .- t0) ./ 86_400_000
date_ticks = DateTime(2022, 9, 1):Month(1):t1
yticks_dates = (tday(date_ticks), Dates.format.(date_ticks, "mm/dd"))

# ---------------------------------------------------------------------------
# Low-pass (as filter_rows in mooring_loc_model_comp.jl): row by row of an (x, time) field; per row keep the span
# between the first and last valid sample, fill gaps inside linearly, Butterworth + filtfilt, put NaN back.
# ---------------------------------------------------------------------------
function filter_rows(time, f, Tcut; N = filt_order)
    dt = median(Dates.value.(diff(time))) / 3_600_000          # hours
    out = fill(NaN, size(f))
    for k in axes(f, 1)
        good = findall(!isnan, f[k, :])
        length(good) < 2 && continue
        y = f[k, good[1]:good[end]]
        for (a, b) in zip(good[1:end-1] .- good[1] .+ 1, good[2:end] .- good[1] .+ 1)
            for i in a+1:b-1
                y[i] = y[a] + (y[b] - y[a]) * (i - a) / (b - a)
            end
        end
        yf = lowhighpass_butter(y, Tcut, dt, N, "low")
        out[k, good] = yf[good .- good[1] .+ 1]
    end
    return out
end

# ---------------------------------------------------------------------------
# Read one depth of a transect file -> (x, time) fields, low-passed and cut to [t0, t1]
# ---------------------------------------------------------------------------
# in-situ -> potential temperature (HYCOM only); z (m, negative down), lon/lat broadcast with temp/salt
insitu_to_pt(temp, salt, z, lon, lat) =
    (p = gsw_p_from_z.(z, lat); gsw_pt0_from_t.(gsw_sa_from_sp.(salt, p, lon, lat), temp, p))

function load_level(fname, d; insitu = false)
    NCDataset(fname) do ds
        z = Float64.(ds["z"][:])                       # ascending, negative down
        # anonymous functions: a named `rd(v) = ...` in both branches would be hoisted (the last one always wins)
        if d == 0
            k = argmax(z)                              # top level
            rd = v -> Float64.(nomissing(ds[v][k, :, :], NaN))
        else
            k = clamp(searchsortedlast(z, -d), 1, length(z) - 1)
            w = (-d - z[k]) / (z[k+1] - z[k])
            (0 <= w <= 1) || error("$d m is outside the levels of $fname")
            rd = v -> (a = Float64.(nomissing(ds[v][k:k+1, :, :], NaN)); (1 - w) .* a[1, :, :] .+ w .* a[2, :, :])
        end
        time = ds["time"][:]
        it = findall(t -> t0 <= t <= t1, time)
        lp(f) = filter_rows(time, f, Tcut)[:, it]
        u, v = lp(rd("u")), lp(rd("v"))
        zu = d == 0 ? -z[k] : d
        lon, lat = Float64.(ds["lon"][:]), Float64.(ds["lat"][:])
        temp, salt = rd("temp"), rd("salt")                     # (x, time)
        insitu && (temp = insitu_to_pt(temp, salt, -zu, lon, lat))
        (lon = lon, dist = ds["dist"][:], mooring = ds["mooring"][:], mooring_dist = ds["mooring_dist"][:],
         z_used = zu, time = time[it],
         temp = lp(temp), salt = lp(salt), u = u, v = v,
         speed = hypot.(u, v), dir = mod.(atand.(u, v), 360))     # direction the current flows TOWARD, 0° = north
    end
end

# ---------------------------------------------------------------------------
# Colour range: at most maxfrac of each panel clipped, rounded outward to a tidy step (as in mooring_loc_model_comp.jl)
# ---------------------------------------------------------------------------
function nice_step(r; n = 6)
    raw = r / n
    p = 10.0^floor(log10(raw))
    return p * first(m for m in (1, 2, 2.5, 5, 10) if m * p >= raw)
end
function clim_5pct(fields; symmetric = false, zero_min = false, maxfrac = 0.05)
    vals = filter(!isempty, [filter(isfinite, vec(f)) for f in fields])
    if symmetric
        m = maximum(quantile(abs.(v), 1 - maxfrac) for v in vals)
        st = nice_step(2m); m = round(ceil(m / st) * st, digits = 10)
        return (-m, m), -m:st:m
    end
    hi = maximum(quantile(v, zero_min ? 1 - maxfrac : 1 - maxfrac / 2) for v in vals)
    lo = zero_min ? 0.0 : minimum(quantile(v, maxfrac / 2) for v in vals)
    st = nice_step(hi - lo)
    lo, hi = round(floor(lo / st) * st, digits = 10), round(ceil(hi / st) * st, digits = 10)
    return (lo, hi), lo:st:hi
end
clipped_frac(f, clim) = (v = filter(isfinite, vec(f)); isempty(v) ? 0.0 : count(x -> x < clim[1] || x > clim[2], v) / length(v))

# ---------------------------------------------------------------------------
# One panel per model side by side; moorings on the line as grey dashed lines, named along the top
# ---------------------------------------------------------------------------
function plot_hovmoller(lev, q; clim, ticks, colormap, cblabel, title, fname)
    names = first.(models)
    fig = Figure(; figkw(18, fig_height_cm)..., figure_padding = (4, 8, 4, 4))
    axs, hm, worst = Axis[], nothing, 0.0
    for (col, m) in enumerate(names)
        h = lev[m]
        f = getproperty(h, q)
        ax = Axis(fig[1, col]; xlabel = "Longitude (°E)", ylabel = col == 1 ? "Date (2022)" : "", yreversed = true,
                  yticks = yticks_dates, xgridvisible = false, ygridvisible = false)
        hm = heatmap!(ax, h.lon, tday(h.time), f; colormap = colormap, colorrange = clim,
                      lowclip = q == :dir ? nothing : :black, highclip = q == :dir ? nothing : :magenta,
                      nan_color = :transparent)
        worst = max(worst, clipped_frac(f, clim))
        mlon = [h.lon[clamp(searchsortedfirst(h.dist, d), 1, length(h.lon))] for d in h.mooring_dist]
        vlines!(ax, mlon; color = (:gray20, 0.6), linestyle = :dash, linewidth = 1)
        axt = Axis(fig[1, col]; xaxisposition = :top, xticks = (mlon, h.mooring), xticklabelsize = 6,
                   xticklabelrotation = pi / 3, xgridvisible = false, ygridvisible = false)
        hidespines!(axt); hideydecorations!(axt)
        linkxaxes!(ax, axt)
        Label(fig[0, col], m, fontsize = 9, font = :bold, tellwidth = false)
        col > 1 && hideydecorations!(ax; ticks = false)
        push!(axs, ax)
    end
    linkaxes!(axs...)
    ylims!(axs[1], tday(t1), tday(t0))
    xlims!(axs[1], extrema(lev[names[1]].lon)...)
    Colorbar(fig[1, length(names) + 1], hm; label = cblabel, ticks = ticks)
    Label(fig[-1, :], title, fontsize = 11, tellwidth = false)
    outname = joinpath(figure_path, fname)
    figsave(outname, fig)
    println("saved ", outname, "  clim = ", clim, ", clipped ≤ ", round(100worst, digits = 1), "% per panel")
    return fig
end

# ---------------------------------------------------------------------------
# All depths x variables, 28 h low-pass
# ---------------------------------------------------------------------------
# (field, long name, colourbar label, colormap, kind) ; kind = :plain / :sym / :zero / :dir
hov_vars = (
    (:temp,  "Temperature",          "Potential temperature (°C)",                :thermal, :plain),
    (:salt,  "Salinity",             "Salinity (PSU)",                            :haline,  :plain),
    (:u,     "Eastward velocity u",  "u (m/s)",                                   :balance, :sym),
    (:v,     "Northward velocity v", "v (m/s)",                                   :balance, :sym),
    (:speed, "Current speed",        "Speed (m/s)",                               :speed,   :zero),
    (:dir,   "Current direction",    "Direction toward (° clockwise from north)", :phase,   :dir),
)
tc = round(Int, Tcut)
tr_title = replace(transect_name, "_" => " – ")

for d in (do_hovmoller ? hov_depths : Float64[])
    lev = Dict(m => load_level(f, d; insitu = m == "HYCOM") for (m, f) in models)
    dname = d == 0 ? "surface" : "$(round(Int, d)) m"
    dtag  = d == 0 ? "surf" : "$(round(Int, d))m"
    println("depth $dname: ", join(["$m $(round(lev[m].z_used, digits = 1)) m, $(length(lev[m].time)) records" for (m, _) in models], "; "))
    for (q, long, cblabel, cmap, kind) in hov_vars
        fields = [getproperty(lev[m], q) for (m, _) in models]
        if kind == :dir
            clim, ticks = (0.0, 360.0), 0:90:360
        elseif q == :speed
            smax = speed_max[d]
            clim, ticks = (0.0, smax), 0:0.25:smax
        else
            clim, ticks = clim_5pct(fields; symmetric = kind == :sym, zero_min = kind == :zero)
            if kind == :sym && d > 0 && clim[2] < min_uv_clim
                clim, ticks = (-min_uv_clim, min_uv_clim), Makie.automatic
            end
        end
        plot_hovmoller(lev, q; clim = clim, ticks = ticks, colormap = cmap, cblabel = cblabel,
            title = "$long at $dname along $tr_title, $tc h low-pass", fname = "hov_$(dtag)_$(q)_lp$(tc)h.png")
    end
end

# ============================================================================================================
# (2) Station time-depth sections: x = time, y = depth (0-400 m), rows = models, columns = stations
# ============================================================================================================
# Point files <prefix>_<station>.nc (same folders as the transect files), variables (z, time) with z ASCENDING
# (negative down). Each model is shown on its OWN levels (GLORYS 30 uneven levels down to 380 m, NCOM every 5 m,
# HYCOM 40 levels), no interpolation in z; each depth row is low-passed over the whole record, then cut to [t0, t1].
# Colour ranges are shared by all 9 panels of a figure (clim_5pct), except speed (fixed, station_speed_max) and
# direction (0-360°, cyclic colormap, no contours).
station_list      = ["M1", "M3", "CPIES7"]
station_zmax      = 400.0                     # deepest depth shown (m)
station_speed_max = 1.5                       # speed colour range 0..this (m/s)
station_path      = joinpath(figure_path, "station")
station_file(f, st) = replace(f, transect_name => st)   # GLOBAL_M1_CPIES7.nc -> GLOBAL_M1.nc

function load_station(fname, lon0, lat0; insitu = false)
    NCDataset(fname) do ds
        z = Float64.(ds["z"][:])
        kz = findall(>=(-station_zmax - 30), z)               # a bit below station_zmax, so the bottom edge is filled
        time = ds["time"][:]
        it = findall(t -> t0 <= t <= t1, time)
        rd(v) = Float64.(nomissing(ds[v][kz, :], NaN))        # (z, time)
        temp, salt = rd("temp"), rd("salt")
        insitu && (temp = insitu_to_pt(temp, salt, z[kz], lon0, lat0))
        lp(f) = filter_rows(time, f, Tcut)[:, it]
        u, v = lp(rd("u")), lp(rd("v"))
        (depth = -z[kz], time = time[it], temp = lp(temp), salt = lp(salt), u = u, v = v,
         speed = hypot.(u, v), dir = mod.(atand.(u, v), 360))
    end
end

function plot_station(S, q; clim, ticks, levels, colormap, cblabel, title, fname)
    names = first.(models)
    fig = Figure(; figkw(18, 14)..., figure_padding = (4, 8, 4, 4))
    axs, hm, worst = Axis[], nothing, 0.0
    for (row, m) in enumerate(names), (col, st) in enumerate(station_list)
        s = S[m][st]
        f = getproperty(s, q)
        ax = Axis(fig[row, col]; yreversed = true, xticks = yticks_dates, yticks = 0:100:400,
                  ylabel = col == 1 ? "$m\nDepth (m)" : "", xlabel = row == length(names) ? "Date (2022)" : "",
                  xgridvisible = false, ygridvisible = false)
        # heatmap wants (x, y, f[x, y]); depth is not evenly spaced for GLORYS / HYCOM, Makie takes the cell centres
        o = sortperm(s.depth)
        hm = heatmap!(ax, tday(s.time), s.depth[o], permutedims(f[o, :]); colormap = colormap, colorrange = clim,
                      lowclip = q == :dir ? nothing : :black, highclip = q == :dir ? nothing : :magenta,
                      nan_color = :transparent)
        levels === nothing || contour!(ax, tday(s.time), s.depth[o], permutedims(f[o, :]); levels = levels,
                                       color = (:black, 0.3), linewidth = 0.5)
        worst = max(worst, clipped_frac(f[s.depth .<= station_zmax, :], clim))
        row == 1 && Label(fig[0, col], st, fontsize = 10, font = :bold, tellwidth = false)
        row < length(names) && hidexdecorations!(ax; ticks = false)
        col > 1 && hideydecorations!(ax; ticks = false)
        push!(axs, ax)
    end
    linkaxes!(axs...)
    xlims!(axs[1], tday(t0), tday(t1))
    ylims!(axs[1], station_zmax, 0)
    Colorbar(fig[1:length(names), length(station_list) + 1], hm; label = cblabel, ticks = ticks)
    Label(fig[-1, :], title, fontsize = 11, tellwidth = false)
    rowgap!(fig.layout, 6); colgap!(fig.layout, 6)
    mkpath(station_path)
    outname = joinpath(station_path, fname)
    figsave(outname, fig)
    println("saved ", outname, "  clim = ", clim, ", clipped ≤ ", round(100worst, digits = 1), "% per panel")
    return fig
end

if do_station
    # station locations (for the in-situ -> potential temperature conversion of HYCOM)
    st_loc = Dict(st => NCDataset(ds -> (Float64(ds["lon"][]), Float64(ds["lat"][])),
                                  station_file(models[end][2], st)) for st in station_list)
    S = Dict(m => Dict(st => load_station(station_file(f, st), st_loc[st]...; insitu = m == "HYCOM")
                       for st in station_list) for (m, f) in models)
    for (m, _) in models, st in station_list
        s = S[m][st]
        println("[$m] $st: $(length(s.depth)) levels $(round(minimum(s.depth), digits = 1))-$(round(maximum(s.depth), digits = 1)) m, $(length(s.time)) records")
    end
    for (q, long, cblabel, cmap, kind) in hov_vars
        fields = [getproperty(S[m][st], q)[S[m][st].depth .<= station_zmax, :] for (m, _) in models for st in station_list]
        if kind == :dir
            clim, ticks, levels = (0.0, 360.0), 0:90:360, nothing
        elseif q == :speed
            clim = (0.0, station_speed_max); ticks = 0:0.25:station_speed_max; levels = ticks
        else
            clim, ticks = clim_5pct(fields; symmetric = kind == :sym)
            levels = ticks
        end
        plot_station(S, q; clim = clim, ticks = ticks, levels = levels, colormap = cmap, cblabel = cblabel,
            title = "$long, $tc h low-pass: GLORYS vs NCOM vs HYCOM", fname = "station_$(q)_lp$(tc)h.png")
    end
end
