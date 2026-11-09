# Compare ERA5 and NCOM (2700 m, nest 1) surface wind stress.
#
# ERA5  : avg_iews / avg_inss  — time-mean eastward / northward turbulent surface stress (N m⁻²),
#         mean over the hour ENDING at valid_time ("...stepType-avg.nc")
#         u10 / v10            — 10 m wind (m s⁻¹), instantaneous at valid_time ("...stepType-instant.nc")
# NCOM  : stresu / stresv      — surface wind stress, no units attribute in the files but the
#         magnitudes (mean |τ| ~0.03, max ~0.2) match N m⁻², same as ERA5; hourly snapshots.
#         Taken here as true east/north components, like NCOM's u_velocity/v_velocity
#         (see check_NCOM/check_NCOM_parallel.jl). Note the 2700 m grid is rotated by ~ -27°
#         (ohgrd_1.nc "ang"); check this before trusting the component-wise comparison.

using NCDatasets, Dates, Statistics

## path setting
ERA5dir     = "/home/hsinyi/data_notm/ERA5_DATA/"            # <yyyymm[_early]>/data_stream-oper_stepType-{avg,instant}.nc
NCOMdir     = "/home/hsinyi/roms_data/NCOM_DATA_NC_2700/"    # yyyymmddHH_wnd.nc
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_1.nc"  # 2700 m NCOM grid (904 × 940)

# days to read (inclusive, through 00:00 of the day after day_end);
# NCOM covers Date(2022, 8, 22) to Date(2022, 11, 30)
day_start = Date(2022, 8, 22)
day_end   = Date(2022, 11, 30)
t_start   = DateTime(day_start)
t_end     = DateTime(day_end) + Day(1)

## ---------------------------------------------------------------- ERA5
# ERA5 is split over several folders (202208_early, 202208, 202209, ...). Read every
# file in the time window, then sort by time and drop any duplicated hours.
function load_era5(dir, stream, varnames, t_start, t_end)
    files = sort(filter(isfile, [joinpath(dir, d, "data_stream-oper_stepType-$stream.nc") for d in readdir(dir)]))
    lon = lat = nothing
    times = DateTime[]
    data = Dict(v => Array{Float32,3}[] for v in varnames)
    for f in files
        NCDataset(f) do ds
            t  = ds["valid_time"][:]
            it = findall(x -> t_start <= x <= t_end, t)
            isempty(it) && return
            lon === nothing && (lon = ds["longitude"][:]; lat = ds["latitude"][:])
            append!(times, t[it])
            for v in varnames   # (lon, lat, time) in Julia order
                push!(data[v], Float32.(coalesce.(ds[v][:, :, it], NaN32)))
            end
        end
    end
    isempty(times) && error("no ERA5 $stream data between $t_start and $t_end")
    # sort + unique, in case the folders overlap
    order = sortperm(times)
    keep  = order[[true; diff(times[order]) .!= Millisecond(0)]]
    out   = Dict(v => cat(data[v]...; dims = 3)[:, :, keep] for v in varnames)
    return lon, lat, times[keep], out
end

era5_lon, era5_lat, era5_time_avg, era5_avg = load_era5(ERA5dir, "avg", ["avg_iews", "avg_inss"], t_start, t_end)
era5_taux = era5_avg["avg_iews"]          # (lon, lat, time), N m⁻², eastward
era5_tauy = era5_avg["avg_inss"]          # (lon, lat, time), N m⁻², northward
# avg_* is the mean over the hour ending at valid_time, so it is centred 30 min earlier
era5_time_avg_mid = era5_time_avg .- Minute(30)

_, _, era5_time_inst, era5_inst = load_era5(ERA5dir, "instant", ["u10", "v10", "sst"], t_start, t_end)
era5_u10 = era5_inst["u10"]               # (lon, lat, time), m s⁻¹
era5_v10 = era5_inst["v10"]

# ERA5 land mask: sst is NaN where the cell is more than 50% land (fixed in time)
era5_ocean = isfinite.(era5_inst["sst"][:, :, 1])

# ERA5 latitude is stored north -> south; flip to south -> north for plotting
if era5_lat[1] > era5_lat[end]
    era5_lat = reverse(era5_lat)
    for A in (era5_taux, era5_tauy, era5_u10, era5_v10)
        reverse!(A; dims = 2)
    end
    era5_ocean = reverse(era5_ocean; dims = 2)
end

# land points -> NaN (ERA5 stress over land is large because of the rough surface)
for A in (era5_taux, era5_tauy, era5_u10, era5_v10)
    A[.!era5_ocean, :] .= NaN32
end
println("ERA5 land mask: $(count(.!era5_ocean)) of $(length(era5_ocean)) points are land")
println("ERA5 avg    : $(length(era5_time_avg)) h, $(first(era5_time_avg)) to $(last(era5_time_avg))")
println("ERA5 instant: $(length(era5_time_inst)) h, $(first(era5_time_inst)) to $(last(era5_time_inst))")

## ---------------------------------------------------------------- NCOM
# MT in the NCOM files is days since 1900-12-31 00:00
mt2datetime(mt) = DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000))

# 2700 m grid, (xi, eta) = (904, 940) in Julia order, same layout as stresu/stresv
lon_ncom, lat_ncom, mask_ncom, ang_ncom = NCDataset(parent_grid) do ds
    ds["lon"][:, :], ds["lat"][:, :], ds["mask"][:, :], ds["ang"][:, :]
end

ncom_tags = sort([replace(f, "_wnd.nc" => "") for f in readdir(NCOMdir) if endswith(f, "_wnd.nc")])
ncom_tags = filter(tag -> day_start <= Date(tag[1:8], "yyyymmdd") <= day_end, ncom_tags)

# Each daily file holds 25 hourly records (00..24 h); record 25 is the same time as
# record 1 of the next day, so it is only kept for the last day.
# Float32: (904, 940, 2425 h) is ~8 GB per variable for the whole period.
function load_ncom_wnd(dir, tags)
    nx, ny = NCDataset(joinpath(dir, tags[1] * "_wnd.nc")) do ds
        size(ds["stresu"])[1:2]
    end
    nt = 24 * length(tags) + 1
    time_out = Vector{DateTime}(undef, nt)
    taux, tauy = (Array{Float32}(undef, nx, ny, nt) for _ in 1:2)
    for (k, tag) in enumerate(tags)
        NCDataset(joinpath(dir, tag * "_wnd.nc")) do ds
            MT   = ds["MT"][:]
            recs = k == length(tags) ? eachindex(MT) : eachindex(MT)[1:end-1]
            length(recs) == (k == length(tags) ? 25 : 24) ||
                error("$tag: expected 25 hourly records, found $(length(MT))")
            r = 24 * (k - 1) .+ (1:length(recs))
            time_out[r]     = mt2datetime.(MT[recs])
            taux[:, :, r]   = Float32.(coalesce.(ds["stresu"][:, :, recs], NaN))
            tauy[:, :, r]   = Float32.(coalesce.(ds["stresv"][:, :, recs], NaN))
        end
        println("done $tag")
    end
    # incomplete records are left at the raw NetCDF fill value (~9.97e36)
    for F in (taux, tauy)
        F[abs.(F) .> 1f30] .= NaN32
    end
    return time_out, taux, tauy
end

ncom_time, ncom_taux, ncom_tauy = load_ncom_wnd(NCOMdir, ncom_tags)   # (xi, eta, time), N m⁻²
# land points
land = coalesce.(mask_ncom, 0f0) .== 0
ncom_taux[land, :] .= NaN32
ncom_tauy[land, :] .= NaN32
println("NCOM        : $(length(ncom_time)) h, $(first(ncom_time)) to $(last(ncom_time))")

## quick sanity check: magnitudes should be of the same order
println("mean |τ| ERA5 = ", mean(filter(isfinite, hypot.(era5_taux, era5_tauy))), " N m⁻²")
println("mean |τ| NCOM = ", mean(filter(isfinite, hypot.(ncom_taux, ncom_tauy))), " N m⁻²")
println("mean |U10| ERA5 = ", mean(filter(isfinite, hypot.(era5_u10, era5_v10))), " m s⁻¹")

## ---------------------------------------------------------------- geometry: outlines, moorings, common box
using CairoMakie, Printf
include("/home/hsinyi/Documents/Julia/function/fig_setup.jl")      # figkw, figsave
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")   # topdown_axis, plot_curvilinear!, grid_boundary
include("/home/hsinyi/Documents/Julia/function/mooring_transect.jl")   # read_moorings
CairoMakie.activate!()

lon_ncom_f, lat_ncom_f = Float64.(lon_ncom), Float64.(lat_ncom)
lon_ncom_b, lat_ncom_b = grid_boundary(lon_ncom_f, lat_ncom_f)                 # NCOM domain outline
era5_box = ([era5_lon[1], era5_lon[end], era5_lon[end], era5_lon[1], era5_lon[1]],
            [era5_lat[1], era5_lat[1], era5_lat[end], era5_lat[end], era5_lat[1]])   # ERA5 box outline

# ROMS 900 m child grid outline (lon in 0..360 -> -180..180)
roms_grid = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"
lon_roms, lat_roms = NCDataset(roms_grid) do ds
    Float64.(ds["lon_rho"][:, :]), Float64.(ds["lat_rho"][:, :])
end
lon_roms[lon_roms .> 180] .-= 360
lon_roms_b, lat_roms_b = grid_boundary(lon_roms, lat_roms)

# moorings: french, M1..M4, CPIES1..9 (locations from the grid file attributes)
moor_name, moor_lon, moor_lat = read_moorings("/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc")
println("moorings: ", join(moor_name, ", "))

# ray-casting point-in-polygon test (polygon given as vertex vectors px, py)
function inpolygon(x, y, px, py)
    inside = false
    j = length(px)
    for i in eachindex(px)
        if (py[i] > y) != (py[j] > y) && x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
            inside = !inside
        end
        j = i
    end
    return inside
end

# Largest lon/lat rectangle covered by both data sets: the largest block of ERA5 grid
# points that all lie inside the (rotated) NCOM domain. Brute force over all index
# boxes, with a 2D prefix sum to count the inside points in O(1) per box.
# Returns the index ranges into era5_lon / era5_lat.
function largest_common_box(era5_lon, era5_lat, bx, by)
    inside = [inpolygon(x, y, bx, by) for x in era5_lon, y in era5_lat]
    nx, ny = size(inside)
    S = zeros(Int, nx + 1, ny + 1)
    for j in 1:ny, i in 1:nx
        S[i+1, j+1] = inside[i, j] + S[i, j+1] + S[i+1, j] - S[i, j]
    end
    best, box = 0.0, (1, 1, 1, 1)
    for i1 in 1:nx, i2 in i1+1:nx, j1 in 1:ny, j2 in j1+1:ny
        n = (i2 - i1 + 1) * (j2 - j1 + 1)
        S[i2+1, j2+1] - S[i1, j2+1] - S[i2+1, j1] + S[i1, j1] == n || continue
        # area on the sphere ∝ Δlon · cos(mean lat) · Δlat
        area = abs(era5_lon[i2] - era5_lon[i1]) * abs(era5_lat[j2] - era5_lat[j1]) *
               cosd((era5_lat[j1] + era5_lat[j2]) / 2)
        area > best && (best = area; box = (i1, i2, j1, j2))
    end
    i1, i2, j1, j2 = box
    return i1:i2, j1:j2
end

box_i, box_j = largest_common_box(era5_lon, era5_lat, lon_ncom_b, lat_ncom_b)
box_lon, box_lat = extrema(era5_lon[box_i]), extrema(era5_lat[box_j])
println("common box: lon $(box_lon), lat $(box_lat)")

## ---------------------------------------------------------------- common 0.25° grid
# The common grid is the ERA5 grid inside the common box, so ERA5 is only subset.
# NCOM is bin-averaged: each 0.25° cell (centre ± 0.125°) gets the mean of the NCOM
# ocean points inside it (~100 points per cell), which removes the scales ERA5 cannot
# resolve (bilinear interpolation would alias them).
# A cell is kept only if it is ERA5 ocean, ≥ ocean_frac_min NCOM ocean points, and
# its four corners lie inside the NCOM domain (fully covered).
ncom_hourmean  = true   # NCOM: average the snapshots at t-1 h and t to match ERA5's hour-mean ending at t
ocean_frac_min = 0.9

com_lon = era5_lon[box_i]
com_lat = era5_lat[box_j]
dll     = 0.25
ncx, ncy = length(com_lon), length(com_lat)

# cell (linear index) of every NCOM point, 0 = outside the common grid
ci = round.(Int, (lon_ncom_f .- com_lon[1]) ./ dll) .+ 1
cj = round.(Int, (lat_ncom_f .- com_lat[1]) ./ dll) .+ 1
ncom_cell = [(1 <= ci[p] <= ncx && 1 <= cj[p] <= ncy) ? LinearIndices((ncx, ncy))[ci[p], cj[p]] : 0
             for p in eachindex(ci)]

ncom_wet  = .!land                                 # NCOM ocean points
n_all     = zeros(Int, ncx, ncy)                   # NCOM points per cell
n_wet     = zeros(Int, ncx, ncy)                   # NCOM ocean points per cell
for p in eachindex(ncom_cell)
    c = ncom_cell[p]
    c == 0 && continue
    n_all[c] += 1
    n_wet[c] += ncom_wet[p]
end
ocean_frac = n_wet ./ max.(n_all, 1)
covered = [all(inpolygon(lo + sx * dll / 2, la + sy * dll / 2, lon_ncom_b, lat_ncom_b)
               for sx in (-1, 1), sy in (-1, 1)) for lo in com_lon, la in com_lat]
com_mask = era5_ocean[box_i, box_j] .& (ocean_frac .>= ocean_frac_min) .& covered
println("common grid: $ncx × $ncy cells, $(count(com_mask)) kept (ocean, fully covered)")

# bin average of one NCOM field (xi, eta) onto the common grid; NaNs are skipped
function bin_average!(out, cnt, F, cell, wet)
    fill!(out, 0f0); fill!(cnt, 0)
    @inbounds for p in eachindex(F)
        c = cell[p]
        (c == 0 || !wet[p]) && continue
        v = F[p]
        isfinite(v) || continue
        out[c] += v
        cnt[c] += 1
    end
    @inbounds for c in eachindex(out)
        out[c] = cnt[c] > 0 ? out[c] / cnt[c] : NaN32
    end
    return out
end

# common time axis: ERA5 valid times where the NCOM record (and the one an hour before) exists
ncom_idx = Dict(t => k for (k, t) in enumerate(ncom_time))
com_time = filter(t -> haskey(ncom_idx, t) && (!ncom_hourmean || haskey(ncom_idx, t - Hour(1))), era5_time_avg)
it_era5_com = [findfirst(==(t), era5_time_avg) for t in com_time]

function ncom_to_common(F, times)
    out = Array{Float32}(undef, ncx, ncy, length(times))
    tmp, cnt = zeros(Float32, ncx, ncy), zeros(Int, ncx, ncy)
    for (n, t) in enumerate(times)
        k = ncom_idx[t]
        out[:, :, n] = bin_average!(tmp, cnt, view(F, :, :, k), ncom_cell, ncom_wet)
        if ncom_hourmean
            out[:, :, n] .= (out[:, :, n] .+ bin_average!(tmp, cnt, view(F, :, :, k - 1), ncom_cell, ncom_wet)) ./ 2
        end
    end
    out[.!com_mask, :] .= NaN32
    return out
end

com_ncom_taux = ncom_to_common(ncom_taux, com_time)            # (lon, lat, time), N m⁻²
com_ncom_tauy = ncom_to_common(ncom_tauy, com_time)
com_era5_taux = era5_taux[box_i, box_j, it_era5_com]
com_era5_tauy = era5_tauy[box_i, box_j, it_era5_com]
for A in (com_era5_taux, com_era5_tauy)
    A[.!com_mask, :] .= NaN32
end
println("common time: $(length(com_time)) h, $(first(com_time)) to $(last(com_time))")

## ---------------------------------------------------------------- snapshot plots
# ERA5 avg_* at valid_time = t_snap is the mean over the hour ending at t_snap,
# NCOM (native) is the instantaneous field at t_snap.
figure_path = "/home/hsinyi/figure/20261023_NCOM2700/wind_comp"
mkpath(figure_path)

t_snap = DateTime(2022, 9, 10, 0)
it_ncom = findfirst(==(t_snap), ncom_time)
it_era5 = findfirst(==(t_snap), era5_time_avg)
it_com  = findfirst(==(t_snap), com_time)
(it_ncom === nothing || it_era5 === nothing || it_com === nothing) &&
    error("$t_snap not in the loaded NCOM/ERA5 time range")

ncom_snap = (ncom_taux[:, :, it_ncom], ncom_tauy[:, :, it_ncom])
ncom_snap = (ncom_snap..., hypot.(ncom_snap...))
era5_snap = (era5_taux[:, :, it_era5], era5_tauy[:, :, it_era5])
era5_snap = (era5_snap..., hypot.(era5_snap...))

comp_titles = ("τx (east)", "τy (north)", "|τ|")
cmaps       = (Reverse(:RdBu), Reverse(:RdBu), :viridis)   # red = positive, blue = negative

# colour limit from the 99th percentile of |values|, rounded up to `step`
round_lim(x, step) = max(ceil(quantile(filter(isfinite, abs.(x)), 0.99) / step) * step, step)
lon_ticks(lon_lims) = (d = lon_lims[2] - lon_lims[1] > 20 ? 10 : 5;   # spacing so the labels do not overlap
                       (ceil(lon_lims[1] / d) * d):d:lon_lims[2])

# ROMS 900 m outline (optional) and moorings: French mooring as a star, M1..M4 / CPIES as circles
function add_overlays!(ax; roms = false)
    roms && lines!(ax, lon_roms_b, lat_roms_b; color = :magenta, linewidth = 1.5)
    fr = moor_name .== "french"
    scatter!(ax, moor_lon[.!fr], moor_lat[.!fr]; marker = :circle, markersize = 5,
             color = :white, strokecolor = :black, strokewidth = 0.8)
    scatter!(ax, moor_lon[fr], moor_lat[fr]; marker = :star5, markersize = 11,
             color = :white, strokecolor = :black, strokewidth = 0.8)
end

# One 2 × 3 snapshot figure of the native fields zoomed to lon_lims × lat_lims;
# colour limits from |τ| of both data sets inside that window.
# outlines: the other data set's coverage; roms: the ROMS 900 m grid outline.
function snapshot_figure(outname, lon_lims, lat_lims; size_cm = (24, 15), outlines = true, roms = false)
    in_win(lo, la) = lon_lims[1] <= lo <= lon_lims[2] && lat_lims[1] <= la <= lat_lims[2]
    ncom_win = in_win.(lon_ncom_f, lat_ncom_f)
    era5_win = [in_win(lo, la) for lo in era5_lon, la in era5_lat]
    tau_lim = round_lim(vcat(ncom_snap[3][ncom_win], era5_snap[3][era5_win]), 0.05)
    crange  = ((-tau_lim, tau_lim), (-tau_lim, tau_lim), (0, tau_lim))

    fig = Figure(; figkw(size_cm...)...)
    Label(fig[0, 1:3], "Surface wind stress, $(Dates.format(t_snap, "yyyy-mm-dd HH:MM")) UTC\n" *
          "(NCOM instantaneous; ERA5 mean of the preceding hour)"; fontsize = 13)
    xt = lon_ticks(lon_lims)
    hms = Vector{Any}(undef, 3)
    for c in 1:3
        ax1 = topdown_axis(fig[1, c]; title = "NCOM 2700 m  " * comp_titles[c], xticks = xt,
                           xlabel = "", ylabel = c == 1 ? "Latitude" : "")
        hms[c] = plot_curvilinear!(ax1, lon_ncom_f, lat_ncom_f, Float64.(ncom_snap[c]);
                                   colormap = cmaps[c], colorrange = crange[c], nan_color = :gray80)
        ax2 = topdown_axis(fig[2, c]; title = "ERA5  " * comp_titles[c], xticks = xt,
                           ylabel = c == 1 ? "Latitude" : "")
        heatmap!(ax2, era5_lon, era5_lat, era5_snap[c];
                 colormap = cmaps[c], colorrange = crange[c], nan_color = :gray80)
        if outlines
            lines!(ax1, era5_box...; color = :black, linewidth = 1.5, linestyle = :dash)
            lines!(ax2, lon_ncom_b, lat_ncom_b; color = :black, linewidth = 1.5)
        end
        for ax in (ax1, ax2)
            add_overlays!(ax; roms = roms)
            xlims!(ax, lon_lims...)
            ylims!(ax, lat_lims...)
        end
    end
    Colorbar(fig[3, 1:2], hms[1]; vertical = false, label = "τx, τy (N m⁻²)", flipaxis = false, width = Relative(0.6))
    Colorbar(fig[3, 3],   hms[3]; vertical = false, label = "|τ| (N m⁻²)",    flipaxis = false)
    figsave(outname, fig)
    println("saved plot to ", outname)
end

# 3 × 3 figure on the common 0.25° grid: NCOM (bin-averaged) / ERA5 / NCOM − ERA5;
# columns τx, τy, |τ|. Bias and RMSE over the kept cells in the difference titles.
# tau_lim / dif_lim: fixed colour limits (for a movie); default: from this snapshot.
function diff_figure(outname, n; tau_lim = nothing, dif_lim = nothing, dpi = 300, verbose = true)
    ncom = (com_ncom_taux[:, :, n], com_ncom_tauy[:, :, n])
    ncom = (ncom..., hypot.(ncom...))
    era5 = (com_era5_taux[:, :, n], com_era5_tauy[:, :, n])
    era5 = (era5..., hypot.(era5...))
    dif  = ncom .- era5

    tau_lim === nothing && (tau_lim = round_lim(vcat(vec(ncom[3]), vec(era5[3])), 0.05))
    dif_lim === nothing && (dif_lim = round_lim(vcat(vec.(dif)...), 0.01))
    crange  = ((-tau_lim, tau_lim), (-tau_lim, tau_lim), (0, tau_lim))
    xt = lon_ticks(box_lon)

    # layout rows: 1 NCOM, 2 ERA5, 3 their colorbars, 4 NCOM − ERA5, 5 its colorbar
    fig = Figure(; figkw(20, 26)..., figure_padding = (4, 16, 4, 4))
    Label(fig[0, 1:3], "Surface wind stress on the common 0.25° grid\n" *
          "$(Dates.format(com_time[n], "yyyy-mm-dd HH:MM")) UTC\n" *
          "(NCOM bin-averaged" * (ncom_hourmean ? " + hour-averaged" : "") *
          "; ERA5 mean of the preceding hour)"; fontsize = 12)
    hms = Matrix{Any}(undef, 3, 3)
    for c in 1:3
        r = dif[c][com_mask]
        bias, rmse = mean(r), sqrt(mean(r .^ 2))
        rows = ((1, ncom[c], "NCOM " * comp_titles[c], "", cmaps[c], crange[c]),
                (2, era5[c], "ERA5 " * comp_titles[c], "", cmaps[c], crange[c]),
                (4, dif[c],  "NCOM − ERA5 " * comp_titles[c], @sprintf("bias %+.3f, RMSE %.3f", bias, rmse),
                 Reverse(:RdBu), (-dif_lim, dif_lim)))
        for (k, (rr, F, ttl, sub, cm, cr)) in enumerate(rows)
            # tick labels only on the outer panels (all panels share the same lon/lat)
            ax = topdown_axis(fig[rr, c]; title = ttl, subtitle = sub, xticks = xt,
                              xlabel = rr == 4 ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "",
                              xticklabelsvisible = rr != 1, yticklabelsvisible = c == 1)
            hms[k, c] = heatmap!(ax, com_lon, com_lat, F; colormap = cm, colorrange = cr, nan_color = :gray80)
            add_overlays!(ax; roms = true)
            xlims!(ax, box_lon[1] - dll / 2, box_lon[2] + dll / 2)
            ylims!(ax, box_lat[1] - dll / 2, box_lat[2] + dll / 2)
        end
    end
    Colorbar(fig[3, 1:2], hms[1, 1]; vertical = false, flipaxis = false, width = Relative(0.7), label = "τx, τy (N m⁻²)")
    Colorbar(fig[3, 3],   hms[1, 3]; vertical = false, flipaxis = false, label = "|τ| (N m⁻²)")
    Colorbar(fig[5, 1:3], hms[3, 1]; vertical = false, flipaxis = false, width = Relative(0.5),
             label = "NCOM − ERA5 (N m⁻²)")
    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 6)
    resize_to_layout!(fig)   # shrink the figure to the fixed-aspect panels: no empty margins
    figsave(outname, fig; dpi = dpi)
    verbose && println("saved plot to ", outname)
end

tag = Dates.format(t_snap, "yyyymmddHH")
# full domains, with each data set's coverage outlined on the other
snapshot_figure(joinpath(figure_path, "wind_stress_NCOM_ERA5_$tag.png"),
                extrema(lon_ncom_f), extrema(lat_ncom_f))
# only the largest rectangle covered by both
snapshot_figure(joinpath(figure_path, "wind_stress_NCOM_ERA5_commonbox_$tag.png"),
                box_lon, box_lat; size_cm = (20, 16), outlines = false, roms = true)
# both on the common 0.25° grid, with their difference
diff_figure(joinpath(figure_path, "wind_stress_NCOM_ERA5_diff_$tag.png"), it_com)

## ---------------------------------------------------------------- domain-mean time series
# Mean over the kept cells of the common grid (same cells for both products), hourly
# (thin) and daily mean (thick), to see how fast the wind stress changes.
domain_mean(A) = [mean(view(A, :, :, n)[com_mask]) for n in axes(A, 3)]
ts_ncom_x, ts_ncom_y = domain_mean(com_ncom_taux), domain_mean(com_ncom_tauy)
ts_era5_x, ts_era5_y = domain_mean(com_era5_taux), domain_mean(com_era5_tauy)

# daily means (each ERA5 hour-mean ending at t belongs to the day of t - 30 min)
com_day  = Date.(com_time .- Minute(30))
days     = unique(com_day)
daymean(ts) = [mean(ts[com_day .== d]) for d in days]
day_mid  = DateTime.(days) .+ Hour(12)

# how fast the field changes: spatial RMS of F(t + L) − F(t), averaged over t,
# relative to the spatial-temporal std of F (≈ 1.41 for an uncorrelated field)
function change_vs_lag(A, lags)
    s = std(filter(isfinite, A))
    [mean(sqrt(mean((view(A, :, :, n + L)[com_mask] .- view(A, :, :, n)[com_mask]) .^ 2))
          for n in 1:size(A, 3) - L) / s for L in lags]
end
lags = [1, 3, 6, 12, 24]
println("relative field change after lag (h) = ", lags)
for (name, A) in (("NCOM τx", com_ncom_taux), ("NCOM τy", com_ncom_tauy),
                  ("ERA5 τx", com_era5_taux), ("ERA5 τy", com_era5_tauy))
    println(rpad(name, 9), join([@sprintf("%6.2f", v) for v in change_vs_lag(A, lags)]))
end

# x axis in days since the first record, labelled by date
t0   = DateTime(first(com_day))
tday(t) = Dates.value(t - t0) / 86_400_000
xt_d = DateTime(first(com_day)):Day(7):DateTime(last(com_day)) + Day(1)
xticks_dates = (tday.(xt_d), Dates.format.(xt_d, "mm-dd"))

col_ncom, col_era5 = RGBf(0.85, 0.37, 0.0), RGBf(0.0, 0.45, 0.70)   # orange / blue (colour-blind safe)
fig = Figure(; figkw(18, 11)..., figure_padding = (4, 12, 4, 4))
Label(fig[-1, 1], "Domain-mean surface wind stress on the common 0.25° grid\n" *
      "($(count(com_mask)) cells; thin: hourly, thick: daily mean)"; fontsize = 11, tellwidth = false)
axs = []
for (r, (lab, n_ts, e_ts)) in enumerate((("τx (east) (N m⁻²)", ts_ncom_x, ts_era5_x),
                                          ("τy (north) (N m⁻²)", ts_ncom_y, ts_era5_y)))
    ax = Axis(fig[r, 1]; ylabel = lab, xticks = xticks_dates, xticklabelsvisible = r == 2,
              xlabel = r == 2 ? "Date (2022)" : "")
    hlines!(ax, 0; color = :gray60, linewidth = 0.8)
    lines!(ax, tday.(com_time), e_ts; color = (col_era5, 0.35), linewidth = 0.7)
    lines!(ax, tday.(com_time), n_ts; color = (col_ncom, 0.35), linewidth = 0.7)
    lines!(ax, tday.(day_mid), daymean(e_ts); color = col_era5, linewidth = 2, label = "ERA5")
    lines!(ax, tday.(day_mid), daymean(n_ts); color = col_ncom, linewidth = 2, label = "NCOM 2700 m")
    xlims!(ax, 0, tday(DateTime(last(com_day)) + Day(1)))
    push!(axs, ax)
end
linkxaxes!(axs...)
Legend(fig[0, 1], axs[1]; orientation = :horizontal, framevisible = false, tellwidth = false, padding = 0)
rowgap!(fig.layout, 6)
outname = joinpath(figure_path, "wind_stress_NCOM_ERA5_domainmean_timeseries.png")
figsave(outname, fig)
println("saved plot to ", outname)

## ---------------------------------------------------------------- quiver time series
# Port of quiver_time_plot.m (phd_island_wake/matlab_tool): arrows from (t, 0) with both
# components multiplied by the same `scale` (time units per N m⁻²) and an equal data
# aspect (MATLAB daspect([1 1 1]) -> DataAspect()), so the arrow angles are true.
# Default scale: one time step per 1.2 × |mean vector|. The y axis has no physical units,
# so its ticks are hidden; the reference arrow gives the magnitude.
function time_step_scale(t, u, v)
    dt = mean(diff(t))
    return dt / (abs(mean(complex.(u, v))) * 1.2)
end

function quiver_time_plot!(ax, t, u, v; scale, color = :black, label = nothing,
                           shaftwidth = 1.2, tipwidth = 5, tiplength = 5)
    arrows2d!(ax, Point2f.(t, 0), Vec2f.(scale .* u, scale .* v); color = color, label = label,
              shaftwidth = shaftwidth, tipwidth = tipwidth, tiplength = tiplength)
end

# One row per month (x = day of month) so that the panel is not ~70× wider than tall:
# the angles stay true (same scale for u and v, DataAspect) in every row.
q_ncom = (daymean(ts_ncom_x), daymean(ts_ncom_y))
q_era5 = (daymean(ts_era5_x), daymean(ts_era5_y))
q_fac   = 2                                                  # arrows 2× the MATLAB default length
q_scale = q_fac * time_step_scale(1:length(days), q_era5...) # days per N m⁻², same for both products
ref_tau = 0.05                                               # reference arrow (N m⁻², eastward)
months  = unique(firstdayofmonth.(days))
xd      = Dates.day.(days)                                   # day of month
ymax    = 1.15 * maximum(abs, q_scale .* vcat(q_era5[2], q_ncom[2]))
q_xlim  = (-2.5, 32.5)                                       # room for westward arrows on day 1

fig = Figure(; figkw(18, 12)..., figure_padding = (4, 12, 4, 4))
Label(fig[-1, 1], "Domain-mean surface wind stress vectors, daily means " *
      "(common 0.25° grid, $(count(com_mask)) cells)"; fontsize = 11, tellwidth = false)
Legend(fig[0, 1], [LineElement(color = col_era5, linewidth = 2), LineElement(color = col_ncom, linewidth = 1)],
       ["ERA5", "NCOM 2700 m"]; orientation = :horizontal, framevisible = false, tellwidth = false, padding = 0)
for (r, m) in enumerate(months)
    sel = firstdayofmonth.(days) .== m
    last_row = r == length(months)
    ax = Axis(fig[r, 1]; width = 15 * cm_to_pt, aspect = DataAspect(),
              ylabel = Dates.format(m, "u yyyy"), ylabelrotation = 0,
              xticks = [1; 5:5:30], xticklabelsvisible = last_row, xlabel = last_row ? "Day of month" : "",
              yticksvisible = false, yticklabelsvisible = false, ygridvisible = false)
    hlines!(ax, 0; color = :black, linestyle = :dash, linewidth = 0.8)
    quiver_time_plot!(ax, xd[sel], q_era5[1][sel], q_era5[2][sel]; scale = q_scale, color = col_era5, shaftwidth = 2)
    quiver_time_plot!(ax, xd[sel], q_ncom[1][sel], q_ncom[2][sel]; scale = q_scale, color = col_ncom, shaftwidth = 1)
    if r == 1   # reference arrow in the first row, left of the first data day
        x_ref = max(q_xlim[1] + 8, minimum(xd[sel]) - q_scale * ref_tau - 4)
        arrows2d!(ax, [Point2f(x_ref, 0.5ymax)], [Vec2f(q_scale * ref_tau, 0)]; color = :black,
                  shaftwidth = 1.5, tipwidth = 5, tiplength = 5)
        text!(ax, x_ref - 0.5, 0.5ymax; text = "$(ref_tau) N m⁻²", align = (:right, :center), fontsize = 9)
    end
    xlims!(ax, q_xlim...)
    ylims!(ax, -ymax, ymax)
end
rowgap!(fig.layout, 4)
resize_to_layout!(fig)
outname = joinpath(figure_path, "wind_stress_NCOM_ERA5_quiver.png")
figsave(outname, fig)
println("saved plot to ", outname)

## ---------------------------------------------------------------- scatter NCOM vs ERA5
# Points: hourly domain means (the time series above), coloured by date.
# r² = squared Pearson correlation; also given for all cells × all hours.
function fit_stats(x, y)
    ok = isfinite.(x) .& isfinite.(y)
    x, y = Float64.(x[ok]), Float64.(y[ok])
    r = cor(x, y)
    slope = cov(x, y) / var(x)
    (; r2 = r^2, slope, icpt = mean(y) - slope * mean(x), bias = mean(y .- x),
       rmse = sqrt(mean((y .- x) .^ 2)), n = length(x))
end

fig = Figure(; figkw(18, 9.5)..., figure_padding = (4, 12, 4, 4))
Label(fig[0, 1:2], "NCOM 2700 m vs ERA5 surface wind stress, hourly domain means " *
      "(common 0.25° grid)"; fontsize = 11, tellwidth = false)
tcol = tday.(com_time)
sc = nothing
for (c, (name, xs, ys, A_e, A_n)) in enumerate((("τx (east)", ts_era5_x, ts_ncom_x, com_era5_taux, com_ncom_taux),
                                               ("τy (north)", ts_era5_y, ts_ncom_y, com_era5_tauy, com_ncom_tauy)))
    s  = fit_stats(xs, ys)
    sa = fit_stats(vec(A_e), vec(A_n))            # every cell and hour
    println(@sprintf("%-10s domain mean: r² = %.4f, slope = %.2f, bias = %+.4f, RMSE = %.4f (n = %d)",
                     name, s.r2, s.slope, s.bias, s.rmse, s.n))
    println(@sprintf("%-10s all cells  : r² = %.4f, slope = %.2f, bias = %+.4f, RMSE = %.4f (n = %d)",
                     name, sa.r2, sa.slope, sa.bias, sa.rmse, sa.n))
    lo, hi = extrema(vcat(xs, ys))
    pad = 0.05 * (hi - lo); lo -= pad; hi += pad
    ax = Axis(fig[1, c]; aspect = AxisAspect(1), title = name,
              xlabel = "ERA5 (N m⁻²)", ylabel = c == 1 ? "NCOM (N m⁻²)" : "")
    lines!(ax, [lo, hi], [lo, hi]; color = :gray50, linestyle = :dash, linewidth = 1)          # 1:1
    global sc = scatter!(ax, xs, ys; color = tcol, colormap = :viridis, markersize = 3)
    lines!(ax, [lo, hi], s.icpt .+ s.slope .* [lo, hi]; color = :black, linewidth = 1.2)       # fit
    text!(ax, 0.04, 0.96; space = :relative, align = (:left, :top), fontsize = 9,
          text = @sprintf("r² = %.3f\nslope = %.2f\nbias = %+.4f\nRMSE = %.4f\nall cells r² = %.3f",
                          s.r2, s.slope, s.bias, s.rmse, sa.r2))
    xlims!(ax, lo, hi); ylims!(ax, lo, hi)
end
Colorbar(fig[1, 3], sc; label = "Date (2022)",
         ticks = (tday.(DateTime.(Date(2022, 9, 1):Month(1):Date(2022, 12, 1))),
                  Dates.format.(Date(2022, 9, 1):Month(1):Date(2022, 12, 1), "mm-dd")))
colgap!(fig.layout, 10)
outname = joinpath(figure_path, "wind_stress_NCOM_ERA5_scatter.png")
figsave(outname, fig)
println("saved plot to ", outname)

## ---------------------------------------------------------------- ERA5 drag coefficient
# Bulk formula τ = ρ_air C_d |U10| U10, solved for C_d with ERA5's own stress and 10 m wind,
# on the ocean cells of the common box (com_mask) over the common period.
# avg_* is the mean over the hour ending at t, so the instantaneous wind term |U|U at
# t-1 h and t is averaged to match it.
# ERA5's stress depends on stability and the wave model (Charnock), so this is an
# effective C_d, not the neutral one.
rho_air  = 1.165   # kg m⁻³ (1.16–1.17 over the tropical ocean)
inst_idx = Dict(t => k for (k, t) in enumerate(era5_time_inst))
cd_time  = filter(t -> haskey(inst_idx, t) && haskey(inst_idx, t - Hour(1)), com_time)
ncd, ntd = count(com_mask), length(cd_time)
cd_Sx, cd_Sy, cd_U2, cd_U = (zeros(ncd, ntd) for _ in 1:4)   # |U|U (x, y), |U|², |U| (hour means)
cd_Tx, cd_Ty = zeros(ncd, ntd), zeros(ncd, ntd)               # ERA5 stress
for (n, t) in enumerate(cd_time)
    for k in (inst_idx[t - Hour(1)], inst_idx[t])
        u = Float64.(view(era5_u10, box_i, box_j, k)[com_mask])
        v = Float64.(view(era5_v10, box_i, box_j, k)[com_mask])
        s = hypot.(u, v)
        cd_Sx[:, n] .+= s .* u ./ 2;  cd_Sy[:, n] .+= s .* v ./ 2
        cd_U2[:, n] .+= s .^ 2 ./ 2;  cd_U[:, n]  .+= s ./ 2
    end
    ka = findfirst(==(t), era5_time_avg)
    cd_Tx[:, n] = view(era5_taux, box_i, box_j, ka)[com_mask]
    cd_Ty[:, n] = view(era5_tauy, box_i, box_j, ka)[com_mask]
end
cd_ok   = isfinite.(cd_Tx) .& isfinite.(cd_Ty) .& isfinite.(cd_Sx)
cd_tau  = hypot.(cd_Tx, cd_Ty)
# (1) vector least squares through the origin: minimises Σ|τ - ρ C_d |U|U|²
cd_fit  = sum((cd_Tx .* cd_Sx .+ cd_Ty .* cd_Sy)[cd_ok]) / (rho_air * sum((cd_Sx .^ 2 .+ cd_Sy .^ 2)[cd_ok]))
# (2) ratio of means: mean|τ| / (ρ mean|U|²)
cd_rom  = mean(cd_tau[cd_ok]) / (rho_air * mean(cd_U2[cd_ok]))
# (3) point by point C_d = |τ| / (ρ|U|²), only where |U10| > 3 m s⁻¹ (C_d is ill-defined in weak wind)
cd_pt   = cd_tau ./ (rho_air .* cd_U2)
cd_sel  = cd_ok .& (cd_U .> 3)
cd_q    = quantile(cd_pt[cd_sel], (0.25, 0.5, 0.75))
# alignment of stress with the wind
cd_cos  = mean(((cd_Tx .* cd_Sx .+ cd_Ty .* cd_Sy) ./ (cd_tau .* hypot.(cd_Sx, cd_Sy)))[cd_sel])
println(@sprintf("ERA5 C_d (ρ_air = %.3f kg m⁻³, %d cells × %d h):", rho_air, ncd, ntd))
println(@sprintf("  vector least-squares fit     : %.3f × 10⁻³", 1e3 * cd_fit))
println(@sprintf("  mean|τ| / (ρ mean|U|²)       : %.3f × 10⁻³", 1e3 * cd_rom))
println(@sprintf("  point C_d (|U| > 3), median  : %.3f × 10⁻³ (IQR %.3f–%.3f), mean %.3f × 10⁻³",
                 1e3 * cd_q[2], 1e3 * cd_q[1], 1e3 * cd_q[3], 1e3 * mean(cd_pt[cd_sel])))
println(@sprintf("  mean |U10| = %.2f m s⁻¹, mean cos(τ, U) = %.4f", mean(cd_U[cd_ok]), cd_cos))
# ρ_air sensitivity: C_d ∝ 1/ρ_air
println(@sprintf("  fit with ρ_air = 1.16 / 1.17 : %.3f / %.3f × 10⁻³",
                 1e3 * cd_fit * rho_air / 1.16, 1e3 * cd_fit * rho_air / 1.17))

# C_d in 1 m s⁻¹ wind-speed bins, with Large & Yeager (2004) neutral C_d for reference
cd_edges = 0:1:14
cd_bins  = [(; u = (a + b) / 2, q = (sel = cd_ok .& (a .<= cd_U .< b); count(sel) >= 100 ?
             quantile(cd_pt[sel], (0.25, 0.5, 0.75)) : (NaN, NaN, NaN)), n = count(cd_ok .& (a .<= cd_U .< b)))
            for (a, b) in zip(cd_edges[1:end-1], cd_edges[2:end])]
for b in cd_bins
    b.n > 0 && println(@sprintf("  |U10| %4.1f m s⁻¹: median C_d %.3f × 10⁻³ (n = %d)", b.u, 1e3 * b.q[2], b.n))
end
cd_LY04(u) = (2.7 / u + 0.142 + u / 13.09) * 1e-3

fig = Figure(; figkw(18, 8.5)..., figure_padding = (4, 12, 4, 4))
Label(fig[0, 1:3], @sprintf("ERA5 drag coefficient from τ = ρ_air C_d |U10| U10 (ρ_air = %.3f kg m⁻³), common box, hourly",
                            rho_air); fontsize = 10, tellwidth = false)
# (a) 2D histogram of |τ| vs ρ|U|²
xa, ya = rho_air .* cd_U2[cd_ok], cd_tau[cd_ok]
xmax = quantile(xa, 0.999); ymax = quantile(ya, 0.999)
nb = 120
H  = zeros(nb, nb)
for (x, y) in zip(xa, ya)
    i = floor(Int, x / xmax * nb) + 1; j = floor(Int, y / ymax * nb) + 1
    (i <= nb && j <= nb) && (H[i, j] += 1)
end
H[H .== 0] .= NaN
ax = Axis(fig[1, 1]; xlabel = "ρ_air |U10|² (kg m⁻¹ s⁻²)", ylabel = "ERA5 |τ| (N m⁻²)", title = "(a) |τ| vs ρ|U10|²")
hm = heatmap!(ax, range(0, xmax; length = nb + 1), range(0, ymax; length = nb + 1), log10.(H); colormap = :viridis)
lines!(ax, [0, xmax], 2e-3 .* [0, xmax]; color = :black, linestyle = :dash, linewidth = 1, label = "C_d = 2×10⁻³")
lines!(ax, [0, xmax], 1e-3 .* [0, xmax]; color = :black, linestyle = :dot, linewidth = 1.2, label = "C_d = 1×10⁻³")
lines!(ax, [0, xmax], cd_fit .* [0, xmax]; color = :red, linewidth = 1.5, label = @sprintf("fit %.2f×10⁻³", 1e3 * cd_fit))
axislegend(ax; position = :lt, framevisible = true, framecolor = :gray70, backgroundcolor = :white,
           labelsize = 8, rowgap = 0, patchsize = (14, 8), padding = (4, 4, 2, 2))
xlims!(ax, 0, xmax); ylims!(ax, 0, ymax)
Colorbar(fig[1, 2], hm; label = "log₁₀ count", width = 8)
# (b) C_d vs wind speed
ax = Axis(fig[1, 3]; xlabel = "|U10| (m s⁻¹)", ylabel = "C_d (×10⁻³)", title = "(b) C_d vs wind speed")
good = filter(b -> isfinite(b.q[2]), cd_bins)
band!(ax, [b.u for b in good], [1e3 * b.q[1] for b in good], [1e3 * b.q[3] for b in good]; color = (col_era5, 0.25))
scatterlines!(ax, [b.u for b in good], [1e3 * b.q[2] for b in good]; color = col_era5, markersize = 5, label = "ERA5 median (IQR)")
uu = range(2, 14; length = 100)
lines!(ax, uu, 1e3 .* cd_LY04.(uu); color = :black, linestyle = :dash, linewidth = 1, label = "Large & Yeager 2004 (neutral)")
hlines!(ax, 1e3 * cd_fit; color = :red, linewidth = 1, label = @sprintf("fit %.2f×10⁻³", 1e3 * cd_fit))
hlines!(ax, 1e3 * cd_q[2]; color = col_era5, linewidth = 1, linestyle = :dash,
        label = @sprintf("median %.2f×10⁻³ (|U10| > 3)", 1e3 * cd_q[2]))
xlims!(ax, 0, 14); ylims!(ax, 0, 3)
axislegend(ax; position = :rt, framevisible = false, labelsize = 8, rowgap = 0, patchsize = (14, 8))
colgap!(fig.layout, 8)
outname = joinpath(figure_path, "wind_stress_ERA5_Cd.png")
figsave(outname, fig)
println("saved plot to ", outname)

## ---------------------------------------------------------------- diff movie (3-hourly)
# Frames: the _diff figure every 3 h with colour limits fixed over the whole period,
# saved in <figure_path>/diff_png/; the movie (8 frames = 1 day per second) in figure_path.
make_movie = true
if make_movie
    frame_dir = joinpath(figure_path, "diff_png")
    mkpath(frame_dir)
    foreach(f -> endswith(f, ".png") && rm(joinpath(frame_dir, f)), readdir(frame_dir))   # no stale frames
    frames = findall(t -> hour(t) % 3 == 0, com_time)

    # fixed colour limits from all frames
    mag_n = hypot.(com_ncom_taux[:, :, frames], com_ncom_tauy[:, :, frames])
    mag_e = hypot.(com_era5_taux[:, :, frames], com_era5_tauy[:, :, frames])
    tau_lim_mov = round_lim(vcat(vec(mag_n), vec(mag_e)), 0.05)
    dif_lim_mov = round_lim(vcat(vec(com_ncom_taux[:, :, frames] .- com_era5_taux[:, :, frames]),
                                 vec(com_ncom_tauy[:, :, frames] .- com_era5_tauy[:, :, frames]),
                                 vec(mag_n .- mag_e)), 0.01)
    println("movie: $(length(frames)) frames, |τ| limit $tau_lim_mov, difference limit $dif_lim_mov N m⁻²")

    for (k, n) in enumerate(frames)
        diff_figure(joinpath(frame_dir, "wind_stress_NCOM_ERA5_diff_$(Dates.format(com_time[n], "yyyymmddHH")).png"), n;
                    tau_lim = tau_lim_mov, dif_lim = dif_lim_mov, dpi = 150, verbose = false)
        k % 50 == 0 && println("frame $k / $(length(frames))")
    end

    movie = joinpath(figure_path, "wind_stress_NCOM_ERA5_diff_3h.mp4")
    CairoMakie.Makie.FFMPEG_jll.ffmpeg() do exe
        run(`$exe -y -loglevel error -framerate 8 -pattern_type glob -i $(joinpath(frame_dir, "*.png"))
             -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -crf 20 $movie`)
    end
    println("saved movie to ", movie)
end
