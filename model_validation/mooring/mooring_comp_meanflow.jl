using NCDatasets, CairoMakie, Dates, Statistics, GibbsSeaWater, DSP
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")
##
## DIFFERENT data compare : 
    # (1) ncom parent roms 900m grid
    # (2) cmems parent roms 2700m grid
    # (3) ncom 900m model
    # (4) ncom 2700m model
    # (5) cmems 1/12 deg model
    # (6) french mooring data (only on "french" location)

mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hsinyi/figure/20261027_meanflow_comp"

# one entry per data source; `path` is a template where <moor> is replaced by the
# mooring name (french, M1..M4, CPIES1..9), see src_file below
struct Source
    key::String     # short name, used in code / file names
    label::String   # legend label
    path::String    # file to read, with <moor> for the mooring name
    grid::String    # ROMS only: grid file (h at the mooring); "" otherwise
    edata::String   # ROMS only: mooring index file (<moor>_mor = 0-based xi, eta, angle); "" otherwise
end
Source(key, label, path) = Source(key, label, path, "", "")

# ROMS 2700m (CMEMS parent) run: "3mon" (2022-08-24 to 11-29) or "1mon" (2022-08-24 to 09-29)
roms_cmems_run  = "3mon"
roms_cmems_path = Dict("3mon" => "/home/hsinyi/roms_data/mooring_ext_model/cmems3mon/<moor>.nc",
                       "1mon" => "/home/hsinyi/roms_data/mooring_ext_model/cmems2700/mooring_merged/<moor>.nc")

sources = [
    Source("roms_ncom",  "ROMS 900m (NCOM parent)",  "/home/hsinyi/roms_data/mooring_ext_model/full_3mon/mooring_merged/<moor>.nc",
           "/home/hsinyi/roms_data/grid/roms_grd_900m.nc", "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"),
    Source("ncom900",    "NCOM 900m",                "/home/hsinyi/roms_data/NCOM_mooring/NCOM_<moor>.nc"),
    Source("ncom2700",   "NCOM 2700m",               "/home/hsinyi/roms_data/NCOM_mooring_2700/NCOM_<moor>.nc"),
    Source("roms_cmems", "ROMS 2700m (CMEMS parent)", roms_cmems_path[roms_cmems_run],
           "/home/hsinyi/roms_data/grid/roms_cmems_grd_2700m.nc", "/home/hsinyi/roms_data/grid/roms_cmems_grd_2700m_mor_edata.nc"),
    Source("cmems",      "CMEMS 1/12°",              "/home/hsinyi/roms_data/GLOBAL_mooring/GLOBAL_<moor>.nc"),
    Source("french",     "French mooring",           "/home/hsinyi/data_notm/French_mooring"),   # folder: adcp_newalgo_raw_corr_angle.nc, ctd_corrected_raw.nc; only at "french"
]

src_file(s::Source, moor) = replace(s.path, "<moor>" => moor)
src(key) = sources[findfirst(s -> s.key == key, sources)]   # e.g. src_file(src("ncom2700"), "M2")

# ROMS station index of the mooring: (xi, eta) as a Julia rho index and the angle (rad) its u/v are
# output at (= local grid angle, so u/v are along the grid axes). The Julia rho index is the edata
# index + 1.5 (checked on both grids: lon_rho/lat_rho interpolated there give the mooring location).
function mooring_index(s::Source, moor)
    isempty(s.edata) && error("$(s.key) has no ROMS mooring index file")
    xi, eta, ang = NCDataset(s.edata) do ds
        ds["$(moor)_mor"][1, :]
    end
    return xi + 1.5, eta + 1.5, ang
end

# ROMS water depth (m, positive) at the mooring: bilinear interp of the grid h at the station index
function mooring_h(s::Source, moor)
    xi, eta, _ = mooring_index(s, moor)
    i, j = floor(Int, xi), floor(Int, eta)
    a, b = xi - i, eta - j
    NCDataset(s.grid) do ds
        h = ds["h"][i:i+1, j:j+1]
        (1-a)*(1-b)*h[1,1] + a*(1-b)*h[2,1] + (1-a)*b*h[1,2] + a*b*h[2,2]
    end
end

mkpath(figure_path)

# the mooring extract (french.nc) has no S-coordinate parameters, so take them from the initial file
ini_fname = "/home/hsinyi/roms_data/ini/roms_ini_900m2022082400.nc"
theta_s, theta_b, hc = NCDataset(ini_fname) do ds
    # theta_s/theta_b/hc live on a size-1 "one" dimension in this file, so [1]
    ds["theta_s"][1], ds["theta_b"][1], ds["hc"][1]
end
##
mor_loc = NCDataset(mooring_loc) do ds
    out = Dict{String,NTuple{2,Float64}}()
    for k in keys(ds.attrib)
        endswith(k, "_mor_info") || continue
        m = match(r"location:\((-?[\d.]+);(-?[\d.]+)\)", ds.attrib[k])
        m === nothing && continue
        out[replace(k, "_mor_info" => "")] = (parse(Float64, m[1]), parse(Float64, m[2]))
    end
    out
end


## a fixed z grid and save data into sturcture.
# Every source interpolated onto one fixed z grid at one mooring.
#   z          : height relative to the instantaneous sea surface (m, negative down, ascending)
#   fields     : (nz, nt), NaN outside the depth range of the data (no extrapolation)
#   u, v       : true east/north; ROMS is rotated from its grid axes, NCOM, CMEMS and the ADCP are already east/north
#   temp       : potential temperature (°C), salt: practical salinity (psu)
# T/S and u/v keep their own time axes, because the French CTD (6 min) and ADCP (30 min) sample
# at different times; for the models time_ts == time_uv.
struct MoorData
    z::Vector{Float64}
    time_uv::Vector{DateTime}
    u::Matrix{Float64}
    v::Matrix{Float64}
    time_ts::Vector{DateTime}
    temp::Matrix{Float64}
    salt::Matrix{Float64}
end

zg    = collect(-400.0:5.0:0.0)
tlim  = (DateTime(2022, 8, 24), DateTime(2022, 12, 1))   # records outside this period are dropped
t_ref = DateTime(1994, 1, 1)                             # ROMS ocean_time is in seconds since t_ref

intime(t) = findall(tlim[1] .<= t .<= tlim[2])
salt_range = (34.5, 37.5)   # French CTD salinity outside this (psu) is treated as a bad sample
rd(ds, name) = Float64.(nomissing(ds[name][:, :], NaN))   # (nz, nt) with NaN in the gaps

# (nz, nt) field f on levels z (vector, or (nz, nt) when the levels move; ascending in every column)
# onto the fixed levels zg, linear in z, column by column
function to_zgrid(z, f, zg)
    out = fill(NaN, length(zg), size(f, 2))
    for t in axes(f, 2)
        zc = z isa AbstractVector ? z : view(z, :, t)
        good = findall(isfinite.(zc) .& isfinite.(view(f, :, t)))
        length(good) < 2 && continue
        zt, ft = zc[good], f[good, t]
        for (k, d) in enumerate(zg)
            (d < zt[1] || d > zt[end]) && continue
            i = clamp(searchsortedlast(zt, d), 1, length(zt) - 1)
            w = (d - zt[i]) / (zt[i+1] - zt[i])
            out[k, t] = (1 - w) * ft[i] + w * ft[i+1]
        end
    end
    return out
end

"""
    load_moor(s, moor, zg) -> MoorData (or nothing if the source has no data at this mooring)

Read u, v, temp, salt of source `s` at mooring `moor` and put them on the fixed levels `zg`.
"""
function load_moor(s::Source, moor, zg)
    if startswith(s.key, "roms")
        # rows are s_rho levels, bottom first; theta_s/theta_b/hc from the 900m ini file are used for both runs
        ocean_time, zeta, temp, salt, u, v = NCDataset(src_file(s, moor)) do ds
            ds["ocean_time"][:], ds["zeta"][1, :],
            ds["temp"][1, :, :], ds["salt"][1, :, :], ds["u"][1, :, :], ds["v"][1, :, :]
        end
        time = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000))
        it = intime(time)
        N, nt = size(temp, 1), length(it)
        _, _, ang = mooring_index(s, moor)
        z_r, _ = zlevs3(fill(mooring_h(s, moor), 1, nt), reshape(zeta[it], 1, nt), theta_s, theta_b, hc, N, "r", "new2008")
        z = z_r[:, 1, :] .- zeta[it]'    # relative to the moving surface
        ue = u[:, it] .* cos(ang) .- v[:, it] .* sin(ang)
        vn = u[:, it] .* sin(ang) .+ v[:, it] .* cos(ang)
        return MoorData(zg, time[it], to_zgrid(z, ue, zg), to_zgrid(z, vn, zg),
                        time[it], to_zgrid(z, temp[:, it], zg), to_zgrid(z, salt[:, it], zg))

    elseif startswith(s.key, "ncom") || s.key == "cmems"
        # fixed z levels (positive up, deepest first); the surface moving by ssh is ignored
        time, z, temp, salt, u, v = NCDataset(src_file(s, moor)) do ds
            nomissing(ds["time"][:]), Float64.(ds["z"][:]),
            rd(ds, "temp"), rd(ds, "salt"), rd(ds, "u"), rd(ds, "v")
        end
        it = intime(time)
        f(a) = to_zgrid(z, a[:, it], zg)
        return MoorData(zg, time[it], f(u), f(v), time[it], f(temp), f(salt))

    elseif s.key == "french"
        moor == "french" || return nothing
        # CTD: 4 instruments, rows shallowest first, depth positive down -> flip to ascending z
        ctd_time, ctd_z, CT, SA, salt = NCDataset(joinpath(s.path, "ctd_corrected_raw.nc")) do ds
            flip(a) = reverse(a, dims=1)
            nomissing(ds["time"][:]), flip(-rd(ds, "depth")),
            flip(rd(ds, "conservative temperature")), flip(rd(ds, "absolute salinity")), flip(rd(ds, "salinity"))
        end
        # despike: a few single CTD samples have bad salinity (down to 26 psu at ~250 m on 10-02); drop T and S
        # of those samples (CT/SA are derived from it too). to_zgrid then interpolates over the gap.
        bad = .!(salt_range[1] .<= salt .<= salt_range[2]) .& .!isnan.(salt)
        salt[bad] .= NaN; CT[bad] .= NaN; SA[bad] .= NaN
        println("french CTD: removed $(count(bad)) samples with salinity outside $salt_range psu")
        # ADCP: 32 bins, bin 1 the deepest, bin depth (positive down) moves with the mooring
        adcp_time, adcp_z, u, v = NCDataset(joinpath(s.path, "adcp_newalgo_raw_corr_angle.nc")) do ds
            nomissing(ds["time"][:]), -rd(ds, "z"), rd(ds, "u"), rd(ds, "v")
        end
        ic, ia = intime(ctd_time), intime(adcp_time)
        ptemp = gsw_pt_from_ct.(SA[:, ic], CT[:, ic])   # potential temperature, like the models
        return MoorData(zg, adcp_time[ia], to_zgrid(adcp_z[:, ia], u[:, ia], zg), to_zgrid(adcp_z[:, ia], v[:, ia], zg),
                        ctd_time[ic], to_zgrid(ctd_z[:, ic], ptemp, zg), to_zgrid(ctd_z[:, ic], salt[:, ic], zg))
    end
    error("load_moor: no reader for source $(s.key)")
end

# all sources at one mooring: data["ncom2700"].u, ...
moor = "french"
data = Dict{String,MoorData}()
for s in sources
    d = load_moor(s, moor, zg)
    d === nothing && continue
    data[s.key] = d
    println(rpad(s.key, 11), ": u/v ", d.time_uv[1], " to ", d.time_uv[end], " (", length(d.time_uv), ")",
            ",  T/S ", d.time_ts[1], " to ", d.time_ts[end], " (", length(d.time_ts), ")")
end

## 28 hr lp filter
Tcut = 28.0       # low-pass cutoff period (hours)
filt_order = 4

# Butterworth + filtfilt (butter_filters.jl) along time, row by row of a (nz, nt) field. filtfilt needs an
# evenly sampled series with no NaN, so per row: keep the span between the first and last valid sample, fill
# the gaps inside it linearly in time, filter, then put NaN back where the data were missing.
function lowpass_rows(time, f, Tcut; N = filt_order)
    dt = median(Dates.value.(diff(time))) / 3_600_000      # sampling interval (hours), same unit as Tcut
    out = fill(NaN, size(f))
    for k in axes(f, 1)
        good = findall(!isnan, f[k, :])
        length(good) < 2 && continue
        y = f[k, good[1]:good[end]]
        for (a, b) in zip(good[1:end-1] .- good[1] .+ 1, good[2:end] .- good[1] .+ 1)   # fill the gaps
            for i in a+1:b-1
                y[i] = y[a] + (y[b] - y[a]) * (i - a) / (b - a)
            end
        end
        yf = lowhighpass_butter(y, Tcut, dt, N, "low")
        out[k, good] = yf[good .- good[1] .+ 1]
    end
    return out
end

lowpass(d::MoorData, Tcut) = MoorData(d.z,
    d.time_uv, lowpass_rows(d.time_uv, d.u, Tcut), lowpass_rows(d.time_uv, d.v, Tcut),
    d.time_ts, lowpass_rows(d.time_ts, d.temp, Tcut), lowpass_rows(d.time_ts, d.salt, Tcut))

# low-passed copy of every source, same layout: data_lp["ncom2700"].u, ...
data_lp = Dict(k => lowpass(d, Tcut) for (k, d) in data)

### ===== time-depth figures: one figure per variable, one panel per source (stacked), raw and low-passed =====
site = "$moor mooring ($(mor_loc[moor][2])°N, $(-mor_loc[moor][1])°W)"

# time axis as days since tlim[1], labelled with the real dates
tday(t) = Dates.value.(t .- tlim[1]) ./ 86_400_000
date_ticks = tlim[1]:Day(10):tlim[2]
xticks = (tday(date_ticks), Dates.format.(date_ticks, "mm/dd"))

speed(d)     = hypot.(d.u, d.v)
direction(d) = mod.(atand.(d.u, d.v), 360)    # direction the current flows TOWARD: 0° = north, 90° = east

# variable => (field, time axis, colour limits, contour levels, colormap, colourbar label)
# direction uses the cyclic :phase colormap (0° and 360° the same colour) and no contours
time_depth_vars = [
    "temp"      => (d -> d.temp,  d -> d.time_ts, (9, 30),     9:3:30,       :thermal, "Potential temperature (°C)"),
    "salt"      => (d -> d.salt,  d -> d.time_ts, (34.0, 37.0), 34:0.5:37,   :haline,  "Salinity (psu)"),
    "u"         => (d -> d.u,     d -> d.time_uv, (-1.5, 1.5), -1.5:0.5:1.5, :balance, "u, eastward (m/s)"),
    "v"         => (d -> d.v,     d -> d.time_uv, (-1.5, 1.5), -1.5:0.5:1.5, :balance, "v, northward (m/s)"),
    "speed"     => (speed,        d -> d.time_uv, (0, 1.5),    0:0.25:1.5,   :speed,   "Speed (m/s)"),
    "direction" => (direction,    d -> d.time_uv, (0, 360),    nothing,      :phase,   "Direction toward (° clockwise from north)"),
]

# model field f (nz, nt) on times t, blanked wherever the mooring field fm (nz, ntm) on times tm is NaN at the
# nearest mooring sample, or where the nearest mooring sample is more than 2 mooring steps away (a gap in the record)
function mask_like(t, f, tm, fm)
    out = copy(f)
    dtm = Millisecond(round(Int, 2 * median(Dates.value.(diff(tm)))))
    for n in axes(f, 2)
        j = clamp(searchsortedfirst(tm, t[n]), 2, length(tm))
        j = abs(tm[j-1] - t[n]) < abs(tm[j] - t[n]) ? j - 1 : j
        if abs(tm[j] - t[n]) > dtm
            out[:, n] .= NaN
        else
            out[isnan.(view(fm, :, j)), n] .= NaN
        end
    end
    return out
end

# masked = true: the models are shown only where (and when) the French mooring has data for that variable
function plot_time_depth(D, var; tag, masked = false)
    field, tfun, clim, levels, colormap, cblabel = Dict(time_depth_vars)[var]
    masked && !haskey(D, "french") && return nothing
    srcs = filter(s -> haskey(D, s.key), sources)
    fig = Figure(; figkw(18, 2.6 * length(srcs) + 1.5)..., figure_padding = (4, 8, 4, 4))
    axs, hm = Axis[], nothing
    for (row, s) in enumerate(srcs)
        d = D[s.key]
        ax = Axis(fig[row, 1]; title = s.label, titlealign = :left, titlegap = 2, titlesize = 9, ylabel = "z (m)",
                  xticks = xticks, yticks = -400:100:0, xgridvisible = false, ygridvisible = false)
        t, f = tfun(d), field(d)
        if masked && s.key != "french"
            m = D["french"]
            f = mask_like(t, f, tfun(m), field(m))
        end
        hm = heatmap!(ax, tday(t), d.z, f'; colormap = colormap, colorrange = clim)
        levels === nothing || contour!(ax, tday(t), d.z, f'; levels = levels, color = (:black, 0.3), linewidth = 0.6)
        push!(axs, ax)
    end
    linkaxes!(axs...)
    xlims!(axs[end], tday(tlim[1]), tday(tlim[2]))
    ylims!(axs[end], zg[1], zg[end])
    hidexdecorations!.(axs[1:end-1]; ticks = false)
    axs[end].xlabel = "Date (2022)"
    Colorbar(fig[1:length(srcs), 2], hm; label = cblabel, ticks = levels === nothing ? (0:90:360) : levels)
    Label(fig[0, :], "$(var), $(tag == "raw" ? "raw" : "$(round(Int, Tcut)) h low-passed"), at the $site" *
                     (masked ? ", masked to mooring coverage" : ""), fontsize = 11, tellwidth = false)
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 6)
    outdir = masked ? joinpath(figure_path, "masked") : figure_path
    mkpath(outdir)
    fname = joinpath(outdir, "$(moor)_$(var)_time_depth_$(tag)$(masked ? "_masked" : "").png")
    figsave(fname, fig)
    println("saved ", fname)
    return fig
end

for (D, tag) in ((data, "raw"), (data_lp, "lp$(round(Int, Tcut))h")), (var, _) in time_depth_vars, masked in (false, true)
    plot_time_depth(D, var; tag = tag, masked = masked)
end

## ===== line plots at fixed depths, and scatter plots of each model against the mooring =====
line_depths = [-200.0, -250.0, -300.0]                 # must be levels of zg
kd = [findfirst(==(d), zg) for d in line_depths]
depth_titles = ["$(round(Int, -d)) m" for d in line_depths]

# one colour per source in every line / scatter plot; short = label on the scatter axes
src_style = Dict(
    "roms_ncom"  => (color = "#2a78d6", short = "ROMS 900m"),
    "ncom900"    => (color = "#eb6834", short = "NCOM 900m"),
    "ncom2700"   => (color = "#c9a400", short = "NCOM 2700m"),
    "roms_cmems" => (color = "#8e44ad", short = "ROMS 2700m"),
    "cmems"      => (color = "#2e9d5b", short = "CMEMS"),
    "french"     => (color = :black,    short = "Mooring"),
)
var_unit = Dict("temp" => "°C", "salt" => "psu", "u" => "m/s", "v" => "m/s", "speed" => "m/s", "direction" => "°")
tagname(tag) = tag == "raw" ? "raw" : "$(round(Int, Tcut)) h low-passed"

# one panel per depth, one line per source (direction as dots: a line would jump at the 0/360 wrap)
function plot_lines_at_depth(D, var; tag)
    field, tfun, _, _, _, _ = Dict(time_depth_vars)[var]
    ylab = "$var ($(var_unit[var]))"
    srcs = filter(s -> haskey(D, s.key), sources)
    fig = Figure(; figkw(18, 4.2 * length(kd) + 2.5)..., figure_padding = (4, 8, 4, 4))
    axs = Axis[]
    for (row, k) in enumerate(kd)
        ax = Axis(fig[row, 1]; title = depth_titles[row], titlealign = :left, titlegap = 2, titlesize = 9,
                  ylabel = ylab, xticks = xticks)
        var == "direction" && (ax.yticks = 0:90:360)
        for s in srcs
            d, st = D[s.key], src_style[s.key]
            t, f = tday(tfun(d)), field(d)[k, :]
            if var == "direction"
                scatter!(ax, t, f; color = st.color, markersize = 2, label = s.label)
            else
                lines!(ax, t, f; color = st.color, linewidth = s.key == "french" ? 1.2 : 0.9, label = s.label)
            end
        end
        push!(axs, ax)
    end
    linkxaxes!(axs...)                   # y is left free: each depth has its own range
    xlims!(axs[end], tday(tlim[1]), tday(tlim[2]))
    hidexdecorations!.(axs[1:end-1]; ticks = false, grid = false)
    axs[end].xlabel = "Date (2022)"
    Legend(fig[length(kd)+1, 1], axs[1]; orientation = :horizontal, nbanks = 2, framevisible = false,
           tellwidth = false, markersize = 8)
    Label(fig[0, :], "$var at fixed depths, $(tagname(tag)), at the $site", fontsize = 11, tellwidth = false)
    rowgap!(fig.layout, 5)
    outdir = joinpath(figure_path, "lines"); mkpath(outdir)
    fname = joinpath(outdir, "$(moor)_$(var)_lines_at_depth_$(tag).png")
    figsave(fname, fig)
    println("saved ", fname)
    return fig
end

for (D, tag) in ((data, "raw"), (data_lp, "lp$(round(Int, Tcut))h")), (var, _) in time_depth_vars
    plot_lines_at_depth(D, var; tag = tag)
end

# --- scatter: model vs mooring, on common time bins ---
# Average a (nrow, nt) field into time bins of length dt starting at tlim[1], the same for every source, NaN
# ignored. Returns (bin-centre times, (nrow, nbin)). A bin is NaN when it holds fewer than minfrac of the
# samples a full bin would have.
function time_average(time, f, dt; t_start = tlim[1], t_end = tlim[2], minfrac = 0.5)
    ms = Dates.toms(dt)
    nb = floor(Int, Dates.value(t_end - t_start) / ms) + 1
    tb = t_start .+ Millisecond.(round.(Int, ((1:nb) .- 0.5) .* ms))
    nfull = ms / median(Dates.value.(diff(time)))          # samples in a full bin
    acc, cnt = zeros(size(f, 1), nb), zeros(Int, size(f, 1), nb)
    for (j, t) in enumerate(time)
        b = floor(Int, Dates.value(t - t_start) / ms) + 1
        1 <= b <= nb || continue
        for k in axes(f, 1)
            isnan(f[k, j]) && continue
            acc[k, b] += f[k, j]
            cnt[k, b] += 1
        end
    end
    out = acc ./ cnt
    out[cnt .< max(1, minfrac * nfull)] .= NaN
    return tb, out
end

# all variables at line_depths on bins of dt; speed and direction from the binned u, v (a direction
# cannot be averaged as a number, 359° and 1° would give 180°)
function binned(d, dt)
    tb, u = time_average(d.time_uv, d.u[kd, :], dt)
    v = time_average(d.time_uv, d.v[kd, :], dt)[2]
    return Dict("time" => tb, "u" => u, "v" => v, "speed" => hypot.(u, v), "direction" => mod.(atand.(u, v), 360),
                "temp" => time_average(d.time_ts, d.temp[kd, :], dt)[2],
                "salt" => time_average(d.time_ts, d.salt[kd, :], dt)[2])
end

# raw: 6 h bins (the CMEMS time step; the others are 6 h means); low-passed: 1 day (nothing faster is left)
scatter_dt = Dict("raw" => Hour(6), "lp$(round(Int, Tcut))h" => Day(1))

# One row per model, one column per depth, with the 1:1 line; points coloured by date. r = correlation,
# N = number of bins where both are valid. For direction the axes are 0–360 and r is replaced by the mean
# absolute angular difference (the short way round), so points in the top-left / bottom-right corners are close too.
# pairs = [(y source key, x source key), …], one row each; default: every model against the mooring.
# The x axis is labelled on every row when the rows have different x sources, otherwise on the last row only.
function plot_scatter(B, var; tag, unit,
                      pairs = [(s.key, "french") for s in sources if s.key != "french" && haskey(B, s.key)],
                      subdir = "scatter", what = "models vs mooring")
    isdir = var == "direction"
    crange = (tday(tlim[1]), tday(tlim[2]))
    ms = tag == "raw" ? 4 : 8
    xlabel_all = length(unique(last.(pairs))) > 1
    fig = Figure(; figkw(18, (xlabel_all ? 5.7 : 5.2) * length(pairs) + 2)..., figure_padding = (4, 8, 4, 4))
    for (row, (ykey, xkey)) in enumerate(pairs), (k, dtitle) in enumerate(depth_titles)
        x, y, st = B[xkey][var], B[ykey][var], src_style[ykey]
        tb = B[xkey]["time"]
        good = findall(.!isnan.(x[k, :]) .& .!isnan.(y[k, :]))
        xs, ys = x[k, good], y[k, good]
        stat = isempty(good) ? "no data" :
               isdir ? "mean |Δ| = $(round(Int, mean(abs.(mod.(ys .- xs .+ 180, 360) .- 180))))°" :
               "r = $(length(good) > 2 ? round(cor(xs, ys), digits = 2) : NaN)"
        ax = Axis(fig[row, k]; title = "$dtitle: $stat, N = $(length(good))", titlecolor = st.color,
                  titlealign = :left, titlegap = 2, titlesize = 8, aspect = 1,
                  xlabel = xlabel_all || row == length(pairs) ? "$(src_style[xkey].short) ($unit)" : "",
                  ylabel = k == 1 ? "$(st.short) ($unit)" : "")
        isempty(good) && continue
        lo, hi = isdir ? (0.0, 360.0) : (min(minimum(xs), minimum(ys)), max(maximum(xs), maximum(ys)))
        pad = isdir ? 0.0 : max(0.05 * (hi - lo), 1e-6)
        isdir ? (ax.xticks = 0:90:360; ax.yticks = 0:90:360) : (ax.xticks = LinearTicks(3); ax.yticks = LinearTicks(3))
        lines!(ax, [lo - pad, hi + pad], [lo - pad, hi + pad]; color = (:black, 0.5), linewidth = 1)
        scatter!(ax, xs, ys; color = tday(tb[good]), colormap = :viridis, colorrange = crange,
                 markersize = ms, strokewidth = ms > 6 ? 0.4 : 0, strokecolor = :black)
        limits!(ax, lo - pad, hi + pad, lo - pad, hi + pad)
    end
    cb_dates = DateTime(2022, 9, 1):Day(15):tlim[2]
    Colorbar(fig[1:length(pairs), length(depth_titles) + 1]; colormap = :viridis, limits = crange, label = "Date (2022)",
             ticks = (tday(cb_dates), Dates.format.(cb_dates, "mm/dd")))
    Label(fig[0, :], "$var, $(tagname(tag)), $(scatter_dt[tag]) bins: $what, $site", fontsize = 11, tellwidth = false)
    rowgap!(fig.layout, 4); colgap!(fig.layout, 6)
    outdir = joinpath(figure_path, subdir); mkpath(outdir)
    fname = joinpath(outdir, "$(moor)_$(var)_scatter_at_depth_$(tag).png")
    figsave(fname, fig)
    println("saved ", fname)
    return fig
end

# the CMEMS family only: ROMS 2700m (CMEMS parent) vs mooring, CMEMS vs mooring, ROMS 2700m vs its parent CMEMS
cmems_pairs = [("roms_cmems", "french"), ("cmems", "french"), ("roms_cmems", "cmems")]

if haskey(data, "french")
    for (D, tag) in ((data, "raw"), (data_lp, "lp$(round(Int, Tcut))h"))
        B = Dict(k => binned(d, scatter_dt[tag]) for (k, d) in D)
        for (var, _) in time_depth_vars
            plot_scatter(B, var; tag = tag, unit = var_unit[var])
            plot_scatter(B, var; tag = tag, unit = var_unit[var], pairs = cmems_pairs, subdir = "scatter_cmems",
                         what = "CMEMS family")
        end
    end
end
