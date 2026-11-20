using NCDatasets, CairoMakie, Dates, Statistics, GibbsSeaWater, DSP
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")
##
## DIFFERENT data compare : 
    # (1) HYCOM parent roms 2700m grid  (design_1113_hycom_2700, new design grid roms_grd_2700m_1104)
    # (2) GLORYS (CMEMS) parent roms 2700m grid  (design_1116_cmems_2700, same grid)
    # (3) HYCOM GOFS 3.5 1/25° (HYCOM_DATA_NC_2700m_1104, with tides)  OR  ncom 900m model (see `third_model` below)
    # (4) ncom 2700m model
    # (5) glorys 1/12 deg model
    # (6) french mooring data (only on "french" location)

mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # HPC output dir — contains avg/dia/his/rst files mixed together
# 3rd model in the comparison: "hycom" (GOFS 3.5, the parent of ROMS (1); replaces NCOM 900m) or "ncom900"
third_model = "hycom"
figure_path = third_model == "hycom" ? "/home/hsinyi/figure/20261027_meanflow_comp/1119_update" :
                                       "/home/hsinyi/figure/20261027_meanflow_comp/1119_update_ncom900"

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

# both ROMS runs are on the new design 2700m grid (1104, hmin 10 m), same S-coordinate (theta_s 6, theta_b 3, hc 250)
#   design_1113_hycom_2700: HYCOM ini/bry, 2022-08-22 to 11-29
#   design_1116_cmems_2700: CMEMS (GLORYS) ini/bry, 2022-08-01 (tide spin-up) to 11-29
roms_grd_1104   = "/home/hsinyi/roms_data/new_design_fix/roms_grd_2700m_1104.nc"
roms_edata_1104 = "/home/hsinyi/roms_data/new_design_fix/roms_grd_2700m_1104_edata.nc"

sources = [
    Source("roms_hycom", "ROMS 2700m (HYCOM parent)", "/home/hsinyi/roms_data/mooring_ext_model/design_1113_hycom_2700/<moor>.nc",
           roms_grd_1104, roms_edata_1104),
    Source("ncom2700",   "NCOM 2700m",               "/home/hsinyi/roms_data/NCOM_mooring_2700/NCOM_<moor>.nc"),
    third_model == "hycom" ?
    Source("hycom",      "HYCOM GOFS 3.5 1/25°",     "/home/hsinyi/roms_data/HYCOM_mooring_NC/HYCOM_<moor>.nc") :
    Source("ncom900",    "NCOM 900m",                "/home/hsinyi/roms_data/NCOM_mooring/NCOM_<moor>.nc"),
    Source("roms_glorys", "ROMS 2700m (GLORYS parent)", "/home/hsinyi/roms_data/mooring_ext_model/design_1116_cmems_2700/<moor>.nc",
           roms_grd_1104, roms_edata_1104),
    Source("glorys",      "GLORYS 1/12°",              "/home/hsinyi/roms_data/GLOBAL_mooring/GLOBAL_<moor>.nc"),
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
# (900m ini: theta_s 6, theta_b 3, hc 250 = the same as the design_1104 2700m runs)
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
#   u, v       : true east/north; ROMS is rotated from its grid axes, NCOM, GLORYS and the ADCP are already east/north
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
        # rows are s_rho levels, bottom first; theta_s/theta_b/hc from the 900m ini file are used for both runs (same values)
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

    elseif startswith(s.key, "ncom") || s.key in ("glorys", "hycom")
        # fixed z levels (positive up, deepest first); the surface moving by ssh is ignored
        time, z, temp, salt, u, v = NCDataset(src_file(s, moor)) do ds
            nomissing(ds["time"][:]), Float64.(ds["z"][:]),
            rd(ds, "temp"), rd(ds, "salt"), rd(ds, "u"), rd(ds, "v")
        end
        it = intime(time)
        if s.key == "hycom"
            # HYCOM temperature is IN-SITU (sea_water_temperature): convert to potential temperature like the other sources
            lon0, lat0 = mor_loc[moor]
            p  = gsw_p_from_z.(z, lat0)                          # (nz,) dbar
            SA = gsw_sa_from_sp.(salt, p, lon0, lat0)
            temp = gsw_pt0_from_t.(SA, temp, p)                  # NaN stays NaN
        end
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
# filt(y, dt) does the filtering of one gap-free row sampled every dt hours.
function filter_rows(time, f, filt)
    dt = median(Dates.value.(diff(time))) / 3_600_000      # sampling interval (hours)
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
        yf = filt(y, dt)
        out[k, good] = yf[good .- good[1] .+ 1]
    end
    return out
end

lowpass_rows(time, f, Tcut; N = filt_order) = filter_rows(time, f, (y, dt) -> lowhighpass_butter(y, Tcut, dt, N, "low"))

filter_data(d::MoorData, filt) = MoorData(d.z,
    d.time_uv, filter_rows(d.time_uv, d.u, filt), filter_rows(d.time_uv, d.v, filt),
    d.time_ts, filter_rows(d.time_ts, d.temp, filt), filter_rows(d.time_ts, d.salt, filt))
lowpass(d::MoorData, Tcut) = filter_data(d, (y, dt) -> lowhighpass_butter(y, Tcut, dt, filt_order, "low"))

# low-passed copy of every source, same layout: data_lp["ncom2700"].u, ...
data_lp = Dict(k => lowpass(d, Tcut) for (k, d) in data)

## 9-15 h band-pass (semidiurnal / near-inertial) and 9 h high-pass
Tband = (9.0, 15.0)   # band-pass cutoff periods (hours)
Thp   = 9.0           # high-pass cutoff period (hours)

# A source can only be band/high-passed when its Nyquist period (2 dt) is shorter than the shortest cutoff:
# GLORYS (6 h, Nyquist 12 h) is skipped for both. HYCOM GOFS 3.5 has tides (3 h, Nyquist 6 h), so it is kept.
no_tide = String[]   # sources without tides, left out of band/high-pass (GOFS 3.1 HYCOM would go here)
dt_hours(d::MoorData) = median(Dates.value.(diff(d.time_uv))) / 3_600_000
resolves(d::MoorData, Tmin) = 2 * dt_hours(d) < Tmin
for (k, d) in data
    resolves(d, min(Tband[1], Thp)) || println("$k: sampled every $(dt_hours(d)) h, too coarse for the $(Tband[1]) h cutoff, skipped in band/high-pass")
end

data_bp = Dict(k => filter_data(d, (y, dt) -> bandpass_butter(y, Tband[1], Tband[2], dt, filt_order))
               for (k, d) in data if resolves(d, Tband[1]) && !(k in no_tide))
data_hp = Dict(k => filter_data(d, (y, dt) -> lowhighpass_butter(y, Thp, dt, filt_order, "high"))
               for (k, d) in data if resolves(d, Thp) && !(k in no_tide))

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

# eddy kinetic energy (same definition as in mooring_comp.jl):
#   EKE(z, t) = 0.5 * ((u - ubar)^2 + (v - vbar)^2)   (m²/s²)
# ubar(z), vbar(z) = time means at each depth over the source's own valid samples (NaN ignored), so for the
# ADCP rows with gaps the mean covers less of the period than for the models.
# Only plotted from the low-passed data (data_lp): EKE of the 28 h low-passed flow. Kept out of time_depth_vars
# so the line / scatter loops below do not pick it up.
rowmean(a) = [(g = filter(!isnan, a[k, :]); isempty(g) ? NaN : mean(g)) for k in axes(a, 1)]
eke(d) = 0.5 .* ((d.u .- rowmean(d.u)) .^ 2 .+ (d.v .- rowmean(d.v)) .^ 2)
eke_max = 0.3      # colour limit (m²/s²); lower it if most of the panels look washed out
# KE of the band-/high-passed flow: the filtered u, v already have ~zero mean, so KE = 0.5 * (u^2 + v^2)
ke(d) = 0.5 .* (d.u .^ 2 .+ d.v .^ 2)
# envelope of the KE: square it, low-pass at Tcut (28 h), square root back (max(0, ...) guards against
# small negative values from filter ringing)
ke_env(d) = sqrt.(max.(lowpass_rows(d.time_uv, ke(d) .^ 2, Tcut), 0))
ke_max = 0.05      # colour limit (m²/s²) for the band-/high-passed KE; adjust after a first look
extra_time_depth_vars = [
    "EKE" => (eke, d -> d.time_uv, (0, eke_max), 0:eke_max/6:eke_max, :amp, "EKE (m²/s²)"),
    "KE"  => (ke,  d -> d.time_uv, (0, ke_max),  0:ke_max/5:ke_max,   :amp, "KE (m²/s²)"),
    "KE_envelope" => (ke_env, d -> d.time_uv, (0, ke_max), 0:ke_max/5:ke_max, :amp, "KE envelope (m²/s²)"),
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

# figure-title text of each filter tag
tag_desc = Dict("raw" => "raw", "lp$(round(Int, Tcut))h" => "$(round(Int, Tcut)) h low-passed",
                "bp$(round(Int, Tband[1]))-$(round(Int, Tband[2]))h" => "$(round(Int, Tband[1]))-$(round(Int, Tband[2])) h band-passed",
                "hp$(round(Int, Thp))h" => "$(round(Int, Thp)) h high-passed")

# KE and KE_envelope (band-/high-pass) are drawn without contour lines; its levels are only the colourbar ticks
# masked = true: the models are shown only where (and when) the French mooring has data for that variable
function plot_time_depth(D, var; tag, masked = false)
    field, tfun, clim, levels, colormap, cblabel = Dict(vcat(time_depth_vars, extra_time_depth_vars))[var]
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
        (levels === nothing || startswith(var, "KE")) || contour!(ax, tday(t), d.z, f'; levels = levels, color = (:black, 0.3), linewidth = 0.6)
        push!(axs, ax)
    end
    linkaxes!(axs...)
    xlims!(axs[end], tday(tlim[1]), tday(tlim[2]))
    ylims!(axs[end], zg[1], zg[end])
    hidexdecorations!.(axs[1:end-1]; ticks = false)
    axs[end].xlabel = "Date (2022)"
    Colorbar(fig[1:length(srcs), 2], hm; label = cblabel, ticks = levels === nothing ? (0:90:360) : levels)
    Label(fig[0, :], "$(var), $(tag_desc[tag]), at the $site" *
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

# EKE of the 28 h low-passed flow (low-passed data only)
for masked in (false, true)
    plot_time_depth(data_lp, "EKE"; tag = "lp$(round(Int, Tcut))h", masked = masked)
end

# KE of the 9-15 h band-passed and the 9 h high-passed flow
for (D, tag) in ((data_bp, "bp$(round(Int, Tband[1]))-$(round(Int, Tband[2]))h"), (data_hp, "hp$(round(Int, Thp))h")),
    masked in (false, true)
    plot_time_depth(D, "KE"; tag = tag, masked = masked)
    plot_time_depth(D, "KE_envelope"; tag = tag, masked = masked)
end

## ===== line plots at fixed depths, and scatter plots of each model against the mooring =====
line_depths = [-200.0, -250.0, -300.0]                 # must be levels of zg
kd = [findfirst(==(d), zg) for d in line_depths]
depth_titles = ["$(round(Int, -d)) m" for d in line_depths]

# one colour per source in every line / scatter plot; short = label on the scatter axes
src_style = Dict(
    "roms_hycom" => (color = "#2a78d6", short = "ROMS 2700m H"),
    "ncom900"    => (color = "#eb6834", short = "NCOM 900m"),
    "hycom"      => (color = "#eb6834", short = "HYCOM"),       # takes NCOM 900m's place (third_model)
    "ncom2700"   => (color = "#c9a400", short = "NCOM 2700m"),
    "roms_glorys" => (color = "#8e44ad", short = "ROMS 2700m G"),
    "glorys"      => (color = "#2e9d5b", short = "GLORYS"),
    "french"     => (color = :black,    short = "Mooring"),
)
var_unit = Dict("temp" => "°C", "salt" => "psu", "u" => "m/s", "v" => "m/s", "speed" => "m/s", "direction" => "°",
                "EKE" => "m²/s²", "KE" => "m²/s²", "KE_envelope" => "m²/s²")
energy_vars = ("EKE", "KE", "KE_envelope")   # drawn on log axes in the scatter plots
tagname(tag) = tag_desc[tag]

# one panel per depth, one line per source (direction as dots: a line would jump at the 0/360 wrap)
function plot_lines_at_depth(D, var; tag)
    field, tfun, _, _, _, _ = Dict(vcat(time_depth_vars, extra_time_depth_vars))[var]
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

# energy at the fixed depths: EKE of the 28 h low-passed flow, and the KE envelope (KE itself oscillates
# at twice the tidal frequency, the envelope shows the spring-neap / event changes) of the 9-15 h band-passed
# and 9 h high-passed flow
plot_lines_at_depth(data_lp, "EKE"; tag = "lp$(round(Int, Tcut))h")
for (D, tag) in ((data_bp, "bp$(round(Int, Tband[1]))-$(round(Int, Tband[2]))h"), (data_hp, "hp$(round(Int, Thp))h"))
    plot_lines_at_depth(D, "KE_envelope"; tag = tag)
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
# energy = ["EKE" => eke, ...]: extra (nz, nt) fields on the u/v time axis, computed at full resolution and then
# bin-averaged (a bin mean of KE is the mean energy over the bin, not the KE of the mean flow)
function binned(d, dt; energy = Pair{String,Function}[])
    tb, u = time_average(d.time_uv, d.u[kd, :], dt)
    v = time_average(d.time_uv, d.v[kd, :], dt)[2]
    B = Dict("time" => tb, "u" => u, "v" => v, "speed" => hypot.(u, v), "direction" => mod.(atand.(u, v), 360),
             "temp" => time_average(d.time_ts, d.temp[kd, :], dt)[2],
             "salt" => time_average(d.time_ts, d.salt[kd, :], dt)[2])
    for (name, f) in energy
        B[name] = time_average(d.time_uv, f(d)[kd, :], dt)[2]
    end
    return B
end

# raw: 6 h bins (the GLORYS time step; the others are 6 h means); low-passed: 1 day (nothing faster is left);
# band-/high-passed KE: 1 day (averages the KE over ~2 semidiurnal cycles)
scatter_dt = Dict("raw" => Hour(6), "lp$(round(Int, Tcut))h" => Day(1),
                  "bp$(round(Int, Tband[1]))-$(round(Int, Tband[2]))h" => Day(1), "hp$(round(Int, Thp))h" => Day(1))

# Willmott (1981) index of agreement of model m against observation o (vectors, NaN pairs ignored):
#   d = 1 - Σ(M - O)² / Σ(|M - Ō| + |O - Ō|)²,   Ō = mean of the observation
# 1 = perfect, 0 = no skill; unlike r it penalises a bias and a wrong amplitude as well as a wrong phase.
function willmott(m, o)
    g = .!isnan.(m) .& .!isnan.(o)
    count(g) < 3 && return NaN
    m, o = m[g], o[g]
    ob = mean(o)
    den = sum((abs.(m .- ob) .+ abs.(o .- ob)) .^ 2)
    return den > 0 ? 1 - sum((m .- o) .^ 2) / den : NaN
end

# Murphy (1988) skill score of model m against observation o, the climatology (mean of the observation) as
# the reference forecast:
#   SS = 1 - MSE(m, o) / MSE(Ō, o),   MSE(Ō, o) = (1/N) Σ (Ō - Oᵢ)² = σ_o²
# 1 = perfect, 0 = no better than the constant Ō, < 0 = worse than Ō (unbounded below).
function murphy(m, o)
    g = .!isnan.(m) .& .!isnan.(o)
    count(g) < 3 && return NaN
    m, o = m[g], o[g]
    v = mean((o .- mean(o)) .^ 2)
    return v > 0 ? 1 - mean((m .- o) .^ 2) / v : NaN
end

# One row per model, one column per depth, with the 1:1 line; points coloured by date. r = correlation,
# d = Willmott skill (see willmott, on the linear values also for energy),
# N = number of bins where both are valid. For direction the axes are 0–360 and r is replaced by the mean
# absolute angular difference (the short way round), so points in the top-left / bottom-right corners are close too.
# Energy (EKE, KE) spans orders of magnitude: log10 axes, r is the correlation of log10 values, and
# ratio = median of model / mooring (> 1: the model has too much energy).
# pairs = [(y source key, x source key), …], one row each; default: every model against the mooring.
# The x axis is labelled on every row when the rows have different x sources, otherwise on the last row only.
function plot_scatter(B, var; tag, unit,
                      pairs = [(s.key, "french") for s in sources if s.key != "french" && haskey(B, s.key)],
                      subdir = "scatter", what = "models vs mooring")
    isdir = var == "direction"
    islog = var in energy_vars
    crange = (tday(tlim[1]), tday(tlim[2]))
    ms = tag == "raw" ? 4 : 8
    xlabel_all = length(unique(last.(pairs))) > 1
    fig = Figure(; figkw(18, (xlabel_all ? 5.7 : 5.2) * length(pairs) + 2)..., figure_padding = (4, 8, 4, 4))
    for (row, (ykey, xkey)) in enumerate(pairs), (k, dtitle) in enumerate(depth_titles)
        x, y, st = B[xkey][var], B[ykey][var], src_style[ykey]
        tb = B[xkey]["time"]
        ok = .!isnan.(x[k, :]) .& .!isnan.(y[k, :])
        islog && (ok .&= (x[k, :] .> 0) .& (y[k, :] .> 0))
        good = findall(ok)
        xs, ys = x[k, good], y[k, good]
        # two title lines: (line 1, start of line 2); d = Willmott skill of y (model) against x (reference)
        dstr = "d = $(round(willmott(ys, xs), digits = 2))"
        stat1, stat2 = isempty(good) ? ("no data", "") :
               isdir ? ("mean |Δ| = $(round(Int, mean(abs.(mod.(ys .- xs .+ 180, 360) .- 180))))°", "") :
               islog ? ("r(log) = $(length(good) > 2 ? round(cor(log10.(xs), log10.(ys)), digits = 2) : NaN), $dstr",
                        "ratio = $(round(median(ys ./ xs), sigdigits = 2)), ") :
               ("r = $(length(good) > 2 ? round(cor(xs, ys), digits = 2) : NaN), $dstr", "")
        ax = Axis(fig[row, k]; title = "$dtitle: $stat1\n$(stat2)N = $(length(good))", titlecolor = st.color,
                  titlealign = :left, titlegap = 2, titlesize = 8, aspect = 1,
                  xlabel = xlabel_all || row == length(pairs) ? "$(src_style[xkey].short) ($unit)" : "",
                  ylabel = k == 1 ? "$(st.short) ($unit)" : "",
                  xscale = islog ? log10 : identity, yscale = islog ? log10 : identity)
        isempty(good) && continue
        lo, hi = isdir ? (0.0, 360.0) : (min(minimum(xs), minimum(ys)), max(maximum(xs), maximum(ys)))
        if islog
            lo, hi = lo / 1.5, hi * 1.5          # padding, multiplicative on log axes
            dec = 10.0 .^ (ceil(Int, log10(lo)):floor(Int, log10(hi)))   # ticks at whole decades only
            declab = ["10" * map(c -> Dict(zip("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))[c], string(round(Int, log10(v)))) for v in dec]
            ax.xticks = (dec, declab); ax.yticks = (dec, declab)
        else
            pad = isdir ? 0.0 : max(0.05 * (hi - lo), 1e-6)
            lo, hi = lo - pad, hi + pad
            isdir ? (ax.xticks = 0:90:360; ax.yticks = 0:90:360) : (ax.xticks = LinearTicks(3); ax.yticks = LinearTicks(3))
        end
        lines!(ax, [lo, hi], [lo, hi]; color = (:black, 0.5), linewidth = 1)
        scatter!(ax, xs, ys; color = tday(tb[good]), colormap = :viridis, colorrange = crange,
                 markersize = ms, strokewidth = ms > 6 ? 0.4 : 0, strokecolor = :black)
        limits!(ax, lo, hi, lo, hi)
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

# the GLORYS family only: ROMS 2700m (GLORYS parent) vs mooring, GLORYS vs mooring, ROMS 2700m vs its parent GLORYS
glorys_pairs = [("roms_glorys", "french"), ("glorys", "french"), ("roms_glorys", "glorys")]

Bs = Dict{String,Any}()    # binned data of every tag, reused in the quantification section
if haskey(data, "french")
    for (D, tag) in ((data, "raw"), (data_lp, "lp$(round(Int, Tcut))h"))
        B = Bs[tag] = Dict(k => binned(d, scatter_dt[tag]) for (k, d) in D)
        for (var, _) in time_depth_vars
            plot_scatter(B, var; tag = tag, unit = var_unit[var])
            plot_scatter(B, var; tag = tag, unit = var_unit[var], pairs = glorys_pairs, subdir = "scatter_glorys",
                         what = "GLORYS family")
        end
    end

    # energy: EKE of the 28 h low-passed flow, KE of the 9-15 h band-passed and 9 h high-passed flow,
    # daily means at the fixed depths. GLORYS (6 h) has no band-/high-passed data, so the GLORYS family
    # figure is made for EKE only.
    lp_tag = "lp$(round(Int, Tcut))h"
    for (D, tag, var, f) in ((data_lp, lp_tag, "EKE", eke),
                             (data_bp, "bp$(round(Int, Tband[1]))-$(round(Int, Tband[2]))h", "KE", ke),
                             (data_hp, "hp$(round(Int, Thp))h", "KE", ke))
        B = Dict(k => binned(d, scatter_dt[tag]; energy = [var => f]) for (k, d) in D)
        Bs[tag] = B            # for lp this replaces the set above with the same values plus EKE
        plot_scatter(B, var; tag = tag, unit = var_unit[var])
        tag == lp_tag && plot_scatter(B, var; tag = tag, unit = var_unit[var], pairs = glorys_pairs,
                                      subdir = "scatter_glorys", what = "GLORYS family")
    end
end

## ===== quantification: Willmott skill d and r² of every model against the mooring =====
# Same samples as the scatter plots: the binned values (raw 6 h, filtered 1 day bins) at line_depths, pairs where
# both are valid. Both metrics on the linear values (also for EKE / KE). Direction is left out (not a linear
# quantity). Saved as a CSV table and as one heatmap per metric: one panel per depth, models x variables.
if haskey(data, "french")
    lp_tag = "lp$(round(Int, Tcut))h"
    bp_tag = "bp$(round(Int, Tband[1]))-$(round(Int, Tband[2]))h"
    hp_tag = "hp$(round(Int, Thp))h"
    skill_vars = ("temp", "salt", "u", "v", "speed")
    skill_cols = vcat([("raw", v) for v in skill_vars], [(lp_tag, v) for v in skill_vars],
                      [(lp_tag, "EKE"), (bp_tag, "KE"), (hp_tag, "KE")])
    skill_models = [s.key for s in sources if s.key != "french"]

    # skill[(metric)][depth index] = (nmodel, ncol) matrix, NaN where the model has no such data
    skill = Dict(m => [fill(NaN, length(skill_models), length(skill_cols)) for _ in kd] for m in ("d", "r2", "ss"))
    rows = String[]
    for (c, (tag, var)) in enumerate(skill_cols), (i, mk) in enumerate(skill_models), (k, dtitle) in enumerate(depth_titles)
        B = Bs[tag]
        haskey(B, mk) && haskey(B, "french") || continue
        mo, ob = B[mk][var][k, :], B["french"][var][k, :]
        g = .!isnan.(mo) .& .!isnan.(ob)
        n = count(g)
        n < 3 && continue
        d, r, ss = willmott(mo, ob), cor(mo[g], ob[g]), murphy(mo, ob)
        skill["d"][k][i, c], skill["r2"][k][i, c], skill["ss"][k][i, c] = d, r^2, ss
        # Murphy (1988) decomposition, population std (1/N) so that it is exact:
        #   SS = r² - (r - σm/σo)² - ((m̄ - ō)/σo)²
        #        correlation   conditional bias   unconditional bias
        #                      (amplitude)        (mean offset)
        σm, σo = std(mo[g]; corrected = false), std(ob[g]; corrected = false)
        bias = mean(mo[g] .- ob[g])
        cond_bias, uncond_bias = (r - σm / σo)^2, (bias / σo)^2
        abs(r^2 - cond_bias - uncond_bias - ss) < 1e-8 * max(1, abs(ss)) ||
            @warn "Murphy decomposition does not add up for $mk $tag $var $(depth_titles[k])"
        push!(rows, join((tag, var, -line_depths[k], mk, n, round(r, digits = 4), round(r^2, digits = 4),
                          round(d, digits = 4), round(ss, digits = 4),
                          round(cond_bias, digits = 4), round(uncond_bias, digits = 4), round(σm / σo, digits = 4),
                          round(bias, sigdigits = 4), round(sqrt(mean((mo[g] .- ob[g]) .^ 2)), sigdigits = 4),
                          round(σo, sigdigits = 4)), ","))
    end
    qdir = joinpath(figure_path, "skill"); mkpath(qdir)
    csvname = joinpath(qdir, "$(moor)_skill.csv")
    open(csvname, "w") do io
        # murphy_ss = r2 - ss_cond_bias - ss_uncond_bias; sigma_ratio = σ_model / σ_obs; sigma_obs in the variable's unit
        println(io, "filter,var,depth_m,model,N,r,r2,willmott_d,murphy_ss,ss_cond_bias,ss_uncond_bias,sigma_ratio," *
                    "bias_model_minus_obs,rmse,sigma_obs")
        foreach(r -> println(io, r), rows)
    end
    println("saved ", csvname)

    metric_label = Dict("d" => "Willmott skill d", "r2" => "r²", "ss" => "Murphy skill score SS")
    # SS can be < 0 (worse than the mooring mean): diverging colours, 0 = white, clipped at -1
    metric_cmap  = Dict("d" => :viridis, "r2" => :viridis, "ss" => :balance)
    metric_range = Dict("d" => (0, 1), "r2" => (0, 1), "ss" => (-1, 1))
    textcolor(metric, v) = metric == "ss" ? (abs(v) > 0.6 ? :white : :black) : (v > 0.6 ? :black : :white)
    # column groups, separated by a white line: raw | low-passed | energy
    col_group(t, v) = v in energy_vars ? 3 : t == "raw" ? 1 : 2
    # cols: indices into skill_cols to draw; suffix: added to the file name
    function plot_skill(metric, cols; suffix = "")
        C = skill_cols[cols]
        col_labels = ["$(v)\n$(t == "raw" ? "raw" : t)" for (t, v) in C]
        grp = [col_group(t, v) for (t, v) in C]
        seps = [c + 0.5 for c in 1:length(C)-1 if grp[c] != grp[c+1]]
        fig = Figure(; figkw(18, 5.0 * length(kd) + 2)..., figure_padding = (4, 8, 4, 4))
        local hm
        for (k, dtitle) in enumerate(depth_titles)
            S = skill[metric][k][:, cols]
            ax = Axis(fig[k, 1]; title = dtitle, titlealign = :left, titlegap = 2, titlesize = 9,
                      xticks = (1:length(C), k == length(kd) ? col_labels : fill("", length(C))),
                      yticks = (1:length(skill_models), [src_style[m].short for m in skill_models]),
                      xticklabelsize = 7, yticklabelsize = 8, yreversed = true)
            hm = heatmap!(ax, 1:length(C), 1:length(skill_models), S'; colormap = metric_cmap[metric],
                          colorrange = metric_range[metric], lowclip = :navy, nan_color = :grey90)
            for i in axes(S, 1), c in axes(S, 2)
                isnan(S[i, c]) && continue
                text!(ax, c, i; text = string(round(S[i, c], digits = 2)), align = (:center, :center), fontsize = 7,
                      color = textcolor(metric, S[i, c]))
            end
            vlines!(ax, seps; color = :white, linewidth = 2)   # raw | low-passed | energy
        end
        Colorbar(fig[1:length(kd), 2], hm; label = metric_label[metric])   # SS < -1: lowclip colour, value written
        Label(fig[0, :], "$(metric_label[metric]) of each model against the mooring, $site\n" *
              "raw: 6 h bins, filtered: 1 day bins; grey = no data", fontsize = 10, tellwidth = false)
        rowgap!(fig.layout, 5); colgap!(fig.layout, 6)
        fname = joinpath(qdir, "$(moor)_skill_$(metric)$(suffix).png")
        figsave(fname, fig)
        println("saved ", fname)
        return fig
    end

    no_uv = findall(((t, v),) -> !(v in ("u", "v")), skill_cols)   # temp, salt, speed, EKE, KE
    for metric in ("d", "r2", "ss")
        plot_skill(metric, eachindex(skill_cols))              # every variable
        plot_skill(metric, no_uv; suffix = "_noUV")            # without u, v
    end
end
