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

sources = [
    Source("roms_ncom",  "ROMS 900m (NCOM parent)",  "/home/hsinyi/roms_data/mooring_ext_model/full_3mon/mooring_merged/<moor>.nc",
           "/home/hsinyi/roms_data/grid/roms_grd_900m.nc", "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"),
    Source("roms_cmems", "ROMS 2700m (CMEMS parent)", "/home/hsinyi/roms_data/mooring_ext_model/cmems2700/mooring_merged/<moor>.nc",
           "/home/hsinyi/roms_data/grid/roms_cmems_grd_2700m.nc", "/home/hsinyi/roms_data/grid/roms_cmems_grd_2700m_mor_edata.nc"),
    Source("ncom900",    "NCOM 900m",                "/home/hsinyi/roms_data/NCOM_mooring/NCOM_<moor>.nc"),
    Source("ncom2700",   "NCOM 2700m",               "/home/hsinyi/roms_data/NCOM_mooring_2700/NCOM_<moor>.nc"),
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

zg    = collect(-300.0:5.0:0.0)
tlim  = (DateTime(2022, 8, 24), DateTime(2022, 12, 1))   # records outside this period are dropped
t_ref = DateTime(1994, 1, 1)                             # ROMS ocean_time is in seconds since t_ref

intime(t) = findall(tlim[1] .<= t .<= tlim[2])
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