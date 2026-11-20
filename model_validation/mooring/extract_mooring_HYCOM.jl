# extract_mooring_HYCOM.jl
#
# Extract (interpolate) HYCOM output at the mooring locations.
#
# Data: the NetCDF HYCOM set in datadir (HYCOM_DATA_NC_2700m_1104; the same data as the
# binary set in /home/mbui/ModelOutput/NCOM/HYCOM_data), one group of files per daily
# 12Z forecast cycle:
#   <datadir>/hycom_<yyyymmddHH>_grd.nc   Longitude, Latitude (eta, xi), depth(z)
#   <datadir>/hycom_<yyyymmddHH>_ssh.nc   ssh(time, eta, xi)
#   <datadir>/hycom_<yyyymmddHH>_ts.nc    layer_temperature, layer_salinity (time, z, eta, xi)
#   <datadir>/hycom_<yyyymmddHH>_uv.nc    u_velocity, v_velocity (time, z, eta, xi)
# MT(time) = days since 1900-12-31 00:00 (HYCOM model time). Each cycle has 3-hourly
# records tau = 0..24 h, so tau 24 of a cycle is the same time as tau 0 of the next
# one: for a duplicated time the record of the NEWER cycle (the analysis) is kept.
# Land / below the seafloor is NaN.
# The script uses EVERY cycle in datadir that has all four files (an incomplete
# one is skipped with a warning). When more files are added, just run it again.
#
# Besides the moorings, the transect along the mooring line (straight line
# M1 -> CPIES7, through M2, CPIES2, CPIES3, M3, CPIES4..6) is extracted every
# dx_km, on the same points as in the NCOM / GLOBAL scripts.
#
# The grid is a regular lon/lat grid (0.04° lon x 0.02° lat) with 40 fixed z-levels, so (as in
# extract_mooring_GLOBAL.jl) the horizontal interpolation is 4 bilinear weights
# on the 4 surrounding columns. Only one small lon/lat box around all the
# moorings and the transect is read from the files. A corner that is
# land / below the seafloor is dropped and the weights of the wet corners are
# rescaled; a level is NaN only if all 4 are dry.
#
# Output (outdir):
#   HYCOM_<name>.nc     one file per mooring, same layout as GLOBAL_<name>.nc /
#                       NCOM_<name>.nc: variables (z, time), z negative down and
#                       ASCENDING (bottom -> surface), on the 40 native HYCOM levels, plus ssh(time)
#   HYCOM_M1_CPIES7.nc  the transect: variables (z, x, time) + u_along/u_cross, ssh (x, time)
#
# Usage:
#   julia extract_mooring_HYCOM.jl              -> all moorings + the transect
#   julia extract_mooring_HYCOM.jl M2 french    -> only those (name or index)
#   julia extract_mooring_HYCOM.jl transect     -> only the transect

using NCDatasets, Dates
include("/home/hsinyi/Documents/Julia/function/mooring_transect.jl")

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
datadir      = "/home/hsinyi/roms_data/HYCOM_DATA_NC_2700m_1104/"   # hycom_<yyyymmddHH>_{grd,ssh,ts,uv}.nc
outdir       = "/home/hsinyi/roms_data/HYCOM_mooring_NC/"           # one file per mooring: HYCOM_<name>.nc
mooring_file = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"

transect_ends = ("M1", "CPIES7")   # mooring line transect, from -> to
dx_km         = 1.0                # transect point spacing (same in all the model scripts)

# ---------------------------------------------------------------------------
# Mooring list (same as mooring_NCOM_extraction.jl), from the global attributes, e.g.
#   :M1_mor_info = "indices for M1_mor in roms_grd_900m.nc , location:(-45.4802;2.5415)"
# ---------------------------------------------------------------------------
moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
tr = transect_points(moor_name, moor_lon, moor_lat, transect_ends...; dx_km = dx_km)

function find_mooring(a, names)
    k = something(tryparse(Int, a), findfirst(==(a), names), 0)
    1 <= k <= length(names) || error("Unknown mooring '$a'; choose 1-$(length(names)), one of: $(join(names, ", ")), or transect")
    return k
end
do_transect = isempty(ARGS) || "transect" in ARGS
moor_sel = isempty(ARGS) ? collect(eachindex(moor_name)) :
           [find_mooring(a, moor_name) for a in ARGS if a != "transect"]
println("Moorings in this run : ", join(moor_name[moor_sel], ", "),
        do_transect ? "  + transect $(tr.name) ($(length(tr.lon)) points)" : "")

# ---------------------------------------------------------------------------
# Forecast cycles available: group the files by "<yyyymmddHH>" and keep the
# ones that have all four files
# ---------------------------------------------------------------------------
kinds = ("grd", "ssh", "ts", "uv")
file_re = r"^hycom_(\d{10})_(grd|ssh|ts|uv)\.nc$"
snap = Dict{String,Dict{String,String}}()
for f in readdir(datadir)
    m = match(file_re, f)
    m === nothing && continue
    get!(snap, m[1], Dict{String,String}())[m[2]] = joinpath(datadir, f)
end
for (tag, fs) in snap
    missing_kinds = [k for k in kinds if !haskey(fs, k)]
    isempty(missing_kinds) || (@warn "$tag: no $(join(missing_kinds, ", ")) file, skipped"; delete!(snap, tag))
end
tags = sort(collect(keys(snap)))
isempty(tags) && error("no complete HYCOM cycles in $datadir")
println("$(length(tags)) forecast cycles: $(first(tags)) .. $(last(tags))")

# grid: Longitude/Latitude are stored 2D (xi, eta) but the grid is regular; lon in 0..360
# (NOT periodic), lat ascending, 40 z-levels (positive down)
function read_grid(f)
    NCDataset(f) do ds
        LON = Float64.(ds["Longitude"][:, :]); LAT = Float64.(ds["Latitude"][:, :])
        lon, lat = LON[:, 1], LAT[1, :]
        @assert all(LON .== lon) && all(LAT .== lat') "$f: grid is not a regular lon/lat grid"
        lon, lat, Float64.(ds["depth"][:])
    end
end
lon, lat, dep = read_grid(snap[tags[1]]["grd"])
Nz = length(dep)

# ---------------------------------------------------------------------------
# Bilinear weights
# ---------------------------------------------------------------------------
"""
    bilinear_corners(lon, lat, lon0, lat0) -> (corners, w)

The 4 (i, j) grid points around (lon0, lat0) and their bilinear weights.
`lon` is the regional 0..360 HYCOM axis (lon0 may be -180..180), `lat` ascending.
"""
function bilinear_corners(lon, lat, lon0, lat0)
    lat[1] <= lat0 <= lat[end] || error("lat $lat0 is outside the downloaded box: $(lat[1])..$(lat[end])")
    x = mod(lon0, 360)
    lon[1] <= x <= lon[end] || error("lon $lon0 ($x) is outside the downloaded box: $(lon[1])..$(lon[end])")
    i  = clamp(searchsortedlast(lon, x), 1, length(lon) - 1)
    i2 = i + 1
    dlon = lon[i2] - lon[i]
    j = clamp(searchsortedlast(lat, lat0), 1, length(lat) - 1)
    a = (x - lon[i]) / dlon
    b = (lat0 - lat[j]) / (lat[j+1] - lat[j])
    corners = [(i, j), (i2, j), (i, j + 1), (i2, j + 1)]
    w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
    return corners, w
end

"""
    wet_average(cols, w) -> v

Weighted average of the corner values `cols` (one column per corner), ignoring
NaN corners and rescaling the remaining weights; NaN if all corners are NaN.
"""
function wet_average(cols, w)
    v = fill(NaN, size(cols, 1))
    for k in axes(cols, 1)
        s, ws = 0.0, 0.0
        for c in axes(cols, 2)
            isnan(cols[k, c]) && continue
            s += w[c] * cols[k, c]; ws += w[c]
        end
        ws > 0 && (v[k] = s / ws)
    end
    return v
end

moor = map(moor_sel) do k
    corners, w = bilinear_corners(lon, lat, moor_lon[k], moor_lat[k])
    (name = moor_name[k], lon = moor_lon[k], lat = moor_lat[k], corners = corners, w = w)
end
for m in moor
    println("$(m.name): corners (i,j) = ", m.corners, ", weights = ", round.(m.w, digits = 3))
end
tr_cw = do_transect ? [bilinear_corners(lon, lat, x, y) for (x, y) in zip(tr.lon, tr.lat)] : []

# the box (i0:i1, j0:j1) read from the files: every corner of every point
all_corners = vcat([m.corners for m in moor]..., first.(tr_cw)...)
i0, i1 = extrema(first.(all_corners))
j0, j1 = extrema(last.(all_corners))
println("reading box i=$i0:$i1, j=$j0:$j1 ($(i1-i0+1) x $(j1-j0+1) columns)")

# read the box of a 3D variable -> (i, j, depth), or of a 2D one -> (i, j, 1); missing -> NaN
# (in Julia the dimensions are (xi, eta, z, time) = (lon, lat, depth, time))
read_box(var, t) = ndims(var) == 4 ? Float64.(coalesce.(var[i0:i1, j0:j1, :, t], NaN)) :
                                     reshape(Float64.(coalesce.(var[i0:i1, j0:j1, t], NaN)), i1 - i0 + 1, j1 - j0 + 1, 1)

# value at one point from the box: (depth,) or (1,)
interp_point(B, corners, w) = wet_average(stack([B[i - i0 + 1, j - j0 + 1, :] for (i, j) in corners]), w)

# ---------------------------------------------------------------------------
# Time loop: every cycle and every record (tau = 0, 3, .., 24 h) in it
# ---------------------------------------------------------------------------
# (output name, file kind, variable name in the file)
varlist = [("temp", "ts", "layer_temperature"), ("salt", "ts", "layer_salinity"),
           ("u", "uv", "u_velocity"), ("v", "uv", "v_velocity"), ("ssh", "ssh", "ssh")]
mt2datetime(mt) = DateTime(1900, 12, 31) + Millisecond(round(Int64, mt * 86_400_000))
time_out = DateTime[]
out = Dict(m.name => Dict(v => Vector{Float64}[] for (v, _, _) in varlist) for m in moor)
tr_out = Dict(v => Matrix{Float64}[] for (v, _, _) in varlist)    # (depth, x) per record

for (n, tag) in enumerate(tags)
    fs = snap[tag]
    glon, glat, gdep = read_grid(fs["grd"])
    @assert glon == lon && glat == lat && gdep == dep "$tag: different grid"
    dss = Dict(k => NCDataset(fs[k]) for k in ("ssh", "ts", "uv"))
    try
        mt = dss["ts"]["MT"][:]
        for k in ("ssh", "uv")
            @assert dss[k]["MT"][:] == mt "$tag: $k has a different time"
        end
        for t in eachindex(mt)
            push!(time_out, mt2datetime(mt[t]))
            for (v, k, ncvar) in varlist
                B = read_box(dss[k][ncvar], t)
                for m in moor
                    push!(out[m.name][v], interp_point(B, m.corners, m.w))
                end
                do_transect && push!(tr_out[v], stack([interp_point(B, c, w) for (c, w) in tr_cw]))
            end
        end
    finally
        foreach(close, values(dss))
    end
    (n % 10 == 0 || n == length(tags)) && println("done $tag  ($n/$(length(tags)) cycles)")
end

# sort by valid time; tau 24 of a cycle = tau 0 of the next one: keep the record
# read LAST (newer cycle). sortperm is stable, so reverse it, unique, reverse back.
it = reverse(unique(i -> time_out[i], reverse(sortperm(time_out))))
println("$(length(time_out) - length(it)) duplicate times (tau 24 = next tau 0) dropped, kept the newer cycle")
time_out = time_out[it]
dt = unique(diff(time_out))
dt == [Hour(3)] || @warn "time step is not always 3 h: $(unique(dt)) (missing cycles?)"

# ---------------------------------------------------------------------------
# Save, like GLOBAL_<name>.nc: (z, time), z negative down and ASCENDING
# ---------------------------------------------------------------------------
mkpath(outdir)
for m in moor
    o = out[m.name]
    to_zt(x) = reverse(stack(x)[:, it], dims = 1)        # (depth, time) surface-first -> bottom-first
    fname = joinpath(outdir, "HYCOM_$(m.name).nc")
    NCDataset(fname, "c") do ds
        ds.attrib["title"]   = "HYCOM ($(basename(rstrip(datadir, '/')))) interpolated to mooring $(m.name)"
        ds.attrib["station"] = m.name
        ds.attrib["source"]  = "extract_mooring_HYCOM.jl, data $datadir"
        ds.attrib["note"]    = "3-hourly HYCOM, daily 12Z cycles tau 0-21 h (tau 24 replaced by next cycle tau 0); a missing time is skipped (no gap filling)"
        defDim(ds, "z", Nz); defDim(ds, "time", length(time_out))

        defVar(ds, "lon", m.lon, (), attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", m.lat, (), attrib = ["units" => "degrees_north"])
        defVar(ds, "z", reverse(-dep), ("z",), attrib = ["units" => "m", "positive" => "up"])
        defVar(ds, "time", time_out, ("time",), attrib = ["units" => "hours since 2022-01-01 00:00:00"])

        dims3 = ("z", "time")
        defVar(ds, "temp", to_zt(o["temp"]), dims3, attrib = ["long_name" => "temperature", "units" => "degC"])
        defVar(ds, "salt", to_zt(o["salt"]), dims3, attrib = ["long_name" => "salinity", "units" => "psu"])
        defVar(ds, "u", to_zt(o["u"]), dims3, attrib = ["long_name" => "eastward velocity", "units" => "m/s"])
        defVar(ds, "v", to_zt(o["v"]), dims3, attrib = ["long_name" => "northward velocity", "units" => "m/s"])
        defVar(ds, "ssh", vec(stack(o["ssh"]))[it], ("time",), attrib = ["long_name" => "sea surface height", "units" => "m"])
    end
    println("$(m.name): saved $fname  ($(time_out[1]) to $(time_out[end]), $(length(time_out)) records)")
end

# ---------------------------------------------------------------------------
# Transect: (z, x, time), z negative down and ASCENDING, like the mooring files
# ---------------------------------------------------------------------------
if do_transect
    to_zxt(x) = reverse(stack(x)[:, :, it], dims = 1)
    longname = Dict("temp" => ("temperature", "degC"), "salt" => ("salinity", "psu"),
                    "u" => ("eastward velocity", "m/s"), "v" => ("northward velocity", "m/s"))
    save_transect(joinpath(outdir, "HYCOM_$(tr.name).nc"), tr, reverse(-dep), time_out,
                  [v => (to_zxt(tr_out[v]), longname[v]...) for v in ("temp", "salt", "u", "v")];
                  ssh    = stack(tr_out["ssh"])[1, :, it],
                  title  = "HYCOM ($(basename(rstrip(datadir, '/')))) interpolated to the $(tr.from) - $(tr.to) transect",
                  source = "extract_mooring_HYCOM.jl, data $datadir",
                  note   = "3-hourly HYCOM, daily 12Z cycles tau 0-21 h (tau 24 replaced by next cycle tau 0); a missing time is skipped (no gap filling)")
end
