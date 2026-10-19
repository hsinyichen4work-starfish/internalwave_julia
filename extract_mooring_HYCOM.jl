# extract_mooring_HYCOM.jl
#
# Extract (interpolate) GOFS 3.5 HYCOM 1/25° output at the mooring locations.
#
# The HYCOM files come as snapshots with irregular spacing in time
# (gofs35_hycom_glbz04_<yyyymmddhh>_t<tau>_{ssh,ts3z,uv3z}.nc, one record each),
# so instead of a date range the script uses EVERY snapshot found in datadir that
# has all three files (ssh, ts3z, uv3z). When more files are downloaded,
# just run it again: the mooring files are rewritten with all times, sorted.
#
# Besides the moorings, the transect along the mooring line (straight line
# M1 -> CPIES7, through M2, CPIES2, CPIES3, M3, CPIES4..6) is extracted every
# dx_km, on the same points as in the NCOM / GLOBAL scripts.
#
# HYCOM is a regular lon/lat grid with fixed z-levels, so (as in
# extract_mooring_GLOBAL.jl) the horizontal interpolation is 4 bilinear weights
# on the 4 surrounding columns. Only one small lon/lat box around all the
# moorings and the transect is read from the ~12 GB 3D files. A corner that is
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
datadir      = "/home/mbui/ModelOutput/NCOM/HYCOM_data/"
outdir       = "/home/hsinyi/roms_data/HYCOM_mooring/"     # one file per mooring: HYCOM_<name>.nc
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
# Snapshots available: group the files by "<yyyymmddhh>_t<tau>" and keep the
# ones that have all three kinds
# ---------------------------------------------------------------------------
kinds = ("ssh", "ts3z", "uv3z")
file_re = r"^gofs35_hycom_glbz04_(\d{10}_t\d{4})_(ssh|ts3z|uv3z)\.nc$"
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
isempty(tags) && error("no complete HYCOM snapshots in $datadir")
println("$(length(tags)) snapshots: ", join(tags, ", "))

# grid (same for every file): lon 0..360 (periodic), lat ascending, 40 z-levels (positive down)
lon, lat, dep = NCDataset(snap[tags[1]]["ts3z"]) do ds
    Float64.(ds["lon"][:]), Float64.(ds["lat"][:]), Float64.(ds["depth"][:])
end
Nz = length(dep)

# ---------------------------------------------------------------------------
# Bilinear weights
# ---------------------------------------------------------------------------
"""
    bilinear_corners(lon, lat, lon0, lat0) -> (corners, w)

The 4 (i, j) grid points around (lon0, lat0) and their bilinear weights.
`lon` is the periodic 0..360 HYCOM axis (lon0 may be -180..180), `lat` ascending.
"""
function bilinear_corners(lon, lat, lon0, lat0)
    lat[1] <= lat0 <= lat[end] || error("lat $lat0 is outside the grid: $(lat[1])..$(lat[end])")
    x = mod(lon0 - lon[1], 360) + lon[1]
    i  = searchsortedlast(lon, x)
    i2 = i == length(lon) ? 1 : i + 1                     # wrap across 360 -> 0
    dlon = mod(lon[i2] - lon[i], 360)
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
i1 - i0 < length(lon) ÷ 2 || error("points on both sides of lon 0/360; the box read is not set up for that")
println("reading box i=$i0:$i1, j=$j0:$j1 ($(i1-i0+1) x $(j1-j0+1) columns)")

# read the box of a 3D variable -> (i, j, depth), or of a 2D one -> (i, j, 1); missing -> NaN
read_box(var, t) = ndims(var) == 4 ? Float64.(coalesce.(var[i0:i1, j0:j1, :, t], NaN)) :
                                     reshape(Float64.(coalesce.(var[i0:i1, j0:j1, t], NaN)), i1 - i0 + 1, j1 - j0 + 1, 1)

# value at one point from the box: (depth,) or (1,)
interp_point(B, corners, w) = wet_average(stack([B[i - i0 + 1, j - j0 + 1, :] for (i, j) in corners]), w)

# ---------------------------------------------------------------------------
# Time loop: every snapshot (and every record in it, in case a file has more than one)
# ---------------------------------------------------------------------------
varlist = [("temp", "ts3z", "water_temp"), ("salt", "ts3z", "salinity"),
           ("u", "uv3z", "water_u"), ("v", "uv3z", "water_v"), ("ssh", "ssh", "surf_el")]
time_out = DateTime[]
out = Dict(m.name => Dict(v => Vector{Float64}[] for (v, _, _) in varlist) for m in moor)
tr_out = Dict(v => Matrix{Float64}[] for (v, _, _) in varlist)    # (depth, x) per record

for tag in tags
    fs = snap[tag]
    ds_ts, ds_uv, ds_ssh = NCDataset(fs["ts3z"]), NCDataset(fs["uv3z"]), NCDataset(fs["ssh"])
    try
        @assert Float64.(ds_uv["lon"][:]) == lon && Float64.(ds_uv["lat"][:]) == lat "$tag: uv3z on a different grid"
        t_all = ds_ts["time"][:]
        @assert ds_uv["time"][:] == t_all && ds_ssh["time"][:] == t_all "$tag: ts3z/uv3z/ssh have different times"
        for t in eachindex(t_all)
            push!(time_out, t_all[t])
            dsk = Dict("ts3z" => ds_ts, "uv3z" => ds_uv, "ssh" => ds_ssh)
            for (v, kind, ncvar) in varlist
                B = read_box(dsk[kind][ncvar], t)
                for m in moor
                    push!(out[m.name][v], interp_point(B, m.corners, m.w))
                end
                do_transect && push!(tr_out[v], stack([interp_point(B, c, w) for (c, w) in tr_cw]))
            end
        end
    finally
        close(ds_ts); close(ds_uv); close(ds_ssh)
    end
    println("done $tag  ($(length(time_out)) records)")
end

# snapshots are sorted by file name; sort by valid time too, and drop duplicates
# (e.g. the same valid time from two different analyses: the first one is kept)
it = unique(i -> time_out[i], sortperm(time_out))
length(it) < length(time_out) && @warn "$(length(time_out) - length(it)) duplicate times dropped"
time_out = time_out[it]

# ---------------------------------------------------------------------------
# Save, like GLOBAL_<name>.nc: (z, time), z negative down and ASCENDING
# ---------------------------------------------------------------------------
mkpath(outdir)
for m in moor
    o = out[m.name]
    to_zt(x) = reverse(stack(x)[:, it], dims = 1)        # (depth, time) surface-first -> bottom-first
    fname = joinpath(outdir, "HYCOM_$(m.name).nc")
    NCDataset(fname, "c") do ds
        ds.attrib["title"]   = "GOFS 3.5 HYCOM glbz04 interpolated to mooring $(m.name)"
        ds.attrib["station"] = m.name
        ds.attrib["source"]  = "extract_mooring_HYCOM.jl, data $datadir"
        ds.attrib["note"]    = "irregular time sampling: one record per available HYCOM snapshot"
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
                  title  = "GOFS 3.5 HYCOM glbz04 interpolated to the $(tr.from) - $(tr.to) transect",
                  source = "extract_mooring_HYCOM.jl, data $datadir",
                  note   = "irregular time sampling: one record per available HYCOM snapshot")
end
