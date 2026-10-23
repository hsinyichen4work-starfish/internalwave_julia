# extract_mooring_GLOBAL.jl
#
# Extract (interpolate) CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024 (1/12°) at
# every mooring and along the mooring line transect (straight line M1 -> CPIES7,
# same points as in mooring_NCOM_extraction.jl / extract_mooring_HYCOM.jl).
#
# Data: the "mooringcover" files (lon -45.58..-43.67, lat 2.5..5.0, which covers
# every mooring), 6-hourly instantaneous temp, salt, u, v on 30 z-levels
# (0.5-380 m), plus hourly sea level from cmems_ssh_mooringcover.nc.
#
# The GLOBAL grid is a regular lon/lat grid with depth levels that do not change
# in time, so the operator is just 4 bilinear weights on the 4 surrounding
# columns, built once per point and reused for every variable, depth and time.
#
# Output (outdir):
#   GLOBAL_<name>.nc        one file per mooring: temp, salt, u, v (z, time), z negative
#                           down and ASCENDING (bottom -> surface), like NCOM_<name>.nc;
#                           plus ssh (time_ssh), hourly, on its own time axis
#   GLOBAL_M1_CPIES7.nc     the transect: temp, salt, u, v, u_along, u_cross (z, x, time)
#   GLOBAL_M1_CPIES7_ssh.nc transect sea level: sea_surface_height, total_sea_level,
#                           ocean_tide (x, time), hourly

using NCDatasets, Dates
include("/home/hsinyi/Documents/Julia/function/mooring_transect.jl")
##

cmems_dir    = "/home/hsinyi/data_notm/"
cmems_temp   = "cmems_temp_mooringcover.nc"
cmems_salt   = "cmems_salt_mooringcover.nc"
cmems_cur    = "cmems_cur_mooringcover.nc"
cmems_ssh    = "cmems_ssh_mooringcover.nc"
outdir       = "/home/hsinyi/roms_data/GLOBAL_mooring/"
mooring_file = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"

transect_ends = ("M1", "CPIES7")   # mooring line transect, from -> to
dx_km         = 1.0                # transect point spacing (same in all the model scripts)

ssh_vars = ["sea_surface_height", "total_sea_level", "ocean_tide"]   # in cmems_ssh
##

"""
    read_cmems(fname, vars) -> (lon, lat, dep, time, fields)

Read a CMEMS file; every field in `fields` is (lon, lat, depth, time), NaN on land.
"""
function read_cmems(fname, vars)
    NCDataset(joinpath(cmems_dir, fname)) do ds
        Float64.(ds["longitude"][:]), Float64.(ds["latitude"][:]), Float64.(ds["depth"][:]), ds["time"][:],
        [nomissing(ds[v][:, :, :, :], NaN) for v in vars]
    end
end

# fields are (lon, lat, depth, time); depth is positive down, k=1 at the surface
lon, lat, dep, time_gl, (temp,) = read_cmems(cmems_temp, ["thetao"])
lon_s, lat_s, dep_s, time_s, (salt,) = read_cmems(cmems_salt, ["so"])
lon_u, lat_u, dep_u, time_u, (u, v) = read_cmems(cmems_cur, ["uo", "vo"])
@assert lon_s == lon_u == lon && lat_s == lat_u == lat && dep_s == dep_u == dep "temp, salt and current files are on different grids"
@assert time_s == time_u == time_gl "temp, salt and current files have different time axes"
println("CMEMS: $(length(lon)) x $(length(lat)) x $(length(dep)) levels ($(dep[1])-$(dep[end]) m), ",
        "$(length(time_gl)) records, $(time_gl[1]) to $(time_gl[end])")

# sea level: hourly, on its own (slightly shifted) grid
lon_h, lat_h, _, time_h, ssh_fields = read_cmems(cmems_ssh, ssh_vars)
ssh_long_name = NCDataset(ds -> Dict(v => ds[v].attrib["long_name"] for v in ssh_vars), joinpath(cmems_dir, cmems_ssh))
##

# -----------------------------------------------------------------------
# Horizontal interpolation
# -----------------------------------------------------------------------
"""
    bilinear_coef(lon, lat, lon0, lat0) -> (idx, w)

Bilinear weights `w` and linear indices `idx` (into a (lon, lat) field) of the
4 grid points around (lon0, lat0). `lon` and `lat` are ascending 1D axes.
"""
function bilinear_coef(lon, lat, lon0, lat0)
    (lon[1] <= lon0 <= lon[end] && lat[1] <= lat0 <= lat[end]) ||
        error("($lon0, $lat0) is outside the grid: lon $(lon[1])..$(lon[end]), lat $(lat[1])..$(lat[end])")
    i = clamp(searchsortedlast(lon, lon0), 1, length(lon) - 1)
    j = clamp(searchsortedlast(lat, lat0), 1, length(lat) - 1)
    a = (lon0 - lon[i]) / (lon[i+1] - lon[i])
    b = (lat0 - lat[j]) / (lat[j+1] - lat[j])
    L = LinearIndices((length(lon), length(lat)))
    idx = [L[i, j], L[i+1, j], L[i, j+1], L[i+1, j+1]]
    w   = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
    return idx, w
end

"""
    apply_coef(F, idx, w) -> Fm

Apply the weights to F(lon, lat, ...): the two horizontal dimensions are
replaced by the value at the point, e.g. (lon, lat, depth, time) -> (depth, time).
A level that is NaN (land / below the seafloor) in any of the 4 columns stays NaN.
"""
function apply_coef(F, idx, w)
    rest = size(F)[3:end]
    Fc = reshape(F, size(F, 1) * size(F, 2), rest...)
    return reshape(sum(w .* Fc[idx, ntuple(_ -> :, length(rest))...]; dims=1), rest...)
end

# (depth, time) surface-first -> (z, time) bottom-first, like NCOM_<name>.nc
zt(F) = reverse(F, dims=1)
z = reverse(-dep)
##

# -----------------------------------------------------------------------
# Moorings: one file each
# -----------------------------------------------------------------------
moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
mkpath(outdir)
for (name, mlon, mlat) in zip(moor_name, moor_lon, moor_lat)
    idx, w = bilinear_coef(lon, lat, mlon, mlat)
    idx_h, w_h = bilinear_coef(lon_h, lat_h, mlon, mlat)
    println("$name: corners (i,j) = ", Tuple.(CartesianIndices((length(lon), length(lat)))[idx]),
            ", weights = ", round.(w, digits=3))

    fname = joinpath(outdir, "GLOBAL_$(name).nc")
    NCDataset(fname, "c") do ds
        ds.attrib["title"]   = "GLOBAL_ANALYSISFORECAST_PHY_001_024 interpolated to mooring $name"
        ds.attrib["station"] = name
        ds.attrib["source"]  = "extract_mooring_GLOBAL.jl, data $(cmems_dir){$cmems_temp, $cmems_salt, $cmems_cur, $cmems_ssh}"
        ds.attrib["note"]    = "temp/salt/u/v: instantaneous 6-hourly snapshots; ssh: hourly, on time_ssh"
        defDim(ds, "z", length(z)); defDim(ds, "time", length(time_gl)); defDim(ds, "time_ssh", length(time_h))

        defVar(ds, "lon", mlon, (), attrib=["units" => "degrees_east"])
        defVar(ds, "lat", mlat, (), attrib=["units" => "degrees_north"])
        defVar(ds, "z", z, ("z",), attrib=["units" => "m", "positive" => "up"])
        defVar(ds, "time", time_gl, ("time",), attrib=["units" => "hours since 2022-01-01 00:00:00"])
        defVar(ds, "time_ssh", time_h, ("time_ssh",), attrib=["units" => "hours since 2022-01-01 00:00:00"])

        dims3 = ("z", "time")
        defVar(ds, "temp", zt(apply_coef(temp, idx, w)), dims3, attrib=["long_name" => "potential temperature", "units" => "degC"])
        defVar(ds, "salt", zt(apply_coef(salt, idx, w)), dims3, attrib=["long_name" => "salinity", "units" => "psu"])
        defVar(ds, "u", zt(apply_coef(u, idx, w)), dims3, attrib=["long_name" => "eastward velocity", "units" => "m/s"])
        defVar(ds, "v", zt(apply_coef(v, idx, w)), dims3, attrib=["long_name" => "northward velocity", "units" => "m/s"])
        # ssh: sea_surface_height (no tides), plus total_sea_level and ocean_tide
        for (vname, F) in zip(ssh_vars, ssh_fields)
            defVar(ds, vname == "sea_surface_height" ? "ssh" : vname, vec(apply_coef(F, idx_h, w_h)), ("time_ssh",),
                   attrib=["long_name" => ssh_long_name[vname], "units" => "m"])
        end
    end
    println("$name: saved $fname")
end
##

# -----------------------------------------------------------------------
# Transect along the mooring line
# -----------------------------------------------------------------------
tr = transect_points(moor_name, moor_lon, moor_lat, transect_ends...; dx_km = dx_km)
println("transect $(tr.name): $(length(tr.lon)) points, moorings on it: ", join(tr.moor, ", "))

"""
    on_transect(lon, lat, F) -> Ft

Field F(lon, lat, depth, time) at every transect point -> (depth, x, time).
"""
function on_transect(lon, lat, F)
    tr_w = [bilinear_coef(lon, lat, x, y) for (x, y) in zip(tr.lon, tr.lat)]
    return permutedims(stack([apply_coef(F, idx, w) for (idx, w) in tr_w]), (1, 3, 2))
end

save_transect(joinpath(outdir, "GLOBAL_$(tr.name).nc"), tr, z, time_gl,
              ["temp" => (zt(on_transect(lon, lat, temp)), "potential temperature", "degC"),
               "salt" => (zt(on_transect(lon, lat, salt)), "salinity", "psu"),
               "u"    => (zt(on_transect(lon, lat, u)), "eastward velocity", "m/s"),
               "v"    => (zt(on_transect(lon, lat, v)), "northward velocity", "m/s")];
              title  = "GLOBAL_ANALYSISFORECAST_PHY_001_024 interpolated to the $(tr.from) - $(tr.to) transect",
              source = "extract_mooring_GLOBAL.jl, data $(cmems_dir){$cmems_temp, $cmems_salt, $cmems_cur}",
              note   = "instantaneous 6-hourly snapshots")

save_transect(joinpath(outdir, "GLOBAL_$(tr.name)_ssh.nc"), tr, nothing, time_h, [];
              vars2d = [vname => (on_transect(lon_h, lat_h, F)[1, :, :], ssh_long_name[vname], "m")
                        for (vname, F) in zip(ssh_vars, ssh_fields)],
              title  = "GLOBAL_ANALYSISFORECAST_PHY_001_024 sea level interpolated to the $(tr.from) - $(tr.to) transect",
              source = "extract_mooring_GLOBAL.jl, data $(cmems_dir)$(cmems_ssh)",
              note   = "instantaneous hourly snapshots")
