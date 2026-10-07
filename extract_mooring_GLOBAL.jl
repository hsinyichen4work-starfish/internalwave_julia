using NCDatasets, CairoMakie, Dates, Statistics, LinearAlgebra
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")
##

GLOBALtemp    = "/home/hsinyi/data_notm/GLOBAL_ANALYSISFORECAST_temp.nc"
GLOBALvel    = "/home/hsinyi/data_notm/GLOBAL_ANALYSISFORECAST_current.nc"
outdir       = "/home/hsinyi/roms_data/GLOBAL_mooring/"     # one file per mooring: GLOBAL_<name>.nc

# days to read (inclusive, through 00:00 of the day after day_end);
# the full period is Date(2022, 8, 22) to Date(2022, 11, 30)
day_start = Date(2022, 8, 24)
day_end   = Date(2022, 11, 29)

mor_name = "french"
French_mor_loc = [-45.13 , 3.95] #the French mooring location is  3.95°N 45.13°W.
##

# fields come back as (lon, lat, depth, time); depth is positive down, k=1 at the surface
time_gl, dep, lat, lon, temp = NCDataset(GLOBALtemp) do ds
    show(ds)
    t_all = ds["time"][:]
    it = findfirst(>=(DateTime(day_start)), t_all):findlast(<=(DateTime(day_end) + Day(1)), t_all)
    t_all[it], ds["depth"][:], ds["latitude"][:], ds["longitude"][:], nomissing(ds["thetao"][:,:,:,it], NaN)
end
time_uv, u, v = NCDataset(GLOBALvel) do ds
    show(ds)
    t_all = ds["time"][:]
    it = findfirst(>=(DateTime(day_start)), t_all):findlast(<=(DateTime(day_end) + Day(1)), t_all)
    @assert ds["longitude"][:] == lon && ds["latitude"][:] == lat && ds["depth"][:] == dep "temp and current files are on different grids"
    t_all[it], nomissing(ds["uo"][:,:,:,it], NaN), nomissing(ds["vo"][:,:,:,it], NaN)
end
@assert time_uv == time_gl "temp and current files have different time axes"
##

# -----------------------------------------------------------------------
# Horizontal interpolation to the mooring.
# The GLOBAL grid is a regular lon/lat grid with depth levels that do not
# change in time, so the operator is just 4 bilinear weights on the 4
# surrounding columns; it is built once and reused for every variable,
# depth and time.
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
replaced by the value at the mooring, e.g. (lon, lat, depth, time) -> (depth, time).
A level that is NaN (land / below the seafloor) in any of the 4 columns stays NaN.
"""
function apply_coef(F, idx, w)
    rest = size(F)[3:end]
    Fc = reshape(F, size(F, 1) * size(F, 2), rest...)
    return reshape(sum(w .* Fc[idx, ntuple(_ -> :, length(rest))...]; dims=1), rest...)
end

idx, w = bilinear_coef(Float64.(lon), Float64.(lat), French_mor_loc...)
println("$mor_name: corners (i,j) = ", Tuple.(CartesianIndices((length(lon), length(lat)))[idx]),
        ", weights = ", round.(w, digits=3))

temp_mor = apply_coef(temp, idx, w)     # (depth, time)
u_mor    = apply_coef(u, idx, w)
v_mor    = apply_coef(v, idx, w)
##

# save like NCOM_<name>.nc: variables (z, time), z negative down and ASCENDING (bottom -> surface)
mkpath(outdir)
fname = joinpath(outdir, "GLOBAL_$(mor_name).nc")
NCDataset(fname, "c") do ds
    ds.attrib["title"]   = "GLOBAL_ANALYSISFORECAST_PHY_001_024 interpolated to mooring $mor_name"
    ds.attrib["station"] = mor_name
    ds.attrib["source"]  = "extract_mooring_GLOBAL.jl, data $GLOBALtemp, $GLOBALvel"
    defDim(ds, "z", length(dep)); defDim(ds, "time", length(time_gl))

    defVar(ds, "lon", French_mor_loc[1], (), attrib=["units" => "degrees_east"])
    defVar(ds, "lat", French_mor_loc[2], (), attrib=["units" => "degrees_north"])
    defVar(ds, "z", reverse(-Float64.(dep)), ("z",), attrib=["units" => "m", "positive" => "up"])
    defVar(ds, "time", time_gl, ("time",), attrib=["units" => "hours since 2022-01-01 00:00:00"])

    dims3 = ("z", "time")
    defVar(ds, "temp", reverse(temp_mor, dims=1), dims3, attrib=["long_name" => "potential temperature", "units" => "degC"])
    defVar(ds, "u", reverse(u_mor, dims=1), dims3, attrib=["long_name" => "eastward velocity", "units" => "m/s"])
    defVar(ds, "v", reverse(v_mor, dims=1), dims3, attrib=["long_name" => "northward velocity", "units" => "m/s"])
end
println("$mor_name: saved $fname  ($(time_gl[1]) to $(time_gl[end]), size(temp) = $(size(temp_mor)))")
