# cmems_input_make.jl
#
# Builds the ROMS initial file and the daily boundary files for the 2700 m
# grid roms_cmems_grd_2700m.nc from CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024
# (downloaded by cmems_download.sh).
#
# The CMEMS fields are first written, one day at a time, into the same
# "parent" NetCDF layout make_bry_need_nc writes for NCOM
# (<par_name>_ssh/_ts/_uv.nc + one static cmems_lthick.nc), so the
# unchanged h2r_make_ini / h2r_bry_hv routines do the interpolation.
#   - CMEMS level depths d_k are turned into layer thicknesses that
#     ncom_zgrid maps back onto d_k exactly (dz_1 = 2 d_1,
#     dz_k = 2 (d_k - d_{k-1}) - dz_{k-1}; all positive for the 50 levels).
#   - T/S/u/v: 6-hourly instantaneous; zeta: hourly-mean zos at the same
#     hours. CMEMS has no tides.
#   - Each daily file holds 00, 06, 12, 18, 24 h (5 records), like the
#     NCOM daily files hold 00..24 h.
#
# Run on a compute node (the login node runs out of memory), e.g.
#     sbatch sbatch_files/cmems_input_job.sh

using Dates, NCDatasets

include("/home/hchen54/internalwave_julia/function/input_nc/load_input_nc.jl")
using .InputNC

## settings (same s-coordinate as the 900 m NCOM inputs)
dx = 2700
chd_thetas = 6; chd_thetab = 3; chd_hc = 250; chd_N = 128
chd_ang = "rad"; chdscoord = "new2008"
ndomx = 2; ndomy = 2
chdscd = (N = chd_N, theta_s = chd_thetas, theta_b = chd_thetab, hc = chd_hc, scoord = chdscoord)

dating = Date(2022, 8, 24):Day(1):Date(2022, 9, 30)   # one bry file per day
ini_time = DateTime(2022, 8, 24)
rec_hours = 0:6:24

## paths
cmems_path = "/expanse/lustre/projects/uso101/hchen54/CMEMS_DATA/"
out_root = "/expanse/lustre/projects/uso101/hchen54/input_cmems/"
chdgrd = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_cmems_grd_$(dx)m.nc"
parent_path = joinpath(out_root, "CMEMS_DATA_NC")   # intermediate parent files
initial_path = joinpath(out_root, "ini")
boundary_path = joinpath(out_root, "bry")
grid_path = joinpath(out_root, "grid")
foreach(mkpath, (parent_path, initial_path, boundary_path, grid_path))
remake_parent = false

# copy of the grid next to the inputs, so input_cmems is self-contained
isfile(joinpath(grid_path, basename(chdgrd))) || cp(chdgrd, joinpath(grid_path, basename(chdgrd)))

## CMEMS sources (Julia order: lon, lat, depth, time)
uv_file = joinpath(cmems_path, "cmems_uv_6h.nc")
t_file = joinpath(cmems_path, "cmems_temp_6h.nc")
s_file = joinpath(cmems_path, "cmems_salt_6h.nc")
zos_file = joinpath(cmems_path, "cmems_zos_1h.nc")

lon, lat, depth, t3d = NCDataset(t_file) do ds
    Float64.(ds["longitude"][:]), Float64.(ds["latitude"][:]), Float64.(ds["depth"][:]), ds["time"][:]
end
tzos = NCDataset(ds -> ds["time"][:], zos_file)
nlon, nlat, nz = length(lon), length(lat), length(depth)

# MT: days since 1900-12-31, as in the NCOM parent files
const MT0 = DateTime(1900, 12, 31)
mt(t::DateTime) = Dates.value(t - MT0) / 86_400_000

nan(x) = Float64.(coalesce.(x, NaN))

## static parent grid file: Longitude, Latitude, layer_thickness
par_grd = joinpath(parent_path, "cmems_lthick.nc")
if remake_parent || !isfile(par_grd)
    dz = similar(depth)
    dz[1] = 2depth[1]
    for k in 2:nz
        dz[k] = 2(depth[k] - depth[k-1]) - dz[k-1]
    end
    minimum(dz) > 0 || error("negative layer thickness from the CMEMS depths")
    NCDataset(par_grd, "c"; format = :netcdf4_classic) do ds
        defDim(ds, "xi", nlon); defDim(ds, "eta", nlat); defDim(ds, "z", nz); defDim(ds, "time", Inf)
        defVar(ds, "Longitude", Float64, ("xi", "eta"))[:, :] = [x for x in lon, _ in lat]
        defVar(ds, "Latitude", Float64, ("xi", "eta"))[:, :] = [y for _ in lon, y in lat]
        defVar(ds, "layer_thickness", Float64, ("xi", "eta", "z", "time"))[:, :, :, 1] =
            repeat(reshape(dz, 1, 1, nz), nlon, nlat, 1)
    end
    println("Wrote ", par_grd)
end

## daily parent files: <par_name>_ssh.nc, _ts.nc, _uv.nc
function make_cmems_day(par_name, times)
    files = (ssh = joinpath(parent_path, par_name * "_ssh.nc"),
             ts = joinpath(parent_path, par_name * "_ts.nc"),
             uv = joinpath(parent_path, par_name * "_uv.nc"))
    (!remake_parent && all(isfile, files)) && return files

    i3 = [findfirst(==(t), t3d) for t in times]
    iz = [findfirst(==(t), tzos) for t in times]
    any(isnothing, i3) && error("CMEMS 6-hourly files miss a time of $par_name")
    any(isnothing, iz) && error("CMEMS zos file misses a time of $par_name")
    MT = mt.(times)

    function create(file, vars, has_z)
        isfile(file) && rm(file)
        NCDataset(file, "c"; format = :netcdf4_classic) do ds
            defDim(ds, "xi", nlon); defDim(ds, "eta", nlat)
            has_z && defDim(ds, "z", nz)
            defDim(ds, "time", Inf)
            for v in vars
                defVar(ds, v, Float64, has_z ? ("xi", "eta", "z", "time") : ("xi", "eta", "time"))
            end
            defVar(ds, "MT", Float64, ("time",))[1:length(MT)] = MT
        end
    end

    create(files.ssh, ["ssh"], false)
    create(files.ts, ["layer_temperature", "layer_salinity"], true)
    create(files.uv, ["u_velocity", "v_velocity"], true)
    NCDataset(zos_file) do src
        NCDataset(files.ssh, "a") do ds
            for (j, k) in enumerate(iz)
                ds["ssh"][:, :, j] = nan(src["zos"][:, :, 1, k])
            end
        end
    end
    for (file, pairs) in ((files.ts, (("layer_temperature", t_file, "thetao"), ("layer_salinity", s_file, "so"))),
                          (files.uv, (("u_velocity", uv_file, "uo"), ("v_velocity", uv_file, "vo"))))
        NCDataset(file, "a") do ds
            for (vout, src_file, vin) in pairs
                NCDataset(src_file) do src
                    for (j, k) in enumerate(i3)
                        ds[vout][:, :, :, j] = nan(src[vin][:, :, :, k])
                    end
                end
            end
        end
    end
    println("Wrote CMEMS parent files for ", par_name)
    return files
end

## initial file
ini_par = Dates.format(ini_time, "yyyymmddHH")
ini_file = joinpath(initial_path, "roms_ini_$(dx)m$(ini_par).nc")
if !isfile(ini_file)
    f = make_cmems_day(ini_par, [ini_time + Hour(h) for h in rec_hours])
    println(">>> Creating initial file: ", ini_file)
    h2r_create_ini(ini_file, chdgrd, chd_N, chdscd)
    h2r_make_ini(par_grd, 1, f.uv, f.uv, f.ts, f.ts, f.ssh,
        chdgrd, ini_file, chdscd, chdscoord, ndomx, ndomy, chd_ang, nz)
end

## boundary files, one per day
obcflag = [1, 1, 1, 1]   # S E N W open
limits = h2r_bry_subgrid(par_grd, chdgrd, obcflag)
cache = Dict{Int,Any}()   # coefficients, reused while the parent mask is unchanged
for d in dating
    t0 = DateTime(d)
    par_name = Dates.format(t0, "yyyymmddHH")
    bryfile = joinpath(boundary_path, "roms_bry_$(dx)m_$(par_name).nc")
    isfile(bryfile) && continue
    times = [t0 + Hour(h) for h in rec_hours]
    f = make_cmems_day(par_name, times)

    println("Creating boundary file: ", basename(bryfile))
    tmp = bryfile * ".tmp"   # only a finished file gets the final name
    h2r_create_bry(tmp, chdgrd, obcflag, chdscd)
    for ii in eachindex(times)
        h2r_bry_hv(par_grd, chdgrd, f.ssh, f.ts, f.uv, nz, tmp, chdscd, obcflag, limits, ii; cache = cache)
    end
    # bry_time: days since 1994-01-01, one per record (h2r_bry_hv writes the first MT everywhere)
    for (tout, t) in enumerate(times)
        ncwrite(tmp, "bry_time", Dates.value(t - DateTime(1994, 1, 1)) / 86_400_000, tout)
    end
    mv(tmp, bryfile; force = true)
    println("bry_time: ", ncread(bryfile, "bry_time"))
end
println("done")
