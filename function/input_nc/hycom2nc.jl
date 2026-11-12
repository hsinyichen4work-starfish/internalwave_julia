# hycom2nc.jl
#
# Convert one HYCOM forecast cycle (GOFS 3.5 glbz04 z-level NetCDF:
# *_ssh.nc, *_ts3z.nc, *_uv3z.nc, one file per forecast hour) into the
# "parent" NetCDF files the h2r routines read — the HYCOM counterpart of
# make_bry_need_nc (ncom2nc.jl). Port of HYCOM2nc/make_bry_need_nc_hycom.m.
#
#   hycom_<par_name>_grd.nc : Longitude, Latitude (xi, eta), depth (z)
#   hycom_<par_name>_ssh.nc : ssh (xi, eta, time), MT
#   hycom_<par_name>_ts.nc  : layer_temperature, layer_salinity (xi, eta, z, time), MT
#   hycom_<par_name>_uv.nc  : u_velocity, v_velocity (xi, eta, z, time), MT
#
# Differences from the NCOM files:
#   - the grid file holds the 1-D z-levels `depth` (m, positive down)
#     instead of `layer_thickness`; parent_zgrid uses them directly.
#   - temperature is already degC; u/v are already at the cell centres
#     (same points as T/S) and eastward/northward: no conversion, no
#     face averaging, no rotation.
#   - land and below-seafloor points are NaN (HYCOM _FillValue).
#   - only the box around the child grid (+ margin, degrees) is written.

const HYCOM_DATA_PATH = "/home/mbui/ModelOutput/NCOM/HYCOM_data/subset/"

# read a HYCOM variable with CF scaling (scale_factor/add_offset) applied
# and fill values as NaN. (ncread in nc_tools.jl returns the raw stored
# shorts, which is wrong for these files.)
_hycom_read(file, var, idx...) =
    NCDataset(ds -> Float64.(coalesce.(ds[var][idx...], NaN)), file, "r")

"""
    hycom_valid_time(file) -> (valid, tau, analysis)

Times of one HYCOM file, from its metadata (not its name): `valid` =
`time` (hours since 2000-01-01, the time the data is valid for), `tau` =
forecast hour, `analysis` = tau's time_origin (the cycle / analysis time).
"""
function hycom_valid_time(file::AbstractString)
    NCDataset(file, "r") do ds
        # raw variables: indexing ds["tau"] makes NCDatasets try (and fail) to
        # parse its units "hours since analysis" as a CF time
        vtime, vtau = variable(ds, "time"), variable(ds, "tau")
        valid = DateTime(2000, 1, 1) + Millisecond(round(Int, vtime[1] * 3_600_000))
        tau = Float64(vtau[1])
        analysis = DateTime(vtau.attrib["time_origin"], dateformat"yyyy-mm-dd HH:MM:SS")
        (valid = valid, tau = tau, analysis = analysis)
    end
end

"""
    hycom_find_time(t; hycom_path = HYCOM_DATA_PATH) -> (cycle, tau, file)

The HYCOM cycle holding valid time `t` (from each file's `time`
variable). When several cycles hold it (e.g. t0024 of one cycle and t0000
of the next), the one with the smallest tau — the most recent analysis —
is returned. `cycle` is "yyyymmddHH" of the analysis time.
"""
function hycom_find_time(t::DateTime; hycom_path::AbstractString = HYCOM_DATA_PATH)
    best = nothing
    for f in readdir(hycom_path)
        (startswith(f, "gofs35_hycom_glbz04_") && endswith(f, "_ssh.nc")) || continue
        m = hycom_valid_time(joinpath(hycom_path, f))
        m.valid == t || continue
        if best === nothing || m.tau < best.tau
            best = (cycle = Dates.format(m.analysis, "yyyymmddHH"), tau = m.tau, file = f)
        end
    end
    best === nothing && error("hycom_find_time: no HYCOM file in $hycom_path is valid at $t")
    return best
end

"""
    make_bry_need_nc_hycom(par_name, out_path, remake = false; chdgrd,
                           hycom_path = HYCOM_DATA_PATH, margin = 0.25)
        -> (par_grd, parinie, parinit, pariniu)

Write the parent files of HYCOM cycle `par_name` ("yyyymmddHH", GOFS
cycles are 12Z, e.g. "2022082312") into `out_path`, holding every
forecast hour available for that cycle (t0000..t0021 or t0024, 3-hourly).
`chdgrd` (child grid file) sets the subset box. Existing files are kept
unless `remake`. MT = days since 1900-12-31, as for NCOM.
"""
function make_bry_need_nc_hycom(par_name::AbstractString, out_path::AbstractString, remake::Bool = false;
                                chdgrd::AbstractString, hycom_path::AbstractString = HYCOM_DATA_PATH,
                                margin::Real = 0.25)
    par_grd = "hycom_" * par_name * "_grd.nc"
    parinie = "hycom_" * par_name * "_ssh.nc"
    parinit = "hycom_" * par_name * "_ts.nc"
    pariniu = "hycom_" * par_name * "_uv.nc"
    fgrd, fe, ft, fu = (joinpath(out_path, f) for f in (par_grd, parinie, parinit, pariniu))
    if !remake && all(isfile, (fgrd, fe, ft, fu))
        return par_grd, parinie, parinit, pariniu
    end

    # files of this cycle, sorted by forecast hour (t#### is zero padded)
    pre = "gofs35_hycom_glbz04_" * par_name * "_t"
    names = sort(filter(f -> startswith(f, pre) && endswith(f, "_ssh.nc"), readdir(hycom_path)))
    isempty(names) && error("make_bry_need_nc_hycom: no $(pre)*_ssh.nc files in $hycom_path")
    tags = [f[length(pre)+1:length(pre)+4] for f in names]
    fssh = [joinpath(hycom_path, pre * t * "_ssh.nc") for t in tags]
    fts = [joinpath(hycom_path, pre * t * "_ts3z.nc") for t in tags]
    fuv = [joinpath(hycom_path, pre * t * "_uv3z.nc") for t in tags]
    missing_files = filter(!isfile, vcat(fts, fuv))
    isempty(missing_files) || error("make_bry_need_nc_hycom: missing HYCOM files:\n" * join(missing_files, "\n"))
    nt = length(tags)

    # subset box around the child grid
    lon, lat, depth = NCDataset(fts[1], "r") do ds
        wrap360(Float64.(ds["lon"][:])), Float64.(ds["lat"][:]), Float64.(ds["depth"][:])
    end
    lonc = wrap360(ncread(chdgrd, "lon_rho"))
    latc = ncread(chdgrd, "lat_rho")
    box = (minimum(lonc) - margin, maximum(lonc) + margin, minimum(latc) - margin, maximum(latc) + margin)
    if box[1] < minimum(lon) || box[2] > maximum(lon) || box[3] < minimum(lat) || box[4] > maximum(lat)
        error(@sprintf("make_bry_need_nc_hycom: child grid + margin [%.2f %.2f %.2f %.2f] is not inside the HYCOM data [%.2f %.2f %.2f %.2f]",
            box..., minimum(lon), maximum(lon), minimum(lat), maximum(lat)))
    end
    ii = findall(x -> box[1] <= x <= box[2], lon)
    jj = findall(x -> box[3] <= x <= box[4], lat)
    ir, jr = ii[1]:ii[end], jj[1]:jj[end]
    ni, nj, nz = length(ir), length(jr), length(depth)
    @printf("HYCOM %s: %d time steps, subset %d x %d x %d (lon %.2f-%.2f, lat %.2f-%.2f)\n",
        par_name, nt, ni, nj, nz, lon[ir[1]], lon[ir[end]], lat[jr[1]], lat[jr[end]])

    # grid file
    isfile(fgrd) && rm(fgrd)
    NCDataset(fgrd, "c"; format = :netcdf4_classic) do ds
        defDim(ds, "xi", ni)
        defDim(ds, "eta", nj)
        defDim(ds, "z", nz)
        defVar(ds, "Longitude", [lon[i] for i in ir, j in jr], ("xi", "eta"))
        defVar(ds, "Latitude", [lat[j] for i in ir, j in jr], ("xi", "eta"))
        defVar(ds, "depth", depth, ("z",); attrib = ["units" => "m", "positive" => "down"])
    end

    # data files
    _create_parent_nc(fe, ni, nj, nothing, ["ssh" => ("xi", "eta", "time")])
    _create_parent_nc(ft, ni, nj, nz, ["layer_temperature" => ("xi", "eta", "z", "time"),
                                       "layer_salinity" => ("xi", "eta", "z", "time")])
    _create_parent_nc(fu, ni, nj, nz, ["u_velocity" => ("xi", "eta", "z", "time"),
                                       "v_velocity" => ("xi", "eta", "z", "time")])

    t_hycom = DateTime(2000, 1, 1)   # HYCOM time = hours since 2000-01-01
    MT = zeros(nt)
    NCDataset(fe, "a") do de
        NCDataset(ft, "a") do dt
            NCDataset(fu, "a") do du
                for k in 1:nt
                    tim = [NCDataset(f -> f["time"].var[1], x, "r") for x in (fssh[k], fts[k], fuv[k])]
                    all(==(tim[1]), tim) || error("make_bry_need_nc_hycom: ssh/ts/uv times differ for $par_name t$(tags[k])")
                    valid = t_hycom + Millisecond(round(Int, tim[1] * 3_600_000))
                    MT[k] = Dates.value(valid - DateTime(1900, 12, 31)) / 86_400_000

                    de["ssh"][:, :, k] = _hycom_read(fssh[k], "surf_el", ir, jr, 1)
                    dt["layer_temperature"][:, :, :, k] = _hycom_read(fts[k], "water_temp", ir, jr, :, 1)
                    dt["layer_salinity"][:, :, :, k] = _hycom_read(fts[k], "salinity", ir, jr, :, 1)
                    du["u_velocity"][:, :, :, k] = _hycom_read(fuv[k], "water_u", ir, jr, :, 1)
                    du["v_velocity"][:, :, :, k] = _hycom_read(fuv[k], "water_v", ir, jr, :, 1)
                    println("  t", tags[k], "  ", Dates.format(valid, "yyyy-mm-dd HH:MM"))
                end
            end
        end
    end
    foreach(f -> _add_MT(f, MT), (fe, ft, fu))

    println("Wrote ", par_grd, ", ", parinie, ", ", parinit, ", ", pariniu, " in ", out_path)
    return par_grd, parinie, parinit, pariniu
end
