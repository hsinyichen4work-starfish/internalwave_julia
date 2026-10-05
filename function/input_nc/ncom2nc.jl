# ncom2nc.jl
#
# Convert one day of raw NCOM flat files into the "parent" NetCDF files
# the h2r routines read. Ported from NCOM2nc/make_bry_need_*.m and
# make_frc_need_*.m.
#
#   <par_name>_lthick.nc : Longitude, Latitude, layer_thickness
#   <par_name>_ssh.nc    : ssh
#   <par_name>_ts.nc     : layer_temperature, layer_salinity
#   <par_name>_uv.nc     : u_velocity, v_velocity (cell centers, true east/north), MT
#   <par_name>_flx.nc    : heaflx, salflx, solflx
#   <par_name>_wnd.nc    : stresu, stresv (true east/north)
#   <par_name>_pres.nc   : slpres
#
# All variables are (xi, eta[, z], time) with an unlimited time dimension.
# The MATLAB versions held all 25 time steps of a field in memory before
# writing (33 GB per 3D variable); here each time step is read, processed
# and written on its own.

const NCOM_DATA_PATH = "/home/mbui/ModelOutput/NCOM/data/"
const NCOM_GRID_PATH = "/home/mbui/ModelOutput/NCOM/grid/"

# create `file` (netCDF-4 classic model, like MATLAB's nccreate) with the
# given double variables; vars = [name => (dimnames...)]
function _create_parent_nc(file, igrd, jgrd, lo, vars)
    isfile(file) && rm(file)
    NCDataset(file, "c"; format = :netcdf4_classic) do ds
        defDim(ds, "xi", igrd)
        defDim(ds, "eta", jgrd)
        lo === nothing || defDim(ds, "z", lo)
        defDim(ds, "time", Inf)
        for (name, dims) in vars
            defVar(ds, name, Float64, dims)
        end
    end
    return file
end

# MATLAB: nccreate(file,'MT','Dimensions',{'time',Inf}); ncwrite(file,'MT',MT)
function _add_MT(file, MT)
    NCDataset(file, "a") do ds
        v = defVar(ds, "MT", Float64, ("time",))
        v[1:length(MT)] = MT
    end
    return nothing
end

# (igrd, jgrd, lo) mask: land everywhere lndsea is 0, and below the last
# valid layer kb at sea points. Land kb is the int32 fill value, hence
# the `> 0` test.
function _ncom_mask3d(lndsea::AbstractMatrix, valid_lay::AbstractMatrix, lo::Integer)
    mask3d = repeat(lndsea, 1, 1, lo)
    for j in axes(lndsea, 2), i in axes(lndsea, 1)
        kb = valid_lay[i, j]
        if kb > 0 && kb < lo
            mask3d[i, j, kb+1:end] .= 0
        end
    end
    return mask3d
end

"""
    make_bry_need_grid(lon, lat, vgrd2, par_name, boundary_path) -> par_grd

Write `<par_name>_lthick.nc`: Longitude, Latitude and the (static) layer
thickness computed from the interface depths `vgrd2` (igrd, jgrd, lo+1).
"""
function make_bry_need_grid(lon::AbstractMatrix, lat::AbstractMatrix, vgrd2::AbstractArray{<:Real,3},
                            par_name::AbstractString, boundary_path::AbstractString)
    igrd, jgrd = size(lon)
    lo = size(vgrd2, 3) - 1

    # layer thickness (positive-valued) from interface depths
    dz = -diff(vgrd2; dims = 3)
    any(<(0), dz) &&
        @warn "Negative layer thickness found -- check vgrd2 sign convention/masking before proceeding."

    par_grd = par_name * "_lthick.nc"
    file = joinpath(boundary_path, par_grd)
    # layer_thickness is read by h2r_bry_hv as a 4D field, so it needs an
    # explicit (unlimited) time dimension even with a single time slice
    isfile(file) && rm(file)
    NCDataset(file, "c"; format = :netcdf4_classic) do ds
        defDim(ds, "xi", igrd)
        defDim(ds, "eta", jgrd)
        defDim(ds, "z", lo)
        defDim(ds, "time", Inf)
        vlon = defVar(ds, "Longitude", Float64, ("xi", "eta"))
        vlat = defVar(ds, "Latitude", Float64, ("xi", "eta"))
        vdz = defVar(ds, "layer_thickness", Float64, ("xi", "eta", "z", "time"))
        vlon[:, :] = lon
        vlat[:, :] = lat
        vdz[:, :, :, 1] = dz   # static reference thickness at the first time index
    end

    println("Wrote grid file: ", par_grd)
    return par_grd
end

"""
    make_bry_need_ssh(igrd, jgrd, par_name, path_setup, boundary_path, lndsea = nothing) -> parinie

Write `<par_name>_ssh.nc` from the seahgt* flat files. `lndsea` is the
(igrd, jgrd) land/sea mask (1 = sea, 0 = land); land points are set to
NaN so the isnan-based masking in h2r works.
"""
function make_bry_need_ssh(igrd, jgrd, par_name, path_setup, boundary_path, lndsea = nothing)
    parsed = ncom_file_list(path_setup, "seahgt")
    nfiles = length(parsed)

    parinie = par_name * "_ssh.nc"
    file = _create_parent_nc(joinpath(boundary_path, parinie), igrd, jgrd, nothing,
        ["ssh" => ("xi", "eta", "time")])
    land = lndsea === nothing ? nothing : (lndsea .== 0)
    NCDataset(file, "a") do ds
        for j in 1:nfiles
            field = read_ncom_flatfile(path_setup, parsed[j])
            land === nothing || (field[land] .= NaN)   # mask land explicitly
            ds["ssh"][:, :, j] = field
        end
    end

    println("Wrote SSH file: ", parinie, " (", nfiles, " time steps)")
    return parinie
end

"""
    make_bry_need_temp(igrd, jgrd, lo, valid_lay, par_name, path_setup, boundary_path, lndsea = nothing) -> parinit

Write `<par_name>_ts.nc`: layer_temperature (Celsius) from seatmp* and
layer_salinity from salint*. Land and below-bottom points
(`valid_lay` = NCOM's kb) are NaN.
"""
function make_bry_need_temp(igrd, jgrd, lo, valid_lay, par_name, path_setup, boundary_path, lndsea = nothing)
    land3d = lndsea === nothing ? nothing : (_ncom_mask3d(lndsea, valid_lay, lo) .== 0)

    # temperature and salinity are parsed and sorted separately
    parsed_t = ncom_file_list(path_setup, "seatmp")
    parsed_s = ncom_file_list(path_setup, "salint")
    nfiles_t, nfiles_s = length(parsed_t), length(parsed_s)
    nfiles_s == nfiles_t ||
        @warn "Found $nfiles_t seatmp files but $nfiles_s salint files -- check your data folder."

    parinit = par_name * "_ts.nc"
    file = _create_parent_nc(joinpath(boundary_path, parinit), igrd, jgrd, lo,
        ["layer_temperature" => ("xi", "eta", "z", "time"), "layer_salinity" => ("xi", "eta", "z", "time")])
    NCDataset(file, "a") do ds
        for j in 1:nfiles_t
            field = read_ncom_flatfile(path_setup, parsed_t[j])
            field .-= 273.15   # Kelvin -> Celsius
            land3d === nothing || (field[land3d] .= NaN)
            ds["layer_temperature"][:, :, :, j] = field
        end
        for j in 1:nfiles_s
            field = read_ncom_flatfile(path_setup, parsed_s[j])
            land3d === nothing || (field[land3d] .= NaN)
            ds["layer_salinity"][:, :, :, j] = field
        end
    end

    println("Wrote temp/salt file: ", parinit, " (", nfiles_t, " temp steps, ", nfiles_s, " salt steps)")
    return parinit
end

"""
    make_bry_need_vel(igrd, jgrd, lo, valid_lay, grdang, par_name, path_setup, boundary_path, lndsea = nothing) -> pariniu

Write `<par_name>_uv.nc` from uucurr*/vvcurr*: faces averaged to cell
centers, rotated to true east/north with `grdang` (degrees, (igrd, jgrd)),
then masked. Also writes MT (days since 1900-12-31), one per time step.
"""
function make_bry_need_vel(igrd, jgrd, lo, valid_lay, grdang, par_name, path_setup, boundary_path, lndsea = nothing)
    land3d = lndsea === nothing ? nothing : (_ncom_mask3d(lndsea, valid_lay, lo) .== 0)

    # u and v are parsed and sorted separately
    parsed_u = ncom_file_list(path_setup, "uucurr")
    parsed_v = ncom_file_list(path_setup, "vvcurr")
    nfiles_u, nfiles_v = length(parsed_u), length(parsed_v)
    nfiles_v == nfiles_u ||
        error("Found $nfiles_u uucurr files but $nfiles_v vvcurr files -- check your data folder.")

    pariniu = par_name * "_uv.nc"
    file = _create_parent_nc(joinpath(boundary_path, pariniu), igrd, jgrd, lo,
        ["u_velocity" => ("xi", "eta", "z", "time"), "v_velocity" => ("xi", "eta", "z", "time")])
    NCDataset(file, "a") do ds
        for j in 1:nfiles_u
            # do NOT mask before averaging: faces must stay raw (0 on land)
            uc = avg_face_to_center(read_ncom_flatfile(path_setup, parsed_u[j]), 1)   # x-faces -> centers
            vc = avg_face_to_center(read_ncom_flatfile(path_setup, parsed_v[j]), 2)   # y-faces -> centers
            u_true, v_true = vel_rot(uc, vc, grdang, "grid2geo")
            if land3d !== nothing   # mask centers after averaging
                u_true[land3d] .= NaN
                v_true[land3d] .= NaN
            end
            ds["u_velocity"][:, :, :, j] = u_true
            ds["v_velocity"][:, :, :, j] = v_true
        end
    end
    _add_MT(file, ncom_MT(parsed_u))

    println("Wrote u/v file: ", pariniu, " (", nfiles_u, " u steps, ", nfiles_v, " v steps)")
    return pariniu
end

# the pieces of the static NCOM grid the converters need
function _ncom_grid_info(grid_process_path, nest)
    igrd, jgrd, lon, lat = read_ohgrd(grid_process_path, nest)
    dimz = parse.(Int, split(read(joinpath(grid_process_path, "ovgrd_$(nest).B"), String)))[3]
    return (igrd = igrd, jgrd = jgrd, lon = lon, lat = lat, lo = dimz - 1,
            ohgrd = joinpath(grid_process_path, "ohgrd_$(nest).nc"))
end

"""
    make_bry_need_nc(par_name, boundary_path, remake = false; kwargs...) -> (par_grd, parinie, parinit, pariniu)

Make sure the four parent files needed for the initial/boundary files
exist in `boundary_path` (lthick, ssh, ts, uv), building the missing ones
— or all of them when `remake` is true — and return their file names.

Keywords: `parent_data_path` (holds one folder of flat files per
`par_name`), `grid_process_path` (ohgrd/ovgrd files), `nest`.
"""
function make_bry_need_nc(par_name::AbstractString, boundary_path::AbstractString, remake::Bool = false;
                          parent_data_path::AbstractString = NCOM_DATA_PATH,
                          grid_process_path::AbstractString = NCOM_GRID_PATH, nest::Integer = 2)
    path_setup = joinpath(parent_data_path, par_name)

    par_grd = par_name * "_lthick.nc"
    parinie = par_name * "_ssh.nc"
    parinit = par_name * "_ts.nc"
    pariniu = par_name * "_uv.nc"

    make_g = remake || !isfile(joinpath(boundary_path, par_grd))
    make_u = remake || !isfile(joinpath(boundary_path, pariniu))
    make_e = remake || !isfile(joinpath(boundary_path, parinie))
    make_t = remake || !isfile(joinpath(boundary_path, parinit))
    (make_g || make_u || make_e || make_t) || return par_grd, parinie, parinit, pariniu

    g = _ncom_grid_info(grid_process_path, nest)
    mask = ncread(g.ohgrd, "mask")
    valid_lay = ncread(g.ohgrd, "kb")

    if make_g
        println("create ", par_grd)
        make_bry_need_grid(g.lon, g.lat, read_ovgrdA(grid_process_path, nest), par_name, boundary_path)
    end
    if make_u
        println("create ", pariniu)
        make_bry_need_vel(g.igrd, g.jgrd, g.lo, valid_lay, ncread(g.ohgrd, "ang"), par_name, path_setup,
            boundary_path, mask)
    end
    MT = ncread(joinpath(boundary_path, pariniu), "MT")
    if make_e
        println("create ", parinie)
        make_bry_need_ssh(g.igrd, g.jgrd, par_name, path_setup, boundary_path, mask)
        _add_MT(joinpath(boundary_path, parinie), MT)
    end
    if make_t
        println("create ", parinit)
        make_bry_need_temp(g.igrd, g.jgrd, g.lo, valid_lay, par_name, path_setup, boundary_path, mask)
        _add_MT(joinpath(boundary_path, parinit), MT)
    end

    return par_grd, parinie, parinit, pariniu
end

# read every file of a 2D field, masking land, into `ds[varname][:, :, t]`
function _write_2d_series!(ds, varname, parsed, path_setup, land)
    for j in eachindex(parsed)
        field = read_ncom_flatfile(path_setup, parsed[j])
        land === nothing || (field[land] .= NaN)
        ds[varname][:, :, j] = field
    end
    return nothing
end

"""
    make_frc_need_wind(igrd, jgrd, grdang, par_name, path_setup, forcing_path, lndsea = nothing) -> pariniw

Write `<par_name>_wnd.nc` from stresu*/stresv*. The stresses are already
at cell centers (no face averaging) but still along the model's rotated
grid axes, so they are rotated to true east/north with `grdang`.
"""
function make_frc_need_wind(igrd, jgrd, grdang, par_name, path_setup, forcing_path, lndsea = nothing)
    land = lndsea === nothing ? nothing : (lndsea .== 0)

    parsed_u = ncom_file_list(path_setup, "stresu")
    parsed_v = ncom_file_list(path_setup, "stresv")
    nfiles_u, nfiles_v = length(parsed_u), length(parsed_v)
    nfiles_v == nfiles_u ||
        error("Found $nfiles_u stresu files but $nfiles_v stresv files -- check your data folder.")

    pariniw = par_name * "_wnd.nc"
    file = _create_parent_nc(joinpath(forcing_path, pariniw), igrd, jgrd, nothing,
        ["stresu" => ("xi", "eta", "time"), "stresv" => ("xi", "eta", "time")])
    NCDataset(file, "a") do ds
        for j in 1:nfiles_u
            u = read_ncom_flatfile(path_setup, parsed_u[j])
            v = read_ncom_flatfile(path_setup, parsed_v[j])
            if land !== nothing
                u[land] .= NaN
                v[land] .= NaN
            end
            u_true, v_true = vel_rot(u, v, grdang, "grid2geo")
            ds["stresu"][:, :, j] = u_true
            ds["stresv"][:, :, j] = v_true
        end
    end
    _add_MT(file, ncom_MT(parsed_u))

    println("Wrote wind stress file: ", pariniw, " (", nfiles_u, " u steps, ", nfiles_v, " v steps)")
    return pariniw
end

"""
    make_frc_need_flux(igrd, jgrd, par_name, path_setup, forcing_path, lndsea = nothing) -> parinis

Write `<par_name>_flx.nc`: heaflx, salflx, solflx. MT comes from the
heaflx file list; the three fields are assumed to share time steps.
"""
function make_frc_need_flux(igrd, jgrd, par_name, path_setup, forcing_path, lndsea = nothing)
    land = lndsea === nothing ? nothing : (lndsea .== 0)
    fields = ["heaflx", "salflx", "solflx"]

    parinis = par_name * "_flx.nc"
    file = _create_parent_nc(joinpath(forcing_path, parinis), igrd, jgrd, nothing,
        [fld => ("xi", "eta", "time") for fld in fields])
    parsed_ref = ncom_file_list(path_setup, fields[1])
    NCDataset(file, "a") do ds
        for fld in fields
            parsed = ncom_file_list(path_setup, fld)
            length(parsed) == length(parsed_ref) ||
                @warn "Found $(length(parsed)) $fld files but $(length(parsed_ref)) $(fields[1]) files -- check your data folder."
            _write_2d_series!(ds, fld, parsed, path_setup, land)
        end
    end
    _add_MT(file, ncom_MT(parsed_ref))

    println("Wrote flux file: ", parinis, " (", length(parsed_ref), " time steps)")
    return parinis
end

"""
    make_frc_need_pres(igrd, jgrd, par_name, path_setup, forcing_path, lndsea = nothing) -> parinip

Write `<par_name>_pres.nc`: slpres.
"""
function make_frc_need_pres(igrd, jgrd, par_name, path_setup, forcing_path, lndsea = nothing)
    land = lndsea === nothing ? nothing : (lndsea .== 0)
    parsed = ncom_file_list(path_setup, "slpres")

    parinip = par_name * "_pres.nc"
    file = _create_parent_nc(joinpath(forcing_path, parinip), igrd, jgrd, nothing,
        ["slpres" => ("xi", "eta", "time")])
    NCDataset(file, "a") do ds
        _write_2d_series!(ds, "slpres", parsed, path_setup, land)
    end
    _add_MT(file, ncom_MT(parsed))

    println("Wrote pressure file: ", parinip, " (", length(parsed), " time steps)")
    return parinip
end

"""
    make_frc_need_nc(par_name, forcing_path, remake = false; kwargs...) -> (par_grd, parinis, pariniw, parinip)

Make sure the parent files needed for the forcing file exist in
`forcing_path` (lthick, flx, wnd, pres) and return their file names.
Keywords as in `make_bry_need_nc`.
"""
function make_frc_need_nc(par_name::AbstractString, forcing_path::AbstractString, remake::Bool = false;
                          parent_data_path::AbstractString = NCOM_DATA_PATH,
                          grid_process_path::AbstractString = NCOM_GRID_PATH, nest::Integer = 2)
    path_setup = joinpath(parent_data_path, par_name)

    par_grd = par_name * "_lthick.nc"
    parinis = par_name * "_flx.nc"
    pariniw = par_name * "_wnd.nc"
    parinip = par_name * "_pres.nc"

    make_g = remake || !isfile(joinpath(forcing_path, par_grd))
    make_w = remake || !isfile(joinpath(forcing_path, pariniw))
    make_s = remake || !isfile(joinpath(forcing_path, parinis))
    make_p = remake || !isfile(joinpath(forcing_path, parinip))
    (make_g || make_w || make_s || make_p) || return par_grd, parinis, pariniw, parinip

    g = _ncom_grid_info(grid_process_path, nest)
    mask = ncread(g.ohgrd, "mask")

    if make_g
        println("create ", par_grd)
        make_bry_need_grid(g.lon, g.lat, read_ovgrdA(grid_process_path, nest), par_name, forcing_path)
    end
    if make_w
        println("create ", pariniw)
        make_frc_need_wind(g.igrd, g.jgrd, ncread(g.ohgrd, "ang"), par_name, path_setup, forcing_path, mask)
    end
    if make_s
        println("create ", parinis)
        make_frc_need_flux(g.igrd, g.jgrd, par_name, path_setup, forcing_path, mask)
    end
    if make_p
        println("create ", parinip)
        make_frc_need_pres(g.igrd, g.jgrd, par_name, path_setup, forcing_path, mask)
    end

    return par_grd, parinis, pariniw, parinip
end
