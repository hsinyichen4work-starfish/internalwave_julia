# build_steps.jl
#
# The four build steps of input_make.jl: grid, initial, boundary, forcing.
# Ported from matlab_funtion/ncbuild/ (grd_build.m, ini_build.m,
# bry_build.m, frc_build.m). Those were scripts sharing the caller's
# workspace; here each one is a function of `cfg`, a NamedTuple holding
# the settings input_make.jl defines (see that file for the fields).

_chdgrd(cfg) = joinpath(cfg.grid_path, cfg.grd_name * ".nc")
_chdscd(cfg) = (N = cfg.chd_N, theta_s = cfg.chd_thetas, theta_b = cfg.chd_thetab, hc = cfg.chd_hc,
                scoord = cfg.chdscoord)

# MATLAB datenum -> DateTime
_datenum2datetime(dn::Real) = DateTime(0, 1, 1) + Millisecond(round(Int, (dn - 1) * 86_400_000))

"""
    grd_build(cfg) -> (grd_nc, diag)

Build the child grid file `grid_path/grd_name.nc`: place the grid on the
mooring line, interpolate the GEBCO bathymetry onto it, write the file
and smooth `h`. Then blend the bathymetry with the parent's near the
open boundaries — in the returned `grd_nc` only (`h`, with the smoothed
one kept as `h_orig`); the file keeps the smoothed `h`, as in grd_build.m.

If `cfg.path_figure` is set, the two match_topo_check figures are saved there.
"""
function grd_build(cfg)
    # Build the grid & check the size
    mid, rot_ang = gid_middle(cfg.mid_iter)
    grd, actual_dx_APPROX, actual_dy_APPROX = grid_setting(mid, rot_ang, cfg.dx, cfg.nx, cfg.ny)
    println(cfg.dx, "m : actual_dx_APPROX = ", num2str(actual_dx_APPROX),
        ", actual_dy_APPROX = ", num2str(actual_dy_APPROX))
    grdlon = rad2deg.(grd["lon4"]) .- 360
    grdlat = rad2deg.(grd["lat4"])

    # bathymetry for interpolation
    small = 0.1
    topo = read_topo_subset(joinpath(cfg.bath_path, "GEBCO_2025I2021.nc"),
        (minimum(grdlon) - small, maximum(grdlon) + small),
        (minimum(grdlat) - small, maximum(grdlat) + small))

    # parent grid bath (positive down)
    pgrid = read_nc_fun(cfg.parent_grid)
    wet = pgrid["mask"] .== 1
    if mean(filter(!isnan, pgrid["h"][wet])) < 0
        pgrid["h"] = -pgrid["h"]
    end

    grd = bathy_interp(topo, grd)
    grd_nc = make_roms_ncgrid(grd, cfg.grd_name, mid, rot_ang, cfg.dx, cfg.nx, cfg.ny, cfg.smooth_var, cfg.grid_path)
    grd_nc, diag = match_boundary_topo(pgrid, grd_nc, [1, 1, 1, 1], 4, 5)

    path_figure = get(cfg, :path_figure, nothing)
    if path_figure !== nothing
        # CairoMakie is only loaded when figures are asked for
        isdefined(@__MODULE__, :match_topo_figures) || Base.include(@__MODULE__, joinpath(@__DIR__, "grid_figures.jl"))
        Base.invokelatest(getfield(@__MODULE__, :match_topo_figures), grd_nc, diag, cfg.dx, path_figure)
    end
    return grd_nc, diag
end

"""
    ini_build(cfg, pgrid, par_name)

Build the initial file `initial_path/ini_name` from parent day `par_name`
(time index 1), converting the NCOM flat files to NetCDF first if needed.
`pgrid` is the parent grid Dict (only the number of levels in `zm3` is used).
"""
function ini_build(cfg, pgrid, par_name::AbstractString)
    par_N = size(pgrid["zm3"], 3)   # number of parent vertical levels
    par_tind = 1                    # frame number in parent file
    remake = false
    par_grd, parinie, parinit, pariniu = make_bry_need_nc(par_name, cfg.nc_path_ini_bry, remake;
        parent_data_path = cfg.parent_data_path)

    # Full file paths
    parent_UV = joinpath(cfg.nc_path_ini_bry, pariniu)
    parent_TS = joinpath(cfg.nc_path_ini_bry, parinit)
    parent_E = joinpath(cfg.nc_path_ini_bry, parinie)
    parent_G = joinpath(cfg.nc_path_ini_bry, par_grd)

    chdgrd = _chdgrd(cfg)
    chdini = joinpath(cfg.initial_path, cfg.ini_name)
    chdscd = _chdscd(cfg)

    # Create and fill initial file
    println(">>> Creating initial file: ", chdini)
    h2r_create_ini(chdini, chdgrd, cfg.chd_N, chdscd)
    h2r_make_ini(parent_G, par_tind, parent_UV, parent_UV, parent_TS, parent_TS, parent_E,
        chdgrd, chdini, chdscd, cfg.chdscoord, cfg.ndomx, cfg.ndomy, cfg.chd_ang, par_N)
    return nothing
end

"""
    bry_build(cfg, par_name, bry_filename; state = Dict())

Build the boundary file `boundary_path/bry_filename` from every time
step of parent day `par_name`, converting the NCOM flat files to NetCDF
first if needed. An existing file is replaced.

`state` carries what is the same for every day — the parent subgrid of
each boundary and the interpolation coefficients — so passing the same
Dict to successive calls computes them only once. That is valid as long
as the child grid and the (static) NCOM grid don't change between calls.
"""
function bry_build(cfg, par_name::AbstractString, bry_filename::AbstractString; state::AbstractDict = Dict{Symbol,Any}())
    println("read data from ", joinpath(cfg.parent_data_path, par_name))
    # make data nc file that can be read in boundary file
    remake = false
    par_grd, parinie, parinit, pariniu = make_bry_need_nc(par_name, cfg.nc_path_ini_bry, remake;
        parent_data_path = cfg.parent_data_path)

    # general parent/child grid setting
    parent_UV = joinpath(cfg.nc_path_ini_bry, pariniu)
    parent_TS = joinpath(cfg.nc_path_ini_bry, parinit)
    parent_E = joinpath(cfg.nc_path_ini_bry, parinie)
    parent_G = joinpath(cfg.nc_path_ini_bry, par_grd)
    Np = ncsize(parent_G, "layer_thickness")[3]

    chdgrd = _chdgrd(cfg)
    chdscd = _chdscd(cfg)

    # BOUNDARY FILE setting
    obcflag = [1, 1, 1, 1]      # open boundaries flag (1=open , [S E N W])

    # create empty boundary file first
    bryfile = joinpath(cfg.boundary_path, bry_filename)
    println("Creating boundary file: ", bry_filename)
    h2r_create_bry(bryfile, chdgrd, obcflag, chdscd)

    # Get parent subgrid bounds
    limits = get!(state, :limits) do
        println("\nGet parent subgrids for each open boundary")
        h2r_bry_subgrid(parent_G, chdgrd, obcflag)
    end
    cache = get!(() -> Dict{Int,Any}(), state, :cache)

    # Determine how many time steps are in the consolidated parent files
    ntimes = ncsize(parent_E, "ssh")[end]   # last dimension = time
    println("Found ", ntimes, " time steps in parent files")

    # Loop over every time step, writing each into the matching bry slot
    for ii in 1:ntimes
        println("--- Processing time step ", ii, " of ", ntimes, " ---")
        h2r_bry_hv(parent_G, chdgrd, parent_E, parent_TS, parent_UV, Np, bryfile, chdscd, obcflag, limits, ii;
            cache = cache)
    end

    # fix time: one value per time step, as written by make_bry_need_vel
    MT = ncread(parent_UV, "MT")
    for tout in eachindex(MT)
        ncwrite(bryfile, "bry_time", MT[tout] + _T1 - _T2, tout)
    end
    println("Rewrote bry_time for ", length(MT), " time steps in ", bry_filename)

    # Quick check
    for t in ncread(bryfile, "bry_time")
        println(Dates.format(_datenum2datetime(t + _T2), "dd-mm-yyyy HH:MM:SS"))
    end
    return nothing
end

"""
    frc_build(cfg, par_name, frc_filename; state = Dict())

Build the surface forcing file `forcing_path/frc_filename` from every
time step of parent day `par_name`, converting the NCOM flat files to
NetCDF first if needed. An existing file is replaced.

`state` keeps the parent subgrid limits of the child chunks, which only
depend on the two grids: pass the same Dict to successive calls to
compute them once, regardless of how many dates are processed.
"""
function frc_build(cfg, par_name::AbstractString, frc_filename::AbstractString; state::AbstractDict = Dict{Symbol,Any}())
    chdgrd = _chdgrd(cfg)
    frcname = joinpath(cfg.forcing_path, frc_filename)
    h2r_create_frc(frcname, chdgrd)

    # make data nc file that can be read in forcing file
    remake = false
    par_grd, parinis, pariniw, parinip = make_frc_need_nc(par_name, cfg.nc_path_frc, remake;
        parent_data_path = cfg.parent_data_path)

    parent_FLUX = joinpath(cfg.nc_path_frc, parinis)
    parent_WIND = joinpath(cfg.nc_path_frc, pariniw)
    parent_PRESS = joinpath(cfg.nc_path_frc, parinip)
    parent_G = joinpath(cfg.nc_path_frc, par_grd)
    # surface salinity, for the salflx -> cm/day conversion
    parent_TS = joinpath(cfg.nc_path_ini_bry, par_name * "_ts.nc")

    # Once, regardless of how many dates/time steps are processed
    limits = get!(() -> h2r_frc_subgrid(parent_G, chdgrd, cfg.ndomx, cfg.ndomy), state, :limits)

    # Then per date/par_name
    h2r_make_frc(parent_G, parent_FLUX, parent_WIND, parent_PRESS, chdgrd, frcname, cfg.chd_ang, limits, parent_TS)
    return nothing
end
