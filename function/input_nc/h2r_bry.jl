# h2r_bry.jl
#
# ROMS boundary file from the parent (NCOM) NetCDF files.
# Ported from h2r_GitHub/h2r_create_bry.m, h2r_bry_subgrid.m and
# h2r_bry_hv.m. Boundaries are numbered 1..4 = South, East, North, West.

const _BRY_NAMES = ("south", "east", "north", "west")
const _BRY_LONG = ("southern", "eastern", "northern", "western")

"""
    h2r_create_bry(bryname, grdname, obcflag, param)

Create an empty boundary file (classic NetCDF, unlimited bry_time) for
the grid `grdname`. `obcflag` = open boundary flags [S E N W] (1 = open);
`param` holds theta_s, theta_b, hc, N. An existing file is replaced.
"""
function h2r_create_bry(bryname::AbstractString, grdname::AbstractString, obcflag, param)
    N = param.N
    Lp, Mp = ncsize(grdname, "mask_rho")
    L = Lp - 1
    M = Mp - 1

    isfile(bryname) && rm(bryname)
    NCDataset(bryname, "c"; format = :netcdf3_classic) do ds
        defDim(ds, "xi_u", L)
        defDim(ds, "xi_rho", Lp)
        defDim(ds, "eta_v", M)
        defDim(ds, "eta_rho", Mp)
        defDim(ds, "s_rho", N)
        defDim(ds, "bry_time", Inf)
        defDim(ds, "one", 1)

        att(long_name, units) = ["long_name" => long_name, "units" => units]
        defVar(ds, "theta_s", Float64, ("one",); attrib = att("S-coordinate surface control parameter", "nondimensional"))
        defVar(ds, "theta_b", Float64, ("one",); attrib = att("S-coordinate bottom control parameter", "nondimensional"))
        defVar(ds, "hc", Float64, ("one",); attrib = att("S-coordinate parameter, critical depth", "meter"))
        defVar(ds, "bry_time", Float64, ("bry_time",); attrib = att("time for boundary data", "day"))

        for bnd in 1:4
            obcflag[bnd] == 1 || continue
            side, long = _BRY_NAMES[bnd], _BRY_LONG[bnd]
            # south/north boundaries run along xi, east/west along eta
            rho, udim, vdim = isodd(bnd) ? ("xi_rho", "xi_u", "xi_rho") : ("eta_rho", "eta_rho", "eta_v")
            defVar(ds, "temp_$side", Float32, (rho, "s_rho", "bry_time"); attrib = att("$long boundary potential temperature", "Celsius"))
            defVar(ds, "salt_$side", Float32, (rho, "s_rho", "bry_time"); attrib = att("$long boundary salinity", "PSU"))
            defVar(ds, "u_$side", Float32, (udim, "s_rho", "bry_time"); attrib = att("$long boundary u-momentum component", "meter second-1"))
            defVar(ds, "v_$side", Float32, (vdim, "s_rho", "bry_time"); attrib = att("$long boundary v-momentum component", "meter second-1"))
            defVar(ds, "ubar_$side", Float32, (udim, "bry_time"); attrib = att("$long boundary vertically integrated u-momentum component", "meter second-1"))
            defVar(ds, "vbar_$side", Float32, (vdim, "bry_time"); attrib = att("$long boundary vertically integrated v-momentum component", "meter second-1"))
            defVar(ds, "zeta_$side", Float32, (rho, "bry_time"); attrib = att("$long boundary sea surface height", "meter"))
        end

        # global attributes
        ds.attrib["title"] = "Boundary file produced by r2r"
        ds.attrib["date"] = matlab_date()
        ds.attrib["grd_file"] = grdname
        ds.attrib["type"] = "BOUNDARY file"
        ds.attrib["history"] = "ROMS"
        ds.attrib["VertCoordType"] = "NEW"

        ds["theta_s"][1] = param.theta_s
        ds["theta_b"][1] = param.theta_b
        ds["hc"][1] = param.hc
    end
    return nothing
end

# child index ranges (i0:i1 in xi, j0:j1 in eta) of the 2-point-wide strip along boundary bnd
function _bry_strip(bnd::Integer, mpc::Integer, npc::Integer)
    bnd == 1 && return 1, npc, 1, 2                 # South
    bnd == 2 && return npc - 1, npc, 1, mpc         # East
    bnd == 3 && return 1, npc, mpc - 1, mpc         # North
    return 1, 2, 1, mpc                             # West
end

"""
    h2r_bry_subgrid(parentgrid, childgrid, obcflag) -> limits

Lower and upper index in i and j of the minimal parent subgrid that
contains each open boundary strip of the child grid: `limits[bnd, :] =
[imin, imax, jmin, jmax]` (zeros for a closed boundary).
"""
function h2r_bry_subgrid(parentgrid::AbstractString, childgrid::AbstractString, obcflag)
    Lonc = wrap360(permutedims(ncread(childgrid, "lon_rho")))
    Latc = permutedims(ncread(childgrid, "lat_rho"))
    lonp = wrap360(permutedims(ncread(parentgrid, "Longitude")))
    latp = permutedims(ncread(parentgrid, "Latitude"))
    mpc, npc = size(Lonc)

    limits = zeros(Int, 4, 4)
    for bnd in 1:4
        obcflag[bnd] == 1 || continue
        i0, i1, j0, j1 = _bry_strip(bnd, mpc, npc)
        limits[bnd, :] .= _parent_index_range(lonp, latp, Lonc[j0:j1, i0:i1], Latc[j0:j1, i0:i1];
            label = "h2r_bry_subgrid")
    end
    return limits
end

# Everything about one boundary that does not change from one time step
# to the next: child strip geometry, parent subgrid, interpolation coefficients.
function _bry_setup(pargrd, chdgrd, chdscd, Np, limits, bnd, masks)
    mpc, npc = reverse(ncsize(chdgrd, "h"))
    i0, i1, j0, j1 = _bry_strip(bnd, mpc, npc)
    imin, imax, jmin, jmax = limits[bnd, :]
    li, lj = length(imin:imax), length(jmin:jmax)
    lic, ljc = length(i0:i1), length(j0:j1)

    # Get topography data from childgrid
    hc = _read2d_grid(chdgrd, "h", [i0, j0], [lic, ljc])
    maskc = _read2d_grid(chdgrd, "mask_rho", [i0, j0], [lic, ljc])
    angc = _read2d_grid(chdgrd, "angle", [i0, j0], [lic, ljc])
    lonc = wrap360(_read2d_grid(chdgrd, "lon_rho", [i0, j0], [lic, ljc]))
    latc = _read2d_grid(chdgrd, "lat_rho", [i0, j0], [lic, ljc])
    Mc, Lc = size(maskc)
    maskc3d = reshape(maskc, 1, Mc, Lc)
    umask = maskc3d[:, :, 2:end] .* maskc3d[:, :, 1:end-1]
    vmask = maskc3d[:, 2:end, :] .* maskc3d[:, 1:end-1, :]

    # Parent minimal subgrid
    lons = wrap360(_read2d_grid(pargrd, "Longitude", [imin, jmin], [li, lj]))
    lats = _read2d_grid(pargrd, "Latitude", [imin, jmin], [li, lj])

    # Z-coordinate (3D) on minimal subgrid and child grid
    zs = ncom_zgrid(dropdims(ncread(pargrd, "layer_thickness", [imin, jmin, 1, 1], [li, lj, Np, 1]); dims = 4))
    zc, _ = zlevs3(hc, hc .* 0, chdscd.theta_s, chdscd.theta_b, chdscd.hc, chdscd.N, "r", chdscd.scoord)
    zw, _ = zlevs3(hc, hc .* 0, chdscd.theta_s, chdscd.theta_b, chdscd.hc, chdscd.N, "w", chdscd.scoord)

    println("Computing interpolation coefficients")
    elem2d, coef2d, nnel = get_tri_coef(lons, lats, lonc, latc, masks)
    A = get_hv_coef(zs, zc, coef2d, elem2d, lons, lats, lonc, latc)

    # Prepare for estimating barotropic velocity
    dz = zw[2:end, :, :] .- zw[1:end-1, :, :]
    dzu = 0.5 .* (dz[:, :, 1:end-1] .+ dz[:, :, 2:end])
    dzv = 0.5 .* (dz[:, 1:end-1, :] .+ dz[:, 2:end, :])

    return (masks = masks, maskc = maskc, maskc3d = maskc3d, umask = umask, vmask = vmask,
            cosc = reshape(cos.(angc), 1, Mc, Lc), sinc = reshape(sin.(angc), 1, Mc, Lc),
            elem2d = elem2d, coef2d = coef2d, nnel = nnel, A = A, dzu = dzu, dzv = dzv,
            inpaint = Dict{Symbol,Any}())
end

"""
    h2r_bry_hv(pargrd, chdgrd, parinie, parinit, pariniu, Np, bry_filename, chdscd, obcflag, limits, ii; cache = Dict())

Interpolate parent time step `ii` onto the open boundaries of the child
grid and write it into time index `ii` of `bry_filename`: temp, salt, u,
v, ubar, vbar and zeta for each side.

The interpolation coefficients of a boundary only depend on the grids
and the parent land mask, not on the time step. Pass the same `cache`
Dict to every call for one boundary file and they are computed once per
boundary instead of once per time step (they are rebuilt if the parent
mask ever changes); the MATLAB version recomputed them every call.
"""
function h2r_bry_hv(pargrd, chdgrd, parinie, parinit, pariniu, Np, bry_filename, chdscd, obcflag, limits, ii;
                    cache::AbstractDict = Dict{Int,Any}())
    # Set bry_time. Like the MATLAB routine this always uses the first MT
    # value; bry_build rewrites bry_time with the right one for every step.
    t0 = ncread(pariniu, "MT", [1], [1])[1]
    ocean_time = t0 + _T1 - _T2
    tind = ii
    tout = ii
    ncwrite(bry_filename, "bry_time", ocean_time, tout)

    for bnd in 1:4
        println("-------------------------------------------------------------")
        if obcflag[bnd] != 1
            println("Closed boundary")
            continue
        end
        side = _BRY_NAMES[bnd]
        println(uppercasefirst(side), " boundary")

        # Compute minimal subgrid extracted from parent grid
        imin, imax, jmin, jmax = limits[bnd, :]
        li, lj = length(imin:imax), length(jmin:jmax)

        # Parent land mask on the subgrid, from the surface elevation
        etas = _read2d(parinie, "ssh", [imin, jmin, tind], [li, lj, 1])
        masks = ones(size(etas))
        masks[isnan.(etas)] .= 0

        if !haskey(cache, bnd) || cache[bnd].masks != masks
            cache[bnd] = _bry_setup(pargrd, chdgrd, chdscd, Np, limits, bnd, masks)
        end
        s = cache[bnd]
        elem2d, coef2d, nnel, A = s.elem2d, s.coef2d, s.nnel, s.A

        # Surface elevation on minimal subgrid and child grid
        zetas = fillmask(etas, 1, masks, nnel)
        zetac = apply_tri_coef(elem2d, coef2d, zetas) .* s.maskc

        # pick the boundary row/column out of a (.., eta, xi) strip
        edge(a) = bnd == 1 ? selectdim(a, ndims(a) - 1, 1) :
                  bnd == 2 ? selectdim(a, ndims(a), size(a, ndims(a))) :
                  bnd == 3 ? selectdim(a, ndims(a) - 1, size(a, ndims(a) - 1)) :
                             selectdim(a, ndims(a), 1)

        # Process scalar 3D variables
        for (svar, svarh) in (("layer_temperature", "temp"), ("layer_salinity", "salt"))
            println("--- ", svar)
            fld = _read3d_flipped(parinit, svar, [imin, jmin, 1, tind], [li, lj, Np, 1])
            fld = inpaint_nans(fillmask(fld, 1, masks, nnel), 4; cache = s.inpaint)
            fld = fillmissing_linear!(apply_hv_coef(A, fld), 2)
            fld .*= s.maskc3d   # zero-ing out masked areas
            ncwrite(bry_filename, "$(svarh)_$side", permutedims(edge(fld)), [1, 1, tout])
        end

        # Read in velocities one extra point to the west/south and average
        # neighboring points back onto the subgrid
        ud = _read3d_flipped(pariniu, "u_velocity", [imin - 1, jmin, 1, tind], [length(imin-1:imax), lj, Np, 1])
        ur = 0.5 .* (ud[:, :, 1:end-1] .+ ud[:, :, 2:end])
        vd = _read3d_flipped(pariniu, "v_velocity", [imin, jmin - 1, 1, tind], [li, length(jmin-1:jmax), Np, 1])
        vr = 0.5 .* (vd[:, 1:end-1, :] .+ vd[:, 2:end, :])

        # 3d interpolation of us and vs to child grid
        ur[isnan.(ur)] .= 0
        vr[isnan.(vr)] .= 0
        ud = apply_hv_coef(A, fillmask(ur, 0, masks, nnel))
        vd = apply_hv_coef(A, fillmask(vr, 0, masks, nnel))

        # Rotate to child orientation
        us = ud .* s.cosc .+ vd .* s.sinc
        vs = vd .* s.cosc .- ud .* s.sinc

        # back to staggered u and v points
        u = 0.5 .* (us[:, :, 1:end-1] .+ us[:, :, 2:end]) .* s.umask
        v = 0.5 .* (vs[:, 1:end-1, :] .+ vs[:, 2:end, :]) .* s.vmask

        # Get barotropic velocity
        any(isnan, u) && error("nans in u velocity!")
        any(isnan, v) && error("nans in v velocity!")
        ubar = dropdims(sum(s.dzu .* u; dims = 1) ./ sum(s.dzu; dims = 1); dims = 1)
        vbar = dropdims(sum(s.dzv .* v; dims = 1) ./ sum(s.dzv; dims = 1); dims = 1)

        # Save perimeter zeta, ubar, vbar, u and v data to bryfile
        ncwrite(bry_filename, "ubar_$side", collect(edge(ubar)), [1, tout])
        ncwrite(bry_filename, "vbar_$side", collect(edge(vbar)), [1, tout])
        ncwrite(bry_filename, "zeta_$side", collect(edge(zetac)), [1, tout])
        ncwrite(bry_filename, "u_$side", permutedims(edge(u)), [1, 1, tout])
        ncwrite(bry_filename, "v_$side", permutedims(edge(v)), [1, 1, tout])
    end    # End loop bnd
    return nothing
end
