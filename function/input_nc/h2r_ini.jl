# h2r_ini.jl
#
# ROMS initial file from the parent (NCOM) NetCDF files.
# Ported from h2r_GitHub/h2r_create_ini.m and h2r_make_ini.m, plus the
# helpers the ini/bry/frc routines share.
#
# Inside these routines 2D fields are (eta, xi) and 3D fields (s, eta, xi)
# — the layout the MATLAB code works in after transposing what ncread
# returns — so the index arithmetic carries over unchanged.

# MT is in days since 1900-12-31; ROMS time is counted from 1994-01-01
const _T1 = datenum(1900, 12, 31, 0, 0, 0)
const _T2 = datenum(1994, 1, 1, 0, 0, 0)

"""
    ncom_zgrid(zstt) -> zs

Depths of the parent layer centers from the layer thicknesses `zstt`
(xi, eta, Np) (top layer first): returns (Np, eta, xi), negative
downward, in ascending order along k (deepest first). NaN thicknesses
count as zero. Port of NCOM_zgrid.m.
"""
function ncom_zgrid(zstt::AbstractArray{<:Real,3})
    zst = permutedims(zstt, (3, 2, 1))
    zst[isnan.(zst)] .= 0
    zs = -cumsum(zst; dims = 1) .+ 0.5 .* zst
    return reverse(zs; dims = 1)
end

# ncread a (xi, eta[, time]) slab and return it as (eta, xi)
_read2d(file, var, start, count) = permutedims(dropdims(ncread(file, var, start, count); dims = 3))
_read2d_grid(file, var, start, count) = permutedims(ncread(file, var, start, count))

# ncread a (xi, eta, z, time) block and return it as (z, eta, xi), deepest level first
function _read3d_flipped(file, var, start, count)
    v = dropdims(ncread(file, var, start, count); dims = 4)
    return reverse(permutedims(v, (3, 2, 1)); dims = 1)
end

# i/j index range of the parent grid (Mp,Lp) spanned by the triangles
# that contain the child points: (imin, imax, jmin, jmax)
function _parent_index_range(lonp::AbstractMatrix, latp::AbstractMatrix, lonc, latc; label)
    elem, _ = tsearch_fix(lonp, latp, lonc, latc; label = label)
    ci = CartesianIndices(size(lonp))
    imin, imax, jmin, jmax = typemax(Int), 0, typemax(Int), 0
    for e in elem
        idxj, idxi = Tuple(ci[e])
        imin, imax = min(imin, idxi), max(imax, idxi)
        jmin, jmax = min(jmin, idxj), max(jmax, idxj)
    end
    return imin, imax, jmin, jmax
end

wrap360(lon) = map(x -> x < 0 ? x + 360 : x, lon)

"""
    h2r_create_ini(ininame, grdname, N, chdscd)

Create an empty ROMS initial file for the grid `grdname` with `N`
s-levels. `chdscd` holds theta_s, theta_b, hc. An existing file is replaced.
"""
function h2r_create_ini(ininame::AbstractString, grdname::AbstractString, N::Integer, chdscd)
    Lp, Mp = ncsize(grdname, "mask_rho")
    L = Lp - 1
    M = Mp - 1
    Np = N + 1

    isfile(ininame) && rm(ininame)
    NCDataset(ininame, "c"; format = :netcdf4) do ds
        defDim(ds, "xi_u", L)
        defDim(ds, "xi_v", Lp)
        defDim(ds, "xi_rho", Lp)
        defDim(ds, "eta_u", Mp)
        defDim(ds, "eta_v", M)
        defDim(ds, "eta_rho", Mp)
        defDim(ds, "s_rho", N)
        defDim(ds, "s_w", Np)
        defDim(ds, "tracer", 2)
        defDim(ds, "time", 1)
        defDim(ds, "one", 1)

        att(long_name, units) = ["long_name" => long_name, "units" => units]
        defVar(ds, "tstart", Float32, ("one",); attrib = att("start processing day", "day"))
        defVar(ds, "tend", Float32, ("one",); attrib = att("end processing day", "day"))
        defVar(ds, "theta_s", Float64, ("one",); attrib = att("S-coordinate surface control parameter", "nondimensional"))
        defVar(ds, "theta_b", Float64, ("one",); attrib = att("S-coordinate bottom control parameter", "nondimensional"))
        defVar(ds, "Tclinec", Float64, ("one",); attrib = att("S-coordinate surface/bottom layer width", "meter"))
        defVar(ds, "hc", Float64, ("one",); attrib = att("S-coordinate parameter, critical depth", "meter"))
        defVar(ds, "sc_r", Float64, ("s_rho",))
        defVar(ds, "Cs_r", Float64, ("s_rho",))
        defVar(ds, "Cs_w", Float64, ("s_w",))
        defVar(ds, "ocean_time", Float64, ("time",); attrib = att("time since initialization", "second"))

        defVar(ds, "u", Float32, ("xi_u", "eta_u", "s_rho", "time"); attrib = att("u-momentum component", "meter second-1"))
        defVar(ds, "v", Float32, ("xi_v", "eta_v", "s_rho", "time"); attrib = att("v-momentum component", "meter second-1"))
        defVar(ds, "ubar", Float32, ("xi_u", "eta_u", "time"); attrib = att("vertically integrated u-momentum component", "meter second-1"))
        defVar(ds, "vbar", Float32, ("xi_v", "eta_v", "time"); attrib = att("vertically integrated v-momentum component", "meter second-1"))
        defVar(ds, "zeta", Float32, ("xi_rho", "eta_rho", "time"); attrib = att("sea surface height", "meter"))
        defVar(ds, "temp", Float32, ("xi_rho", "eta_rho", "s_rho", "time"); attrib = att("potential temperature", "Celsius"))
        defVar(ds, "salt", Float32, ("xi_rho", "eta_rho", "s_rho", "time"); attrib = att("salinity", "PSU"))

        # global attributes
        ds.attrib["title"] = "Initial file produced by h2r"
        ds.attrib["date"] = matlab_date()
        ds.attrib["clim_file"] = ininame
        ds.attrib["grd_file"] = grdname
        ds.attrib["type"] = "hycom2roms initial file"
        ds.attrib["history"] = "none"
        ds.attrib["hc"] = Float64(chdscd.hc)
        ds.attrib["VertCoordType"] = "NEW"

        ds["tstart"][1] = 1.0
        ds["tend"][1] = 1.0
        ds["theta_b"][1] = chdscd.theta_b
        ds["theta_s"][1] = chdscd.theta_s
        ds["Tclinec"][1] = chdscd.hc
        ds["hc"][1] = chdscd.hc
        ds["ocean_time"][1] = 1.0 * 24 * 3600
    end
    return nothing
end

"""
    h2r_make_ini(pargrd, par_tind, pariniu, pariniv, parinit, parinis, parinie,
                 chd_grd, chd_data, chdscd, chdscoord, ndomx, ndomy, chd_ang, par_N)

Fill the initial file `chd_data` (from `h2r_create_ini`) by interpolating
parent time index `par_tind` onto the child grid `chd_grd`, working in
ndomx x ndomy child chunks.

- `pargrd`  : parent file with Longitude, Latitude, layer_thickness
- `pariniu/pariniv` : parent files with u_velocity / v_velocity (+ MT in pariniu)
- `parinit/parinis` : parent files with layer_temperature / layer_salinity
- `parinie` : parent file with ssh
- `chdscd`  : child s-coordinate (N, theta_s, theta_b, hc); `chdscoord` e.g. "new2008"
- `chd_ang` : "rad" or "deg", unit of the child grid's `angle`
- `par_N`   : number of parent vertical levels
"""
function h2r_make_ini(pargrd, par_tind, pariniu, pariniv, parinit, parinis, parinie,
                      chd_grd, chd_data, chdscd, chdscoord, ndomx, ndomy, chd_ang, par_N)
    # S-coordinate params for child grid
    N_c = chdscd.N
    theta_b_c = chdscd.theta_b
    theta_s_c = chdscd.theta_s
    hc_c = chdscd.hc

    Np = par_N
    tind = par_tind

    # Set correct time in ini file
    t0 = ncread(pariniu, "MT", [tind], [1])[1]
    ocean_time = (t0 + _T1 - _T2) * 24 * 60 * 60   # time in seconds
    ncwrite(chd_data, "ocean_time", ocean_time, 1)

    # Full parent grid
    lonp = wrap360(permutedims(ncread(pargrd, "Longitude")))
    latp = permutedims(ncread(pargrd, "Latitude"))
    Mpp, Lpp = size(latp)

    # Child grid and chunk size
    Lp_c, Mp_c = ncsize(chd_grd, "h")
    icmin, icmax = _chunk_bounds(Lp_c, ndomx)
    jcmin, jcmax = _chunk_bounds(Mp_c, ndomy)

    # Do the interpolation for all child chunks
    for domx in 1:ndomx, domy in 1:ndomy
        println("chunk domx=", domx, " domy=", domy)
        icb, ice = icmin[domx], icmax[domx]
        jcb, jce = jcmin[domy], jcmax[domy]
        li = length(icb:ice)
        lj = length(jcb:jce)

        # Get topography data from childgrid
        hc = _read2d_grid(chd_grd, "h", [icb, jcb], [li, lj])
        maskc = _read2d_grid(chd_grd, "mask_rho", [icb, jcb], [li, lj])
        lonc = wrap360(_read2d_grid(chd_grd, "lon_rho", [icb, jcb], [li, lj]))
        latc = _read2d_grid(chd_grd, "lat_rho", [icb, jcb], [li, lj])
        angc = _read2d_grid(chd_grd, "angle", [icb, jcb], [li, lj])
        chd_ang == "deg" && (angc = pi / 180.0 .* angc)
        umask = maskc[:, 1:end-1] .* maskc[:, 2:end]
        vmask = maskc[1:end-1, :] .* maskc[2:end, :]
        cosc = cos.(angc)
        sinc = sin.(angc)

        # Compute minimal subgrid extracted from full parent grid
        imin, imax, jmin, jmax = _parent_index_range(lonp, latp, lonc, latc; label = "h2r_make_ini")
        imin = max(1, imin - 1)
        imax = min(imax + 1, Lpp)
        jmin = max(1, jmin - 1)
        jmax = min(jmax + 1, Mpp)
        lpi = length(imin:imax)
        lpj = length(jmin:jmax)

        # Get parent grid and squeeze minimal subgrid
        etas = _read2d(parinie, "ssh", [imin, jmin, tind], [lpi, lpj, 1])
        masks = ones(size(etas))
        masks[isnan.(etas)] .= 0
        lons = wrap360(lonp[jmin:jmax, imin:imax])
        lats = latp[jmin:jmax, imin:imax]

        # Z-coordinate (3D) on minimal subgrid and child grid
        zs = ncom_zgrid(dropdims(ncread(pargrd, "layer_thickness", [imin, jmin, 1, tind], [lpi, lpj, Np, 1]); dims = 4))
        zc, Cs_r = zlevs3(hc, hc .* 0, theta_s_c, theta_b_c, hc_c, N_c, "r", chdscoord)
        zw, Cs_w = zlevs3(hc, hc .* 0, theta_s_c, theta_b_c, hc_c, N_c, "w", chdscoord)
        _, Mc, Lc = size(zc)

        println("    Computing interpolation coefficients")
        elem2d, coef2d, nnel = get_tri_coef(lons, lats, lonc, latc, masks)
        A = get_hv_coef(zs, zc, coef2d, elem2d, lons, lats, lonc, latc)

        println("    => zeta")
        zetas = fillmask(etas, 1, masks, nnel)
        zetac = apply_tri_coef(elem2d, coef2d, zetas) .* maskc

        inpaint_cache = Dict{Symbol,Any}()   # temp and salt share one NaN pattern
        println("    => temp")
        fld = _read3d_flipped(parinit, "layer_temperature", [imin, jmin, 1, tind], [lpi, lpj, Np, 1])
        fld = inpaint_nans(fillmask(fld, 1, masks, nnel), 4; cache = inpaint_cache)
        ini_temp = fillmissing_linear!(apply_hv_coef(A, fld), 2)

        println("    => salt")
        fld = _read3d_flipped(parinis, "layer_salinity", [imin, jmin, 1, tind], [lpi, lpj, Np, 1])
        fld = inpaint_nans(fillmask(fld, 1, masks, nnel), 4; cache = inpaint_cache)
        ini_salt = fillmissing_linear!(apply_hv_coef(A, fld), 2)

        # Read in velocities (already at parent cell centers, true east/north)
        println("    => total velocity")
        us = _read3d_flipped(pariniu, "u_velocity", [imin, jmin, 1, tind], [lpi, lpj, Np, 1])
        vs = _read3d_flipped(pariniv, "v_velocity", [imin, jmin, 1, tind], [lpi, lpj, Np, 1])
        us[isnan.(us)] .= 0
        vs[isnan.(vs)] .= 0
        ud = apply_hv_coef(A, fillmask(us, 0, masks, nnel))
        vd = apply_hv_coef(A, fillmask(vs, 0, masks, nnel))

        # Rotate to child orientation
        cos3 = reshape(cosc, 1, Mc, Lc)
        sin3 = reshape(sinc, 1, Mc, Lc)
        us = ud .* cos3 .+ vd .* sin3
        vs = vd .* cos3 .- ud .* sin3

        # Back to staggered locations
        u = 0.5 .* (us[:, :, 1:Lc-1] .+ us[:, :, 2:Lc])
        v = 0.5 .* (vs[:, 1:Mc-1, :] .+ vs[:, 2:Mc, :])

        # Get barotropic velocity
        println("    => barotropic velocity")
        dz = zw[2:end, :, :] .- zw[1:end-1, :, :]
        dzu = 0.5 .* (dz[:, :, 1:end-1] .+ dz[:, :, 2:end])
        dzv = 0.5 .* (dz[:, 1:end-1, :] .+ dz[:, 2:end, :])
        ubar = dropdims(sum(dzu .* u; dims = 1) ./ sum(dzu; dims = 1); dims = 1) .* umask
        vbar = dropdims(sum(dzv .* v; dims = 1) ./ sum(dzv; dims = 1); dims = 1) .* vmask

        # Zero-ing out the mask
        ini_temp .*= reshape(maskc, 1, Mc, Lc)
        ini_salt .*= reshape(maskc, 1, Mc, Lc)
        u .*= reshape(umask, 1, Mc, Lc - 1)
        v .*= reshape(vmask, 1, Mc - 1, Lc)

        println(">>> Writing ini file")
        ncwrite(chd_data, "Cs_w", Cs_w)
        ncwrite(chd_data, "Cs_r", Cs_r)
        ncwrite(chd_data, "temp", permutedims(ini_temp, (3, 2, 1)), [icb, jcb, 1, 1])
        ncwrite(chd_data, "salt", permutedims(ini_salt, (3, 2, 1)), [icb, jcb, 1, 1])
        ncwrite(chd_data, "u", permutedims(u, (3, 2, 1)), [icb, jcb, 1, 1])
        ncwrite(chd_data, "v", permutedims(v, (3, 2, 1)), [icb, jcb, 1, 1])
        ncwrite(chd_data, "zeta", permutedims(zetac), [icb, jcb, 1])
        ncwrite(chd_data, "ubar", permutedims(ubar), [icb, jcb, 1])
        ncwrite(chd_data, "vbar", permutedims(vbar), [icb, jcb, 1])
    end
    return nothing
end
