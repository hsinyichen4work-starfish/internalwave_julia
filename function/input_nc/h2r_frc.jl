# h2r_frc.jl
#
# ROMS surface forcing file (fluxes, not bulk formula) from the parent
# (NCOM) NetCDF files. Ported from h2r_GitHub/h2r_create_frc.m,
# h2r_frc_subgrid.m and h2r_make_frc.m.

"""
    h2r_create_frc(frcname, grdname)

Create an empty ROMS surface forcing file with
sustr, svstr (wind stress, u/v points), shflux (net surface heat flux,
solar included), swflux (surface freshwater flux), swrad (shortwave
radiation) and Pair (sea level pressure), each on its own unlimited time
dimension. An existing file is replaced.
"""
function h2r_create_frc(frcname::AbstractString, grdname::AbstractString)
    Lp, Mp = ncsize(grdname, "mask_rho")
    L = Lp - 1   # xi_u  size
    M = Mp - 1   # eta_v size

    isfile(frcname) && rm(frcname)
    NCDataset(frcname, "c"; format = :netcdf4) do ds
        defDim(ds, "xi_u", L)
        defDim(ds, "xi_v", Lp)
        defDim(ds, "xi_rho", Lp)
        defDim(ds, "eta_u", Mp)
        defDim(ds, "eta_v", M)
        defDim(ds, "eta_rho", Mp)
        for t in ("sms_time", "shf_time", "swf_time", "srf_time", "pair_time")
            defDim(ds, t, Inf)
        end

        att(long_name, units) = ["long_name" => long_name, "units" => units]
        # Time variables
        defVar(ds, "sms_time", Float64, ("sms_time",); attrib = att("surface momentum stress time", "day"))
        defVar(ds, "shf_time", Float64, ("shf_time",); attrib = att("surface net heat flux time", "day"))
        defVar(ds, "swf_time", Float64, ("swf_time",); attrib = att("surface salt/freshwater flux time", "day"))
        defVar(ds, "srf_time", Float64, ("srf_time",); attrib = att("surface shortwave radiation time", "day"))
        defVar(ds, "pair_time", Float64, ("pair_time",); attrib = att("surface air pressure time", "day"))

        # Wind stress (staggered u/v points)
        defVar(ds, "sustr", Float32, ("xi_u", "eta_u", "sms_time"); attrib = att("surface u-momentum stress", "Newton meter-2"))
        defVar(ds, "svstr", Float32, ("xi_v", "eta_v", "sms_time"); attrib = att("surface v-momentum stress", "Newton meter-2"))
        # Net surface heat flux, rho points
        defVar(ds, "shflux", Float32, ("xi_rho", "eta_rho", "shf_time"); attrib = att("surface net heat flux", "Watt meter-2"))
        # Surface salt/freshwater flux, rho points
        defVar(ds, "swflux", Float32, ("xi_rho", "eta_rho", "swf_time"); attrib = att("surface freshwater flux (E-P)", "centimeter day-1"))
        # Shortwave radiation, rho points
        defVar(ds, "swrad", Float32, ("xi_rho", "eta_rho", "srf_time");
            attrib = vcat(att("solar shortwave radiation", "Watt meter-2"),
                ["positive_value" => "downward flux, heating", "negative_value" => "upward flux, cooling"]))
        # Sea level pressure, rho points (optional)
        defVar(ds, "Pair", Float32, ("xi_rho", "eta_rho", "pair_time"); attrib = att("surface air pressure", "millibar"))

        # Global attributes
        ds.attrib["title"] = "Surface forcing file produced by n2r (NCOM to ROMS)"
        ds.attrib["date"] = matlab_date()
        ds.attrib["grd_file"] = grdname
        ds.attrib["type"] = "FORCING file"
        ds.attrib["history"] = "ROMS"
    end
    return nothing
end

"""
    h2r_frc_subgrid(parentgrid, childgrid, ndomx, ndomy) -> limits

Lower and upper index in i and j of the minimal parent subgrid that
contains each (domx,domy) chunk of the child grid. `limits[domx, domy]`
is a NamedTuple (imin, imax, jmin, jmax, icb, ice, jcb, jce).

The geometry doesn't change across time steps or dates, only the data
does, so `limits` can be reused for every call to `h2r_make_frc` with
the same grids and chunking.
"""
function h2r_frc_subgrid(parentgrid::AbstractString, childgrid::AbstractString, ndomx::Integer, ndomy::Integer)
    Lonc = wrap360(permutedims(ncread(childgrid, "lon_rho")))
    Latc = permutedims(ncread(childgrid, "lat_rho"))
    lonp = wrap360(permutedims(ncread(parentgrid, "Longitude")))
    latp = permutedims(ncread(parentgrid, "Latitude"))
    Mp, Lp = size(lonp)

    Mc, Lc = size(Lonc)
    icmin, icmax = _chunk_bounds(Lc, ndomx)
    jcmin, jcmax = _chunk_bounds(Mc, ndomy)

    return map(Iterators.product(1:ndomx, 1:ndomy)) do (domx, domy)
        icb, ice = icmin[domx], icmax[domx]
        jcb, jce = jcmin[domy], jcmax[domy]
        imin, imax, jmin, jmax = _parent_index_range(lonp, latp, Lonc[jcb:jce, icb:ice], Latc[jcb:jce, icb:ice];
            label = "h2r_frc_subgrid (chunk domx=$domx domy=$domy)")
        (imin = max(1, imin - 1), imax = min(Lp, imax + 1), jmin = max(1, jmin - 1), jmax = min(Mp, jmax + 1),
         icb = icb, ice = ice, jcb = jcb, jce = jce)
    end
end

"""
    h2r_make_frc(parent_G, parent_FLUX, parent_WIND, parent_PRESS, chdgrd, frcname, chd_ang, limits, parent_TS)

Interpolate the parent surface forcing (heaflx/salflx/solflx,
stresu/stresv, slpres) horizontally onto the child grid and write every
available parent time step into `frcname`.

UNITS: NCOM writes kinematic fluxes, ROMS flux_frc.F expects physical
ones, so they are converted here:

    shflux [W/m^2]  = (heaflx + solflx) * rho0*Cp
                      -- heaflx [K m/s] is the NON-solar part only, but
                      ROMS wants the NET flux (it subtracts swrad from it)
    swrad  [W/m^2]  = solflx * rho0*Cp
    swflux [cm/day] = salflx / SSS * 8.64e6
                      -- salflx [psu m/s] is already S*(E-P); ROMS
                      multiplies swflux by its own surface salinity

`parent_TS` is the `<par_name>_ts.nc` file, used only for its surface
salinity (layer 1 = surface). `limits` comes from `h2r_frc_subgrid`.
"""
function h2r_make_frc(parent_G, parent_FLUX, parent_WIND, parent_PRESS, chdgrd, frcname, chd_ang, limits, parent_TS)
    ndomx, ndomy = size(limits)

    # --- Unit conversion constants ---
    # rho0 and Cp must match what ROMS divides by: rho0 in the roms .in file,
    # Cp in scalars.F. That way ROMS recovers exactly NCOM's kinematic flux.
    rho0 = 1027.5
    Cp = 3985
    ms2cmday = 100 * 86400      # m/s -> cm/day
    sss_min = 1                 # psu, floor on SSS to keep salflx/SSS bounded in the river plume

    # --- Child grid ---
    maskc_full = permutedims(ncread(chdgrd, "mask_rho"))
    lonc_full = wrap360(permutedims(ncread(chdgrd, "lon_rho")))
    latc_full = permutedims(ncread(chdgrd, "lat_rho"))
    angc_full = permutedims(ncread(chdgrd, "angle"))
    chd_ang == "deg" && (angc_full = pi / 180.0 .* angc_full)

    # --- Time base ---
    MT = ncread(parent_WIND, "MT")
    nt = length(MT)
    nt_ts = ncsize(parent_TS, "layer_salinity")[end]
    nt_ts == nt || error("$parent_TS has $nt_ts time steps but the forcing files have $nt.")

    for tind in 1:nt
        ocean_time = MT[tind] + _T1 - _T2   # days, matches h2r_create_frc units
        for tname in ("sms_time", "shf_time", "swf_time", "srf_time", "pair_time")
            ncwrite(frcname, tname, ocean_time, tind)
        end
    end

    for domx in 1:ndomx, domy in 1:ndomy
        println("-------------------------------------------------------------")
        println("  chunk domx=", domx, " domy=", domy)

        L = limits[domx, domy]
        icb, ice, jcb, jce = L.icb, L.ice, L.jcb, L.jce
        imin, imax, jmin, jmax = L.imin, L.imax, L.jmin, L.jmax
        li = length(imin:imax)
        lj = length(jmin:jmax)

        maskc = maskc_full[jcb:jce, icb:ice]
        angc = angc_full[jcb:jce, icb:ice]
        cosc = cos.(angc)
        sinc = sin.(angc)
        umask = maskc[:, 1:end-1] .* maskc[:, 2:end]
        vmask = maskc[1:end-1, :] .* maskc[2:end, :]
        lonc = lonc_full[jcb:jce, icb:ice]
        latc = latc_full[jcb:jce, icb:ice]

        lons = wrap360(_read2d_grid(parent_G, "Longitude", [imin, jmin], [li, lj]))
        lats = _read2d_grid(parent_G, "Latitude", [imin, jmin], [li, lj])

        # --- Parent land/sea mask on the subgrid (from first flux time step) ---
        heaflx1 = _read2d(parent_FLUX, "heaflx", [imin, jmin, 1], [li, lj, 1])
        masks = ones(size(heaflx1))
        masks[isnan.(heaflx1)] .= 0

        println("    parent subgrid: ", li, " x ", lj)
        println("    Computing interpolation coefficients")
        elem2d, coef2d, nnel = get_tri_coef(lons, lats, lonc, latc, masks)

        # parent scalar -> child rho points, masked, as (xi, eta) for writing
        to_child(fld) = permutedims(apply_tri_coef(elem2d, coef2d, fillmask(fld, 1, masks, nnel)) .* maskc)

        for tind in 1:nt
            println("    time step ", tind, " of ", nt)

            # --- Scalar fields: heaflx/salflx/solflx/slpres -> shflux/swflux/swrad/Pair ---
            hea = _read2d(parent_FLUX, "heaflx", [imin, jmin, tind], [li, lj, 1])
            sal = _read2d(parent_FLUX, "salflx", [imin, jmin, tind], [li, lj, 1])
            sol = _read2d(parent_FLUX, "solflx", [imin, jmin, tind], [li, lj, 1])
            sss = permutedims(dropdims(ncread(parent_TS, "layer_salinity", [imin, jmin, 1, tind], [li, lj, 1, 1]); dims = (3, 4)))
            # MATLAB's max(sss, sss_min) ignores NaN: a NaN salinity becomes sss_min
            sss = map(x -> isnan(x) ? float(sss_min) : max(x, sss_min), sss)

            # NCOM kinematic units -> ROMS units, see UNITS note in the docstring
            ncwrite(frcname, "shflux", to_child((hea .+ sol) .* rho0 .* Cp), [icb, jcb, tind])
            ncwrite(frcname, "swflux", to_child(sal ./ sss .* ms2cmday), [icb, jcb, tind])
            ncwrite(frcname, "swrad", to_child(sol .* rho0 .* Cp), [icb, jcb, tind])

            pres = _read2d(parent_PRESS, "slpres", [imin, jmin, tind], [li, lj, 1])
            ncwrite(frcname, "Pair", to_child(pres), [icb, jcb, tind])

            # --- Vector field: stresu/stresv -> sustr/svstr ---
            us = _read2d(parent_WIND, "stresu", [imin, jmin, tind], [li, lj, 1])
            vs = _read2d(parent_WIND, "stresv", [imin, jmin, tind], [li, lj, 1])
            us[isnan.(us)] .= 0
            vs[isnan.(vs)] .= 0
            ud = apply_tri_coef(elem2d, coef2d, fillmask(us, 0, masks, nnel))
            vd = apply_tri_coef(elem2d, coef2d, fillmask(vs, 0, masks, nnel))

            # Rotate from earth-relative (true east/north) to child grid orientation
            u_rho = ud .* cosc .+ vd .* sinc
            v_rho = vd .* cosc .- ud .* sinc

            # Average from rho-like points to ROMS staggered u/v points
            sustr = 0.5 .* (u_rho[:, 1:end-1] .+ u_rho[:, 2:end]) .* umask
            svstr = 0.5 .* (v_rho[1:end-1, :] .+ v_rho[2:end, :]) .* vmask

            ncwrite(frcname, "sustr", permutedims(sustr), [icb, jcb, tind])
            ncwrite(frcname, "svstr", permutedims(svstr), [icb, jcb, tind])
        end   # time loop
    end   # domx, domy

    println(">>> Finished writing frc file")
    return nothing
end
