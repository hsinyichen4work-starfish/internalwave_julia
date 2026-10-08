# dbry_fun.jl
#
# Ported from dbry_make.m — the functions behind input_file_make/dbry_make.jl:
# the high-frequency baroclinic energy flux Fx/Fy at the child-grid open
# boundaries, computed from the daily roms_bry_<dx>m_<fod>.nc files and
# written out as one daily flux netCDF file (+ one daily MAT file) each.
#
# All fields are (nalong, nz, nt) or (nalong, nt), same layout as in MATLAB.

using NCDatasets, DSP, GibbsSeaWater, MAT, Statistics, Dates, Printf

include(joinpath(@__DIR__, "zlevs3.jl"))

const DBRY_DIRS = ("north", "south", "east", "west")
const DBRY_G = 9.81

# boundary direction -> (output varname, source flux field, along-boundary
# dimension name). The dimension name must be exactly eta_rho / xi_rho for
# partit to recognize and decompose it, same as zeta_west/zeta_south etc.
const DBRY_FLUX_VARS = Dict(
    "west"  => (varname = "up_west",  src = "Fx", dim = "eta_rho"),
    "east"  => (varname = "up_east",  src = "Fx", dim = "eta_rho"),
    "south" => (varname = "vp_south", src = "Fy", dim = "xi_rho"),
    "north" => (varname = "vp_north", src = "Fy", dim = "xi_rho"),
)

# fields that have time as their last dimension (everything except rho_bar)
const DBRY_SMALL = ("ssh", "p_bar", "u_bar", "v_bar", "Fx", "Fy")
const DBRY_BIG   = ("p_prime", "u_prime", "v_prime", "p_fast", "u_fast", "v_fast")

ncfloat(ds, name) = Float64.(coalesce.(Array(ds[name]), NaN))

"""
    dbry_geo(grid_file) -> Dict(direction => (; bathc, angc, lon, lat))

Depth, grid angle and lon/lat along each of the four open boundaries.
"""
function dbry_geo(grid_file)
    h, angle, lon, lat = NCDataset(grid_file) do ds
        ncfloat(ds, "h"), ncfloat(ds, "angle"), ncfloat(ds, "lon_rho"), ncfloat(ds, "lat_rho")
    end
    lon[lon .> 180] .-= 360
    edge(a, d) = d == "south" ? a[:, 1] : d == "north" ? a[:, end] : d == "east" ? a[end, :] : a[1, :]
    return Dict(d => (bathc = edge(h, d), angc = edge(angle, d), lon = edge(lon, d), lat = edge(lat, d))
                for d in DBRY_DIRS)
end

"""
    read_padded_window(bry_file, fod, p_start, p_end) -> (time, win)

Read files p_start:p_end once each (all 4 directions out of the same
read), dropping the repeated bry_time sample shared by consecutive daily
files. `win[direction][name]` holds ssh/temp/salt/u/v for the whole window.
"""
function read_padded_window(bry_file, fod, p_start, p_end)
    time = Float64[]
    names = (ssh = "zeta", temp = "temp", salt = "salt", u = "u", v = "v")
    parts = Dict(d => Dict(String(k) => Array{Float64}[] for k in keys(names)) for d in DBRY_DIRS)

    for f in p_start:p_end
        NCDataset(bry_file(fod[f])) do ds
            t = ncfloat(ds, "bry_time")
            keep = f == p_start ? (1:length(t)) : (2:length(t))   # drop sample shared with previous file
            append!(time, t[keep])
            for d in DBRY_DIRS, (k, ncname) in pairs(names)
                a = ncfloat(ds, "$(ncname)_$(d)")
                push!(parts[d][String(k)], collect(selectdim(a, ndims(a), keep)))
            end
        end
    end

    win = Dict(d => Dict(k => cat(v...; dims = ndims(v[1])) for (k, v) in parts[d]) for d in DBRY_DIRS)
    return time, win
end

# u-/v-grid -> rho-grid along dim 1 (port of center2face.m: interior points
# averaged, the two edges linearly extrapolated)
function center2face(var::AbstractArray)
    n = size(var, 1)
    rest = ntuple(_ -> Colon(), ndims(var) - 1)
    out = similar(var, n + 1, size(var)[2:end]...)
    out[2:n, rest...] = 0.5 .* (view(var, 1:n-1, rest...) .+ view(var, 2:n, rest...))
    out[1, rest...] = 1.5 .* view(var, 1, rest...) .- 0.5 .* view(var, 2, rest...)
    out[n+1, rest...] = 1.5 .* view(var, n, rest...) .- 0.5 .* view(var, n - 1, rest...)
    return out
end

# in-situ density from depth, potential temperature and practical salinity
# (port of density_calcuation.m); z/temp/salt are (nalong, nz, nt)
function density_insitu(z, lon, lat, temp, salt)
    rho = similar(z)
    Threads.@threads for i in axes(z, 1)
        for t in axes(z, 3), k in axes(z, 2)
            p = max(gsw_p_from_z(z[i, k, t], lat[i]), -1.4)
            # salt < 0 is treated as 0, as gsw_SA_from_SP.m does (the C library returns NaN for it)
            SA = gsw_sa_from_sp(max(salt[i, k, t], 0.0), p, lon[i], lat[i])
            CT = gsw_ct_from_pt(SA, temp[i, k, t])   # temp = potential temp here; no p needed
            rho[i, k, t] = gsw_rho(SA, CT, p)
        end
    end
    return rho
end

# thickness-weighted depth mean (nalong, 1, nt) and the perturbation from it
function depth_mean_bar(var, thickness)
    var_bar = sum(var .* thickness; dims = 2) ./ sum(thickness; dims = 2)
    return var_bar, var .- var_bar
end

# hydrostatic pressure of rho_prime, integrated from the surface layer (k = nz) downward
density_pressure(rho_prime, thickness) =
    reverse(cumsum(reverse(DBRY_G .* rho_prime .* thickness; dims = 2); dims = 2); dims = 2)

# x - lowpass(x) along time (dim 3), so that low + high = x exactly.
# The Butterworth lowpass is applied in transfer-function (b, a) form, like
# MATLAB's butter + filtfilt: filtfilt then pads the record ends the same
# way, which matters for the first/last days where there is no padding.
function highpass_time(x, time, cutoff, N)
    dt = time[2] - time[1]
    f = convert(PolynomialRatio, digitalfilter(Lowpass(1 / cutoff), Butterworth(N); fs = 1 / dt))
    b, a = coefb(f), coefa(f)
    xp = permutedims(x, (3, 1, 2))            # filtfilt works along dim 1
    x2 = reshape(xp, size(xp, 1), :)
    low = similar(x2)
    blocks = Iterators.partition(axes(x2, 2), cld(size(x2, 2), Threads.nthreads()))
    Threads.@threads for cols in collect(blocks)
        low[:, cols] = filtfilt(b, a, x2[:, cols])
    end
    return permutedims(reshape(x2 .- low, size(xp)), (2, 3, 1))
end

"""
    process_direction(d, time, w, g, vert, cutoff_days, N_butter, cols) -> Dict

One boundary, one padded time window: density -> rho' (anomaly from the
TIME-mean profile) -> pressure -> baroclinic p'/u'/v' -> highpass -> flux,
then trimmed to the window columns `cols` before being returned.
"""
function process_direction(d, time, w, g, vert, cutoff_days, N_butter, cols)
    if d in ("north", "south")
        u = center2face(w["u"]); v = w["v"]
    else
        u = w["u"]; v = center2face(w["v"])
    end

    nt = length(time)
    h2 = repeat(g.bathc, 1, nt)
    z_r, _ = zlevs3(h2, w["ssh"], vert.theta_s, vert.theta_b, vert.hc, vert.nz, "r", "new2008")
    z_w, _ = zlevs3(h2, w["ssh"], vert.theta_s, vert.theta_b, vert.hc, vert.nz, "w", "new2008")
    z_r = permutedims(z_r, (2, 1, 3))
    lthick = diff(permutedims(z_w, (2, 1, 3)); dims = 2)

    rho = density_insitu(z_r, g.lon, g.lat, w["temp"], w["salt"])
    all(isfinite, rho) || @warn "$d boundary: $(count(!isfinite, rho)) non-finite density values"
    # rho_bar is the TIME mean at each level (background stratification),
    # not the depth mean -- taken over this padded window, on sigma levels
    rho_bar = mean(rho; dims = 3)
    rho_prime = rho .- rho_bar
    p = density_pressure(rho_prime, lthick)
    p_bar, p_prime = depth_mean_bar(p, lthick)
    u_bar, u_prime = depth_mean_bar(u, lthick)
    v_bar, v_prime = depth_mean_bar(v, lthick)

    p_fast = highpass_time(p_prime, time, cutoff_days, N_butter)
    u_fast = highpass_time(u_prime, time, cutoff_days, N_butter)
    v_fast = highpass_time(v_prime, time, cutoff_days, N_butter)

    Fx = dropdims(sum(p_fast .* u_fast .* lthick; dims = 2); dims = 2)
    Fy = dropdims(sum(p_fast .* v_fast .* lthick; dims = 2); dims = 2)

    return Dict{String,Any}(
        "rho_bar" => rho_bar[:, :, 1],
        "ssh" => w["ssh"][:, cols],
        "p_bar" => p_bar[:, 1, cols], "u_bar" => u_bar[:, 1, cols], "v_bar" => v_bar[:, 1, cols],
        "Fx" => Fx[:, cols], "Fy" => Fy[:, cols],
        "p_prime" => Float32.(p_prime[:, :, cols]), "u_prime" => Float32.(u_prime[:, :, cols]),
        "v_prime" => Float32.(v_prime[:, :, cols]),
        "p_fast" => Float32.(p_fast[:, :, cols]), "u_fast" => Float32.(u_fast[:, :, cols]),
        "v_fast" => Float32.(v_fast[:, :, cols]),
    )
end

"""
    write_flux_nc(fname, day_time, out, cols, rho0)

One day's flux file: up_west/up_east (from Fx), vp_south/vp_north (from
Fy) and bry_time (same raw values as the existing bry file). The fluxes
are the raw grid-relative ones (ROMS's own diag_pflx flux is grid-relative
too), divided by rho0 because ROMS reads them rho0-normalized.
"""
function write_flux_nc(fname, day_time, out, cols, rho0)
    NCDataset(fname, "c"; format = :netcdf4_classic) do ds
        defDim(ds, "bry_time", length(day_time))
        tv = defVar(ds, "bry_time", Float64, ("bry_time",); attrib = [
            "units" => "days since 1900-12-31 00:00:00",
            "long_name" => "time for dynamic boundary flux"])
        tv.var[:] = day_time
        for d in DBRY_DIRS
            m = DBRY_FLUX_VARS[d]
            data = out[d][m.src][:, cols] ./ rho0
            haskey(ds.dim, m.dim) || defDim(ds, m.dim, size(data, 1))
            v = defVar(ds, m.varname, Float64, (m.dim, "bry_time"); attrib = [
                "long_name" => "HF baroclinic energy flux, $d boundary, rho0-normalized",
                "units" => "m4 s-3",
                "rho0" => rho0,
                "note" => "raw flux (W/m) divided by rho0 (kg/m3)"])
            v.var[:, :] = data
        end
    end
end

# One day's diagnostics, same field names as the MATLAB daily MAT files:
# raw (NOT rho0-normalized) Fx and Fy on every boundary plus the *_prime /
# *_fast fields. Geometry is saved as column vectors for north/south and
# row vectors for east/west, as in MATLAB.
function write_daily_mat(fname, day_time, out, cols, geo, cfg)
    S = Dict{String,Any}("dx" => cfg.dx, "cutoff_days" => cfg.cutoff_days, "N_butter" => cfg.N_butter,
        "rho0" => cfg.rho0, "time" => reshape(day_time, :, 1))
    for d in DBRY_DIRS
        for k in (:lon, :lat, :bathc, :angc)
            g = getfield(geo[d], k)
            S["$(d)_$(k)"] = d in ("east", "west") ? reshape(g, 1, :) : reshape(g, :, 1)
        end
        S["$(d)_rho_bar"] = out[d]["rho_bar"]   # time-mean profile of the chunk this day was computed in
        for k in (DBRY_SMALL..., DBRY_BIG...)
            a = out[d][k]
            S["$(d)_$(k)"] = collect(selectdim(a, ndims(a), cols))
        end
    end
    matwrite(fname, S)
end

"""
    dbry_make(cfg)

Compute the flux in chunks of `cfg.chunk_days` days (each read with
`cfg.pad_days` extra days on both sides, so the highpass filter has a
clean transient) and write the daily files of each chunk as soon as it is
done. Consecutive daily files share one time sample; that sample is always
taken from the chunk of the earlier day, so both files hold the same value.
"""
function dbry_make(cfg)
    (; dx, child_grid_file, child_bry_path, flux_out_path, daily_mat_path, dating,
        cutoff_days, N_butter, rho0, chunk_days, pad_days, save_mat, t1) = cfg

    geo = dbry_geo(child_grid_file)
    fod = Dates.format.(DateTime.(dating), "yyyymmddHH")
    nfiles = length(fod)
    bry_file(f) = joinpath(child_bry_path, "roms_bry_$(dx)m_$(f).nc")

    # sizes / vertical-coordinate params from the first file
    vert, ns = NCDataset(bry_file(fod[1])) do ds
        (theta_s = ncfloat(ds, "theta_s")[1], theta_b = ncfloat(ds, "theta_b")[1],
            hc = ncfloat(ds, "hc")[1], nz = ds.dim["s_rho"]), ds.dim["bry_time"]
    end
    chunk_starts = 1:chunk_days:nfiles
    @printf("nz=%d, nfiles=%d, %d samples per file, %d chunk(s), %d thread(s)\n",
        vert.nz, nfiles, ns, length(chunk_starts), Threads.nthreads())

    mkpath(flux_out_path)
    save_mat && mkpath(daily_mat_path)
    carry = Dict{String,Any}()   # last time sample of the previous chunk, per direction

    for (ci, c_start) in enumerate(chunk_starts)
        c_end = min(c_start + chunk_days - 1, nfiles)
        p_start = max(1, c_start - pad_days)
        p_end = min(nfiles, c_end + pad_days)
        @printf("chunk %d/%d: output files %d-%d (%s to %s), read files %d-%d\n",
            ci, length(chunk_starts), c_start, c_end, fod[c_start], fod[c_end], p_start, p_end)

        time, win = read_padded_window(bry_file, fod, p_start, p_end)
        @assert length(time) == (ns - 1) * (p_end - p_start + 1) + 1 "unexpected number of bry_time samples"
        # window columns from the first sample of file c_start to the last of file c_end
        cols = ((c_start - p_start) * (ns - 1) + 1):((c_end - p_start) * (ns - 1) + ns)
        chunk_time = time[cols]

        out = Dict{String,Any}()
        for d in DBRY_DIRS
            o = process_direction(d, time, win[d], geo[d], vert, cutoff_days, N_butter, cols)
            for k in (DBRY_SMALL..., DBRY_BIG...)
                a = o[k]
                ci > 1 && (selectdim(a, ndims(a), 1) .= carry[d][k])
                haskey(carry, d) || (carry[d] = Dict{String,Any}())
                carry[d][k] = collect(selectdim(a, ndims(a), size(a, ndims(a))))
            end
            out[d] = o
        end
        win = nothing

        for f in c_start:c_end
            day_cols = (f - c_start) * (ns - 1) .+ (1:ns)
            day_time = chunk_time[day_cols]
            fname = joinpath(flux_out_path, "roms_dbry_flux_$(dx)m_$(fod[f]).nc")
            write_flux_nc(fname, day_time, out, day_cols, rho0)
            if save_mat
                mat_fname = joinpath(daily_mat_path,
                    "bryfile_dynamic$(dx)_$(Dates.format(dating[f], "yyyymmdd")).mat")
                write_daily_mat(mat_fname, day_time, out, day_cols, geo, cfg)
            end
        end
        to_date(t) = t1 + Millisecond(round(Int, t * 86_400_000))
        println("  wrote days $(fod[c_start]) to $(fod[c_end]) ",
            "(flux time $(to_date(chunk_time[1])) to $(to_date(chunk_time[end])))")
    end

    println("done -- $nfiles daily flux nc files written to $flux_out_path")
    save_mat && println("done -- $nfiles daily MAT files written to $daily_mat_path")
end
