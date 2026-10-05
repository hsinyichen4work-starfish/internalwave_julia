# grid_tools.jl
#
# Building the child ROMS grid: where it sits, its bathymetry, smoothing,
# and matching the bathymetry to the parent along the open boundaries.
# Ported from grid_code/ (gid_middle.m, grid_setting.m, bathy_interp.m,
# make_roms_ncgrid.m, lsmooth_fun.m, rfact.m, match_boundary_topo.m) and
# useful_tools/ (midpoints.m, lonlat2xy.m, lonlat_rad2deg.m).

const MOORING_FILE = "/home/mbui/ModelOutput/NCOM/NOPP_mooring/Amazon_nopp_mooring_final.mat"

"midpoints between adjacent elements of a vector"
midpoints(v::AbstractVector) = (v[1:end-1] .+ v[2:end]) ./ 2

"""
    lonlat2xy(lon, lat, lon_0, lat_0) -> (x, y)

lon/lat (degrees) to a local x-y plane (meters): equirectangular
projection centered at (lon_0, lat_0), which becomes (0,0).
"""
function lonlat2xy(lon, lat, lon_0, lat_0)
    lat0_rad = lat_0 * pi / 180
    dlon = (lon .- lon_0) .* pi ./ 180
    dlat = (lat .- lat_0) .* pi ./ 180
    return R_EARTH .* dlon .* cos(lat0_rad), R_EARTH .* dlat
end

"radians -> degrees, with longitude wrapped to -180/180"
function lonlat_rad2deg(lon, lat)
    lon_deg = lon .* 180 ./ pi
    lon_deg[lon_deg .> 180] .-= 360
    return lon_deg, lat .* 180 ./ pi
end

"""
    gid_middle(mid_iter; mooring_file = MOORING_FILE) -> (mid, rot_ang)

Grid center `mid = [lon, lat]` and rotation `rot_ang` (degrees) from the
NOPP mooring line: start at the midpoint between mooring 1 and the
third-to-last CPIES, then move halfway back towards mooring 1
`mid_iter - 1` times. The grid's y-axis is aligned with the mooring line.
"""
function gid_middle(mid_iter::Integer; mooring_file::AbstractString = MOORING_FILE)
    d = matread(mooring_file)
    mooring_lon, mooring_lat = vec(d["mooring_lon"]), vec(d["mooring_lat"])
    cpies_lon, cpies_lat = vec(d["cpies_lon"]), vec(d["cpies_lat"])

    mid = [NaN, NaN]
    for j in 1:mid_iter
        if j == 1
            mid = [midpoints([mooring_lon[1], cpies_lon[end-2]])[1], midpoints([mooring_lat[1], cpies_lat[end-2]])[1]]
        else
            mid = [midpoints([mooring_lon[1], mid[1]])[1], midpoints([mooring_lat[1], mid[2]])[1]]
        end
    end

    x, y = lonlat2xy([mooring_lon[1], cpies_lon[end-2]], [mooring_lat[1], cpies_lat[end-2]], mid[1], mid[2])
    vec_moring = (-x[1] + x[2], -y[1] + y[2])
    rot_ang = rad2deg(atan(vec_moring[2], vec_moring[1])) - 90
    return mid, rot_ang
end

"""
    grid_setting(mid, rot_ang, dx, nx, ny) -> (grd, actual_dx_APPROX, actual_dy_APPROX)

Run easy_grid for an nx x ny grid of spacing dx (m) centered at `mid`.
`grd` holds lon4, lat4, pm, pn, ang, lone, late (radians, as easy_grid
returns them) plus lon4_deg, lat4_deg, lone_deg, late_deg (degrees,
longitude in -180/180).
"""
function grid_setting(mid, rot_ang, dx, nx, ny)
    size_x = nx * dx
    size_y = ny * dx
    lon4, lat4, pm, pn, ang, lone, late = easy_grid(nx, ny, size_x, size_y, mid[1], mid[2], rot_ang)

    actual_dx_APPROX = mean(1 ./ pm)   # should be ≈ dx
    actual_dy_APPROX = mean(1 ./ pn)

    # easy_grid works in radians --> degrees
    lon4_deg, lat4_deg = lonlat_rad2deg(lon4, lat4)
    lone_deg, late_deg = lonlat_rad2deg(lone, late)

    grd = Dict{String,Any}("lon4" => lon4, "lat4" => lat4, "pm" => pm, "pn" => pn, "ang" => ang,
        "lone" => lone, "late" => late, "lon4_deg" => lon4_deg, "lat4_deg" => lat4_deg,
        "lone_deg" => lone_deg, "late_deg" => late_deg)
    return grd, actual_dx_APPROX, actual_dy_APPROX
end

"""
    read_topo_subset(file, lonlim, latlim; lonname, latname, zname) -> topo

Read only the part of a global bathymetry file (GEBCO) inside
`lonlim`/`latlim`. `topo` = (lon, lat, Z) with Z sized (length(lon), length(lat)).
"""
function read_topo_subset(file::AbstractString, lonlim, latlim;
                          lonname = "Longitude", latname = "Latitude", zname = "elevation")
    NCDataset(file, "r") do ds
        lon = Float64.(ds[lonname].var[:])
        lat = Float64.(ds[latname].var[:])
        ilon = findall(x -> lonlim[1] <= x <= lonlim[2], lon)
        ilat = findall(x -> latlim[1] <= x <= latlim[2], lat)
        (isempty(ilon) || isempty(ilat)) && error("read_topo_subset: $file does not cover lon $lonlim, lat $latlim")
        v = ds[zname].var
        Z = _clean_ncdata(v, v[first(ilon):last(ilon), first(ilat):last(ilat)])
        (lon = lon[first(ilon):last(ilon)], lat = lat[first(ilat):last(ilat)], Z = Z)
    end
end

# Linear interpolation on the triangles of a regular lon/lat grid, each
# cell split along its NW-SE diagonal; NaN outside the grid or when a
# triangle has a NaN corner.
function _interp_regular_tri(lon::AbstractVector, lat::AbstractVector, Z::AbstractMatrix, xq::Real, yq::Real)
    nx, ny = length(lon), length(lat)
    (lon[1] <= xq <= lon[end] && lat[1] <= yq <= lat[end]) || return NaN
    i = clamp(searchsortedlast(lon, xq), 1, nx - 1)
    j = clamp(searchsortedlast(lat, yq), 1, ny - 1)
    s = (xq - lon[i]) / (lon[i+1] - lon[i])
    t = (yq - lat[j]) / (lat[j+1] - lat[j])
    f10, f01 = Z[i+1, j], Z[i, j+1]
    if s + t <= 1
        f00 = Z[i, j]
        return f00 + s * (f10 - f00) + t * (f01 - f00)
    else
        f11 = Z[i+1, j+1]
        return f11 + (1 - s) * (f01 - f11) + (1 - t) * (f10 - f11)
    end
end

"""
    bathy_interp(topo, grd) -> grd

Interpolate the source bathymetry `topo = (lon, lat, Z)` (regular lon/lat
grid, both ascending, Z sized (nlon, nlat)) onto the grid: adds `bath4`
at the rho points and `bathe` at the cell corners.

bathy_interp.m used scatteredInterpolant (Delaunay, 'linear', 'none').
On a regular source grid every cell is a rectangle and either diagonal
is a valid Delaunay split; MATLAB's triangulation of the GEBCO grid uses
the NW-SE one throughout, so each cell is split that way here and
interpolated linearly inside the triangle. This reproduces the MATLAB
values (checked against roms_grd_900m.nc: agreement to 1e-12 m) without
triangulating millions of points.
"""
function bathy_interp(topo, grd::AbstractDict)
    lon, lat, Z = topo.lon, topo.lat, topo.Z
    (issorted(lon) && issorted(lat)) || error("bathy_interp: topo.lon and topo.lat must be ascending")
    @printf("Building interpolant from %d source points...\n", length(Z))

    grd["bath4"] = map((x, y) -> _interp_regular_tri(lon, lat, Z, x, y), grd["lon4_deg"], grd["lat4_deg"])
    grd["bathe"] = map((x, y) -> _interp_regular_tri(lon, lat, Z, x, y), grd["lone_deg"], grd["late_deg"])

    # quick sanity check
    n_nan = count(isnan, grd["bath4"])
    if n_nan > 0
        @printf("Warning: %d NaN points after interpolation (out of source coverage?)\n", n_nan)
    else
        println("No NaNs -- interpolation covers full target grid.")
    end
    return grd
end

"""
    rfact(hr) -> r

rx0 (Beckmann-Haidvogel slope parameter) of a bathymetry field: the max
|rx0| in the two grid directions at each point.
"""
function rfact(hr::AbstractMatrix)
    r1 = zeros(size(hr))
    r2 = zeros(size(hr))
    r1[1:end-1, :] = 0.5 .* (hr[2:end, :] .- hr[1:end-1, :]) ./ (hr[2:end, :] .+ hr[1:end-1, :])
    r2[:, 1:end-1] = 0.5 .* (hr[:, 2:end] .- hr[:, 1:end-1]) ./ (hr[:, 2:end] .+ hr[:, 1:end-1])
    return max.(abs.(r1), abs.(r2))
end

"""
    lsmooth_fun(gridfile, rmax, hmin, offset) -> Dict of the grid file

Log-smooth `hraw` until the slope factor is below `rmax`, and write the
result to `h`. `hraw` is elevation (negative = ocean): it is flipped to
positive depth, shifted by `offset`, clipped at `hmin`, smoothed, and
shifted back.

Based on the ideas of Sasha Shchepetkin, (c) Jeroen Molemaker UCLA, 2008.
"""
function lsmooth_fun(gridfile::AbstractString, rmax::Real, hmin::Real, offset::Real)
    h = permutedims(ncread(gridfile, "hraw"))
    h = -h              # elevation-style (neg = ocean) to positive depth
    h = h .+ offset

    rmax_log = rmax > 0 ? log((1 + rmax * 0.9) / (1 - rmax * 0.9)) : 0.0

    h[h .< hmin] .= hmin

    hl = log.(h ./ hmin)
    hl[hl .< 0] .= 0

    cf1 = 1 / 6
    cf2 = 0.25

    # limited log-slope: cff*(1 - rmax_log/|cff|), zero where |cff| < rmax_log
    op(cff) = abs(cff) < rmax_log ? 0.0 : cff * (1 - rmax_log / abs(cff))

    println("iter   max(r)")
    iter_max = 200
    for iter in 1:iter_max
        Op1 = op.(hl[2:end, 2:end-1] .- hl[1:end-1, 2:end-1])
        Op2 = op.(hl[2:end-1, 2:end] .- hl[2:end-1, 1:end-1])
        Op3 = op.(hl[2:end, 2:end] .- hl[1:end-1, 1:end-1])
        Op4 = op.(hl[1:end-1, 2:end] .- hl[2:end, 1:end-1])

        hl[2:end-1, 2:end-1] .+= cf1 .* (
            Op1[2:end, :] .- Op1[1:end-1, :] .+ Op2[:, 2:end] .- Op2[:, 1:end-1] .+
            cf2 .* (Op3[2:end, 2:end] .- Op3[1:end-1, 1:end-1] .+ Op4[1:end-1, 2:end] .- Op4[2:end, 1:end-1]))

        # No gradient at the domain boundaries, this is required for
        # consistency with the ROMS open boundary conditions
        hl[1, :] = hl[2, :]
        hl[end, :] = hl[end-1, :]
        hl[:, 1] = hl[:, 2]
        hl[:, end] = hl[:, end-1]

        h = hmin .* exp.(hl)

        rt = 2 * nanmax(rfact(h))
        println(" ", iter, "   ", num2str(rt))
        rt < rmax && break
    end

    println("writing lsmoothed h to: ", gridfile)
    h = h .- offset
    ncwrite(gridfile, "h", permutedims(h))

    return read_nc_fun(gridfile)
end

"""
    make_roms_ncgrid(grd, grd_name, mid, rot_ang, dx, nx, ny, smooth_var, grid_path) -> Dict of the grid file

Write `grid_path/grd_name.nc` from the easy_grid output + interpolated
bathymetry in `grd`, then smooth the bathymetry with
`smooth_var = (rmax, hmin, offset)`. An existing file is replaced.
"""
function make_roms_ncgrid(grd::AbstractDict, grd_name, mid, rot_ang, dx, nx, ny, smooth_var, grid_path)
    size_x = nx * dx
    size_y = ny * dx

    gridfile = joinpath(grid_path, grd_name * ".nc")
    make_grid(gridfile, nx, ny, grd["lon4"], grd["lat4"], grd["pn"], grd["pm"], grd["bath4"], grd["ang"],
        size_x, size_y, rot_ang, mid[1], mid[2])
    ncwrite(gridfile, "xy_flip", 0)

    return lsmooth_fun(gridfile, smooth_var.rmax, smooth_var.hmin, smooth_var.offset)
end

# chunk index bounds shared by match_boundary_topo, h2r_make_ini and
# h2r_frc_subgrid: n points cut into ndom chunks, neighbors sharing one
# point so the staggered u/v points between chunks are covered
function _chunk_bounds(n::Integer, ndom::Integer)
    sz = n ÷ ndom
    cmin = collect(0:ndom-1) .* sz
    cmax = collect(1:ndom) .* sz
    cmin[1] = 1
    cmax[end] = n
    return cmin, cmax
end

"""
    match_boundary_topo(pgrid, cgrid, obcflag = [1,1,1,1], ndomx = 1, ndomy = 1; width = 0.06, steep = 50) -> (cgrid, diag)

Blend child grid bathymetry with parent grid bathymetry near the open
boundaries, processed in (ndomx x ndomy) chunks.

- `pgrid` : Dict with lon/lat (or lon_rho/lat_rho), h and optionally mask (parent, e.g. NCOM)
- `cgrid` : Dict with lon_rho, lat_rho, h (child)
- `obcflag` : [South East North West], 1 = open (blend), 0 = closed

Returns a copy of `cgrid` with the blended `h` and the original as
`h_orig`, and `diag = (hpi, alpha)`: the interpolated parent topo and the
parent/child transition weight. Only the returned Dict is changed — the
grid file is not rewritten (same as the MATLAB version).
"""
function match_boundary_topo(pgrid::AbstractDict, cgrid::AbstractDict, obcflag = [1, 1, 1, 1],
                             ndomx::Integer = 1, ndomy::Integer = 1; width = 0.06, steep = 50)
    hc_full = cgrid["h"]
    Mc, Lc = size(hc_full)

    # match parent grid
    pgrid = standardize_name(copy(pgrid))
    lonp_full = map(x -> x < 0 ? x + 360 : x, pgrid["lon_rho"])
    latp_full = pgrid["lat_rho"]

    # -------- chunk index bounds (same pattern as h2r_make_ini) --------
    icmin, icmax = _chunk_bounds(Lc, ndomx)
    jcmin, jcmax = _chunk_bounds(Mc, ndomy)

    hcn = zeros(Mc, Lc)
    hpi_f = zeros(Mc, Lc)
    alpha_f = zeros(Mc, Lc)

    # -------- global distance-to-boundary field --------
    alpha_full = zeros(Mc, Lc)
    for I in 1:Lc, J in 1:Mc
        dist = min(J / Mc + (1 - obcflag[1]) * 1e6,           # South
                   (Lc - I) / Lc + (1 - obcflag[2]) * 1e6,    # East
                   (Mc - J) / Mc + (1 - obcflag[3]) * 1e6,    # North
                   I / Lc + (1 - obcflag[4]) * 1e6)           # West
        alpha_full[J, I] = 0.5 * tanh(steep * (dist - width)) + 0.5
    end

    # -------- process each chunk --------
    for domx in 1:ndomx, domy in 1:ndomy
        @printf("Chunk (%d,%d) of (%d,%d)\n", domx, domy, ndomx, ndomy)

        icb, ice = icmin[domx], icmax[domx]
        jcb, jce = jcmin[domy], jcmax[domy]

        lonc_chunk = cgrid["lon_rho"][jcb:jce, icb:ice]
        latc_chunk = cgrid["lat_rho"][jcb:jce, icb:ice]
        hc_chunk = hc_full[jcb:jce, icb:ice]
        alpha_chunk = alpha_full[jcb:jce, icb:ice]

        # crop parent grid tightly around THIS chunk only
        lon0, lon1 = minimum(lonc_chunk) - 0.05, maximum(lonc_chunk) + 0.05
        lat0, lat1 = minimum(latc_chunk) - 0.05, maximum(latc_chunk) + 0.05
        g = (lon0 .<= lonp_full .<= lon1) .& (lat0 .<= latp_full .<= lat1)

        jidx = findall(vec(any(g; dims = 2)))
        iidx = findall(vec(any(g; dims = 1)))
        (isempty(jidx) || isempty(iidx)) &&
            error("No parent overlap found for chunk ($domx,$domy) -- check coordinate conventions.")
        jmin, jmax = extrema(jidx)
        imin, imax = extrema(iidx)

        hp = pgrid["h"][jmin:jmax, imin:imax]
        lonp = lonp_full[jmin:jmax, imin:imax]
        latp = latp_full[jmin:jmax, imin:imax]
        maskp = haskey(pgrid, "mask") ? pgrid["mask"][jmin:jmax, imin:imax] : ones(size(hp))

        # interpolate parent topo onto this chunk's child points
        elem, coef, _ = get_tri_coef(lonp, latp, lonc_chunk, latc_chunk, maskp)
        hpi_chunk = apply_tri_coef(elem, coef, hp)

        # blend
        hcn_chunk = alpha_chunk .* hc_chunk .+ (1 .- alpha_chunk) .* hpi_chunk

        # place back into full arrays
        hcn[jcb:jce, icb:ice] = hcn_chunk
        hpi_f[jcb:jce, icb:ice] = hpi_chunk
        alpha_f[jcb:jce, icb:ice] = alpha_chunk
    end

    # -------- package outputs --------
    cgrid = copy(cgrid)
    cgrid["h_orig"] = hc_full
    cgrid["h"] = hcn
    return cgrid, (hpi = hpi_f, alpha = alpha_f)
end
