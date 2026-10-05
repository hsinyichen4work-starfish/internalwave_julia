# easy_grid.jl
#
# Easy Grid: a rectangular, orthogonal ROMS grid with minimal grid-size
# variation. Ported from ucla-tools/EGRID_EXP (easy_grid.m, rot_sphere.m,
# tra_sphere.m, gc_dist.m, make_grid.m, create_grid.m),
# (c) 2008 Jeroen Molemaker, UCLA.
#
# Arrays keep the MATLAB orientation, (ny+2, nx+2) = (eta, xi); they are
# transposed to (xi, eta) only when written to the grid file.

const R_EARTH = 6371315.0

# MATLAB meshgrid: X[i,j] = x[j], Y[i,j] = y[i]
_meshgrid(x::AbstractVector, y::AbstractVector) =
    (repeat(reshape(x, 1, :), length(y), 1), repeat(reshape(y, :, 1), 1, length(x)))

"""
    gc_dist(lon1, lat1, lon2, lat2)

Distance (m) between 2 points along a great circle (haversine).
lat and lon in radians!!
"""
function gc_dist(lon1, lat1, lon2, lat2)
    dlat = lat2 .- lat1
    dlon = lon2 .- lon1
    dang = 2 .* asin.(sqrt.(sin.(dlat ./ 2) .^ 2 .+ cos.(lat2) .* cos.(lat1) .* sin.(dlon ./ 2) .^ 2))
    return R_EARTH .* dang
end

# (x,y,z) on the unit sphere back to (lon,lat); shared by rot_sphere/tra_sphere
function _xyz2lonlat(x2, y2, z2)
    lon2 = abs(y2) > 1e-7 ? atan(abs(x2 / y2)) : pi / 2
    y2 < 0 && (lon2 = pi - lon2)
    x2 < 0 && (lon2 = -lon2)

    pr2 = sqrt(x2^2 + y2^2)
    lat2 = abs(pr2) > 1e-7 ? atan(abs(z2 / pr2)) : pi / 2
    z2 < 0 && (lat2 = -lat2)
    return lon2, lat2
end

# angle of (a, z) in the plane of rotation, as rot_sphere.m/tra_sphere.m build it
function _plane_angle(a, z)
    ap = abs(a) > 1e-7 ? atan(abs(z / a)) : pi / 2
    a < 0 && (ap = pi - ap)
    z < 0 && (ap = -ap)
    return ap
end

"""
    rot_sphere(lon1, lat1, rot) -> (lon2, lat2)

Rotate the sphere around its y-axis by `rot` degrees.
Conventions: (lon,lat) = (0,0) is (x,y,z) = (0,-r,0); (0,90) is (0,0,r).
"""
function rot_sphere(lon1::AbstractArray, lat1::AbstractArray, rot::Real)
    rot = rot * pi / 180
    lon2, lat2 = similar(lon1, Float64), similar(lat1, Float64)
    for I in eachindex(lon1)
        x1 = sin(lon1[I]) * cos(lat1[I])
        y1 = cos(lon1[I]) * cos(lat1[I])
        z1 = sin(lat1[I])
        # rotate in the plane orthogonal to the y-axis: y stays constant
        rp1 = sqrt(x1^2 + z1^2)
        ap2 = _plane_angle(x1, z1) + rot
        lon2[I], lat2[I] = _xyz2lonlat(rp1 * cos(ap2), y1, rp1 * sin(ap2))
    end
    return lon2, lat2
end

"""
    tra_sphere(lon1, lat1, tra) -> (lon2, lat2)

Rotate the sphere around its x-axis by `tra` degrees, i.e. translate the
grid in the latitude direction.
"""
function tra_sphere(lon1::AbstractArray, lat1::AbstractArray, tra::Real)
    tra = tra * pi / 180
    lon2, lat2 = similar(lon1, Float64), similar(lat1, Float64)
    for I in eachindex(lon1)
        x1 = sin(lon1[I]) * cos(lat1[I])
        y1 = cos(lon1[I]) * cos(lat1[I])
        z1 = sin(lat1[I])
        # rotate in the plane orthogonal to the x-axis: x stays constant
        rp1 = sqrt(y1^2 + z1^2)
        ap2 = _plane_angle(y1, z1) + tra
        lon2[I], lat2[I] = _xyz2lonlat(x1, rp1 * cos(ap2), rp1 * sin(ap2))
    end
    return lon2, lat2
end

"""
    easy_grid(nx, ny, size_x, size_y, tra_lon, tra_lat, rot) -> (lon4, lat4, pm, pn, ang, lone, late)

Uses a Mercator projection around the equator and then rotates the sphere
around its 3 axes to position the grid where it is desired.

- `nx, ny`         number of grid points in the x / y direction
- `size_x, size_y` domain size in x / y direction (m)
- `tra_lon, tra_lat` desired longitude / latitude of the grid center (degrees)
- `rot`            rotation of the grid direction (0: x direction is west-east)

Returns rho-point lon/lat (radians, lon in 0..2π), pm, pn, the angle of
the local grid x-axis relative to east, all (ny+2, nx+2), and the
(ny+3, nx+3) lon/lat of the cell corners (`lone`, `late`).
"""
function easy_grid(nx::Integer, ny::Integer, size_x::Real, size_y::Real, tra_lon::Real, tra_lat::Real, rot::Real)
    # Mercator projection around the equator
    if size_y > size_x
        len, nl = size_y, ny
        width, nw = size_x, nx
    else
        len, nl = size_x, nx
        width, nw = size_y, ny
    end

    dlon = len / R_EARTH
    lon1d = dlon .* (-0.5:1:nl+0.5) ./ nl .- dlon / 2
    mul = 1.0
    dlat = width / R_EARTH
    y1 = y2 = 0.0
    lat1d = Float64[]
    nwc = round(Int, nw / 2, RoundNearestTiesAway)
    for _ in 1:100
        y1 = log(tan(pi / 4 - dlat / 4))
        y2 = log(tan(pi / 4 + dlat / 4))
        y = (y2 - y1) .* (-0.5:1:nw+0.5) ./ nw .+ y1
        lat1d = atan.(sinh.(y))
        dlat_cen = 0.5 * (lat1d[nwc+1] - lat1d[nwc-1])
        dlon_cen = dlon / nl
        mul = dlat_cen / dlon_cen * len / width * nw / nl
        dlat = dlat / mul
    end

    lon1de = dlon .* (-1:1:nl+1) ./ nl .- dlon / 2
    ye = (y2 - y1) .* (-1:1:nw+1) ./ nw .+ y1
    lat1de = atan.(sinh.(ye)) ./ mul

    lon1, lat1 = _meshgrid(collect(lon1d), lat1d)
    lone, late = _meshgrid(collect(lon1de), lat1de)
    lonu = 0.5 .* (lon1[:, 1:end-1] .+ lon1[:, 2:end])
    latu = 0.5 .* (lat1[:, 1:end-1] .+ lat1[:, 2:end])
    lonv = 0.5 .* (lon1[1:end-1, :] .+ lon1[2:end, :])
    latv = 0.5 .* (lat1[1:end-1, :] .+ lat1[2:end, :])

    if size_y > size_x
        lon1, lat1 = rot_sphere(lon1, lat1, 90)
        lonu, latu = rot_sphere(lonu, latu, 90)
        lonv, latv = rot_sphere(lonv, latv, 90)
        lone, late = rot_sphere(lone, late, 90)
        flipt(a) = permutedims(reverse(a; dims = 1))   # flipdim(a,1)'
        lon1, lat1 = flipt(lon1), flipt(lat1)
        lone, late = flipt(lone), flipt(late)
        lonu, latu, lonv, latv = flipt(lonv), flipt(latv), flipt(lonu), flipt(latu)
    end

    lon2, lat2 = rot_sphere(lon1, lat1, rot)
    lonu, latu = rot_sphere(lonu, latu, rot)
    lonv, latv = rot_sphere(lonv, latv, rot)
    lone, late = rot_sphere(lone, late, rot)

    lon3, lat3 = tra_sphere(lon2, lat2, tra_lat)
    lonu, latu = tra_sphere(lonu, latu, tra_lat)
    lonv, latv = tra_sphere(lonv, latv, tra_lat)
    lone, late = tra_sphere(lone, late, tra_lat)

    wrap(a) = a < -pi ? a + 2pi : a
    lon4 = wrap.(lon3 .+ tra_lon * pi / 180)
    lonu = wrap.(lonu .+ tra_lon * pi / 180)
    lonv = wrap.(lonv .+ tra_lon * pi / 180)
    lone = wrap.(lone .+ tra_lon * pi / 180)
    lat4 = lat3

    # Compute pn and pm:  pm = 1/dx
    pmu = gc_dist(lonu[:, 1:end-1], latu[:, 1:end-1], lonu[:, 2:end], latu[:, 2:end])
    pm = zeros(size(lon4))
    pm[:, 2:end-1] = pmu
    pm[:, 1] = pm[:, 2]
    pm[:, end] = pm[:, end-1]
    pm = 1 ./ pm

    # pn = 1/dy
    pnv = gc_dist(lonv[1:end-1, :], latv[1:end-1, :], lonv[2:end, :], latv[2:end, :])
    pn = zeros(size(lon4))
    pn[2:end-1, :] = pnv
    pn[1, :] = pn[2, :]
    pn[end, :] = pn[end-1, :]
    pn = 1 ./ pn

    # Compute angles of local grid positive x-axis relative to east
    dellat = latu[:, 2:end] .- latu[:, 1:end-1]
    dellon = lonu[:, 2:end] .- lonu[:, 1:end-1]
    dellon = map(d -> d > pi ? d - 2pi : (d < -pi ? d + 2pi : d), dellon)
    dellon = dellon .* cos.(0.5 .* (latu[:, 2:end] .+ latu[:, 1:end-1]))

    ang_s = atan.(dellat ./ (dellon .+ 1e-16))
    for I in eachindex(ang_s)
        if dellon[I] < 0 && dellat[I] < 0
            ang_s[I] -= pi
        elseif dellon[I] < 0 && dellat[I] >= 0
            ang_s[I] += pi
        end
        ang_s[I] > pi && (ang_s[I] -= pi)
        ang_s[I] < -pi && (ang_s[I] += pi)
    end

    ang = copy(lon4)
    ang[:, 2:end-1] = ang_s
    ang[:, 1] = ang[:, 2]
    ang[:, end] = ang[:, end-1]

    pos(a) = a < 0 ? a + 2pi : a
    lon4 = pos.(lon4)
    lone = pos.(lone)

    return lon4, lat4, pm, pn, ang, lone, late
end

# create_grid.m: every variable is double on (xi_rho, eta_rho) or (one)
const _GRID_VARS = [
    ("angle",    "Angle between xi axis and east",                 "radians"),
    ("h",        "Final bathymetry at rho-points",                 "meter"),
    ("hraw",     "Working bathymetry at rho-points",               "meter"),
    ("f",        "Coriolis parameter at rho-points",               "second-1"),
    ("pm",       "curvilinear coordinate metric in xi-direction",  "meter-1"),
    ("pn",       "curvilinear coordinate metric in eta-direction", "meter-1"),
    ("lon_rho",  "longitude of rho-points",                        "degree East"),
    ("lat_rho",  "latitude of rho-points",                         "degree North"),
    ("mask_rho", "mask at rho-points",                             "land/water (0/1)"),
]
const _GRID_SCALARS = [
    ("tra_lon", "Easy grid: Longitudinal translation of base grid", "degree East"),
    ("tra_lat", "Easy grid: Latitudinal translation of base grid",  "degree North"),
    ("rotate",  "Easy grid: Rotation of base grid",                 "degree"),
    ("xy_flip", "Easy grid: XY flip of base grid",                  "True/False (0/1)"),
]

"""
    make_grid(grdname, nx, ny, lon, lat, pn, pm, hraw, angle, xsize, ysize, rot, tra_lon, tra_lat)

Create the ROMS grid file `grdname` and fill it (make_grid.m +
create_grid.m). `lon`, `lat` in radians; all 2D inputs are (ny+2, nx+2).
`h` is left unfilled — `lsmooth_fun` writes it. The MATLAB version always
wrote `roms_grd.nc` in the current folder; here the file name is given.
"""
function make_grid(grdname::AbstractString, nx, ny, lon, lat, pn, pm, hraw, angle, xsize, ysize, rot, tra_lon, tra_lat)
    ROMS_title = string("ROMS grid by Easy Grid. Settings:",
        " nx: ", num2str(nx), " ny: ", num2str(ny),
        " xsize: ", num2str(xsize / 1e3), " ysize: ", num2str(ysize / 1e3),
        " rotate: ", num2str(rot), " Lon: ", num2str(tra_lon), " Lat: ", num2str(tra_lat))
    nxp = nx + 2
    nyp = ny + 2

    f = 4 * pi .* sin.(lat) ./ (24 * 3600)

    # Compute the mask (hraw is elevation: positive = land)
    mask = 0 .* hraw .+ 1
    mask[hraw .> 0] .= 0

    isfile(grdname) && rm(grdname)
    NCDataset(grdname, "c"; format = :netcdf4_classic) do ds
        defDim(ds, "one", 1)
        defDim(ds, "xi_rho", nxp)
        defDim(ds, "eta_rho", nyp)

        sph = defVar(ds, "spherical", Char, ("one",);
            attrib = ["Long_name" => "Grid type logical switch", "option_T" => "spherical"])
        for (name, long_name, units) in _GRID_VARS
            defVar(ds, name, Float64, ("xi_rho", "eta_rho"); attrib = ["Long_name" => long_name, "units" => units])
        end
        for (name, long_name, units) in _GRID_SCALARS
            defVar(ds, name, Float64, ("one",); attrib = ["Long_name" => long_name, "units" => units])
        end
        ds.attrib["Title"] = ROMS_title
        ds.attrib["Date"] = matlab_date()
        ds.attrib["Type"] = "ROMS grid produced by Easy Grid"

        # Fill the grid file
        t = permutedims
        ds["pm"][:, :] = t(pm)
        ds["pn"][:, :] = t(pn)
        ds["angle"][:, :] = t(angle)
        ds["hraw"][:, :] = t(hraw)
        ds["f"][:, :] = t(f)
        ds["mask_rho"][:, :] = t(mask)
        ds["lon_rho"][:, :] = t(lon) .* 180 ./ pi   # (degrees)
        ds["lat_rho"][:, :] = t(lat) .* 180 ./ pi   # (degrees)
        sph[1] = 'T'
        ds["tra_lon"][1] = tra_lon
        ds["tra_lat"][1] = tra_lat
        ds["rotate"][1] = rot
    end
    return grdname
end
