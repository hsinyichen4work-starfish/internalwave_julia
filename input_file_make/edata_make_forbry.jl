# edata_make_forbry.jl
#
# Ported from ucla-tools/edata/bry_extract_obj.m + add_object.m +
# fix_bry_mask.m: writes the extraction objects (edata) that make a PARENT run
# save data along the open boundaries of a CHILD grid, for forcing a later
# child run and for the boundary energy flux (up, vp = pressure flux; the
# parent must be compiled with diag_pflx for those).
#
# For each open boundary there are three objects:
#     <gname>_<bnd>_r   (np, two)    rho points:  i, j           -> zeta, temp, salt
#     <gname>_<bnd>_u   (np, three)  u points:    i, j, angle    -> ubar, u, up
#     <gname>_<bnd>_v   (np, three)  v points:    i, j, angle    -> vbar, v, vp
#   - i, j are fractional PARENT indices in crop_parent.m index space
#     (rho-point k, 1-based, sits at k - 1.5); the angle is the CHILD grid
#     angle, to which extract_data rotates the velocities.
#   - extract_data.F splits object names at the first '_', so gname must not
#     contain '_'.
#   - Dimensions: west/east use <gname>_eta_rho (r, u) and <gname>_eta_v (v);
#     south/north use <gname>_xi_rho (r, v) and <gname>_xi_u (u).
#   - Child points outside the parent: allowed only if masked in the child;
#     they take the index of the nearest valid point on that boundary.
#   - Child points whose parent stencil is all land (fix_bry_mask.m) also take
#     the index of the nearest point that has parent water.
#
# Output goes to its own <parent>_bry_edata.nc, separate from the mooring
# <parent>_mor_edata.nc (edata_make.jl). ROMS reads one edata file, so the
# two are to be merged later by a separate script.
#
# Run, e.g.
#     julia --project=/home/hsinyi/Documents/Julia input_file_make/edata_make_forbry.jl

using NCDatasets

## settings
grid_path   = "/home/hsinyi/roms_data/grid/new_design/"
parent_file = joinpath(grid_path, "roms_grd_2700m_1104.nc")
child_file  = joinpath(grid_path, "roms_grd_900m_1104.nc")
gname       = "c900m"          # object set name (no '_')
edata_file  = joinpath(grid_path, "roms_grd_2700m_1104_bry_edata.nc")

# open boundaries, 1 = yes, 0 = no
obc_west  = 1
obc_east  = 1
obc_south = 1
obc_north = 1

period   = 3600.0              # output period (s)
rho_vars = "zeta, temp, salt"
u_vars   = "ubar, u, up"
v_vars   = "vbar, v, vp"

## functions
# Fractional (a, b) of (x0, y0) in the bilinear map of cell (i, j)..(i+1, j+1),
# by Newton iteration. Not limited to [0, 1]: values outside say which way
# the point lies from the cell.
function cell_ab(x, y, i, j, x0, y0)
    a, b = 0.5, 0.5
    for _ in 1:20
        xa = (1-a)*(1-b)*x[i,j] + a*(1-b)*x[i+1,j] + (1-a)*b*x[i,j+1] + a*b*x[i+1,j+1]
        ya = (1-a)*(1-b)*y[i,j] + a*(1-b)*y[i+1,j] + (1-a)*b*y[i,j+1] + a*b*y[i+1,j+1]
        dxa = (1-b)*(x[i+1,j] - x[i,j]) + b*(x[i+1,j+1] - x[i,j+1])
        dxb = (1-a)*(x[i,j+1] - x[i,j]) + a*(x[i+1,j+1] - x[i+1,j])
        dya = (1-b)*(y[i+1,j] - y[i,j]) + b*(y[i+1,j+1] - y[i,j+1])
        dyb = (1-a)*(y[i,j+1] - y[i,j]) + a*(y[i+1,j+1] - y[i+1,j])
        det = dxa*dyb - dxb*dya
        ex, ey = x0 - xa, y0 - ya
        da = ( dyb*ex - dxb*ey) / det
        db = (-dya*ex + dxa*ey) / det
        a += da; b += db
        abs(da) + abs(db) < 1e-12 && break
    end
    a, b
end

# Walk from cell (i, j) to the cell holding (x0, y0). Returns the 1-based
# fractional rho index (fi, fj), or nothing if the point is outside the grid.
function walk(x, y, x0, y0, i, j)
    nx, ny = size(x)
    tol = 1e-9
    for _ in 1:max(nx, ny)
        a, b = cell_ab(x, y, i, j, x0, y0)
        (-tol <= a <= 1 + tol && -tol <= b <= 1 + tol) && return (i + a, j + b)
        di = a < -tol ? floor(Int, a) : a > 1 + tol ? floor(Int, a) : 0
        dj = b < -tol ? floor(Int, b) : b > 1 + tol ? floor(Int, b) : 0
        ni, nj = clamp(i + di, 1, nx - 1), clamp(j + dj, 1, ny - 1)
        (ni, nj) == (i, j) && return nothing      # pushing against the edge
        i, j = ni, nj
    end
    return nothing
end

function nearest_cell(x, y, x0, y0)
    d = @. (x - x0)^2 + (y - y0)^2
    ic, jc = Tuple(argmin(d))
    clamp(ic, 1, size(x, 1) - 1), clamp(jc, 1, size(x, 2) - 1)
end

# Parent index (crop_parent.m space) of every point of a boundary line;
# NaN where the point is outside the parent grid.
function parent_index(lonp, latp, lon, lat)
    n = length(lon)
    oi, oj = fill(NaN, n), fill(NaN, n)
    guess = nearest_cell(lonp, latp, lon[1], lat[1])
    for k in 1:n
        ij = walk(lonp, latp, lon[k], lat[k], guess...)
        if ij === nothing        # retry from the nearest parent point
            ij = walk(lonp, latp, lon[k], lat[k], nearest_cell(lonp, latp, lon[k], lat[k])...)
        end
        ij === nothing && continue
        oi[k], oj[k] = ij[1] - 1.5, ij[2] - 1.5
        guess = (clamp(floor(Int, ij[1]), 1, size(lonp, 1) - 1),
                 clamp(floor(Int, ij[2]), 1, size(lonp, 2) - 1))
    end
    oi, oj
end

# interp1(xgood, vgood, xbad, 'nearest', 'extrap') on point numbers
function fill_nearest!(v, bad)
    good = findall(.!bad)
    isempty(good) && error("no valid points to fill from")
    for k in findall(bad)
        v[k] = v[good[argmin(abs.(good .- k))]]
    end
    v
end

# add_object.m: parent indices of one object, with the outside-point fix
function object_index(name, lonp, latp, lon, lat, msk)
    nxp, nyp = size(lonp)
    oi, oj = parent_index(lonp, latp, lon, lat)
    if any(isnan, oi)
        any(isnan.(oi) .& (msk .>= 1)) &&
            error("$name: some points are outside the parent grid and not masked")
        bad = isnan.(oi) .| (oi .> nxp - 2) .| (oi .< 0) .| (oj .> nyp - 2) .| (oj .< 0)
        println("  $name: $(sum(isnan, oi)) masked points outside the parent, ",
                "$(sum(bad)) set to the nearest valid point")
        fill_nearest!(oi, bad); fill_nearest!(oj, bad)
    end
    w = msk .>= 1
    if any(w) && (minimum(oi[w]) < 0 || maximum(oi[w]) > nxp - 2 ||
                  minimum(oj[w]) < 0 || maximum(oj[w]) > nyp - 2)
        println("  $name: some points are borderline")
    end
    oi, oj
end

# fix_bry_mask.m: points whose parent stencil is all land take the index of
# the nearest point that has water.
function fix_mask!(name, oi, oj, mskp, kind)
    stencil(m, fi, fj) = begin
        il = clamp(floor(Int, fi), 1, size(m, 1) - 1)
        jl = clamp(floor(Int, fj), 1, size(m, 2) - 1)
        m[il,jl] + m[il+1,jl] + m[il,jl+1] + m[il+1,jl+1]
    end
    if kind == 'r'
        sms = [stencil(mskp, oi[k] + 1.5, oj[k] + 1.5) for k in eachindex(oi)]
    else   # velocities are rotated, so both u and v must have water
        msku = mskp[1:end-1, :] .* mskp[2:end, :]
        mskv = mskp[:, 1:end-1] .* mskp[:, 2:end]
        sms = [stencil(msku, oi[k] + 1.0, oj[k] + 1.5) *
               stencil(mskv, oi[k] + 1.5, oj[k] + 1.0) for k in eachindex(oi)]
    end
    bad = sms .== 0
    println("  $name: fixing $(sum(bad)) points that are inside the parent mask")
    any(bad) && (fill_nearest!(oi, bad); fill_nearest!(oj, bad))
    oi, oj
end

function add_bry_object!(ds, name, dname, lonp, latp, mskp, lon, lat, msk, vars, period;
                         ang = nothing)
    haskey(ds, name) && error("$name already exists in the edata file")
    kind = name[end]
    oi, oj = object_index(name, lonp, latp, lon, lat, msk)
    fix_mask!(name, oi, oj, mskp, kind)
    oi[isnan.(oi)] .= -1e5; oj[isnan.(oj)] .= -1e5

    np = length(oi)
    if haskey(ds.dim, dname)
        ds.dim[dname] == np || error("dimension $dname exists with a different size")
    else
        defDim(ds, dname, np)
    end
    if ang === nothing
        haskey(ds.dim, "two") || defDim(ds, "two", 2)
        v = defVar(ds, name, Float64, (dname, "two"), attrib = [
            "long_name" => "index coordinates of data object",
            "units"     => "non-dimensional"])
        v[:, :] = hcat(oi, oj)
    else
        haskey(ds.dim, "three") || defDim(ds, "three", 3)
        v = defVar(ds, name, Float64, (dname, "three"), attrib = [
            "long_name" => "index coordinates and angle of data object",
            "units"     => "non-dimensional and radian"])
        v[:, :] = hcat(oi, oj, ang)
    end
    v.attrib["output_period"] = period
    v.attrib["output_vars"]   = vars
    println(rpad(name, 18), "np = ", np, "  i ", round.(extrema(oi); digits = 2),
            "  j ", round.(extrema(oj); digits = 2))
end

function make_bry_edata(parent_file, child_file, edata_file, gname;
                        obc = (west = 1, east = 1, south = 1, north = 1),
                        period = 1800.0, rho_vars = "zeta, temp, salt",
                        u_vars = "ubar, u, up", v_vars = "vbar, v, vp")
    occursin('_', gname) && error("gname '$gname' must not contain '_'")
    lonp, latp, mskp = NCDataset(parent_file) do ds
        mod.(Array(ds["lon_rho"]), 360), Array(ds["lat_rho"]), Array(ds["mask_rho"])
    end
    lonr, latr, angr, mskr = NCDataset(child_file) do ds
        mod.(Array(ds["lon_rho"]), 360), Array(ds["lat_rho"]), Array(ds["angle"]),
        Array(ds["mask_rho"])
    end
    avx(f) = 0.5 .* (f[1:end-1, :] .+ f[2:end, :])
    avy(f) = 0.5 .* (f[:, 1:end-1] .+ f[:, 2:end])
    lonu, latu, angu = avx(lonr), avx(latr), avx(angr)
    lonv, latv, angv = avy(lonr), avy(latr), avy(angr)
    msku = mskr[1:end-1, :] .* mskr[2:end, :]
    mskv = mskr[:, 1:end-1] .* mskr[:, 2:end]

    # boundary line of a field: west/east = first/last xi row, south/north = eta
    line(f, bnd) = bnd == "west"  ? f[1, :]   : bnd == "east"  ? f[end, :] :
                   bnd == "south" ? f[:, 1]   : f[:, end]

    pname, cname = basename(parent_file), basename(child_file)
    NCDataset(edata_file, "c") do ds
        for bnd in ("west", "east", "south", "north")
            getfield(obc, Symbol(bnd)) == 1 || continue
            we = bnd in ("west", "east")
            dr = gname * (we ? "_eta_rho" : "_xi_rho")
            du = gname * (we ? "_eta_rho" : "_xi_u")
            dv = gname * (we ? "_eta_v"   : "_xi_rho")
            pre = gname * "_" * bnd
            add_bry_object!(ds, pre * "_r", dr, lonp, latp, mskp,
                line(lonr, bnd), line(latr, bnd), line(mskr, bnd), rho_vars, period)
            add_bry_object!(ds, pre * "_u", du, lonp, latp, mskp,
                line(lonu, bnd), line(latu, bnd), line(msku, bnd), u_vars, period;
                ang = line(angu, bnd))
            add_bry_object!(ds, pre * "_v", dv, lonp, latp, mskp,
                line(lonv, bnd), line(latv, bnd), line(mskv, bnd), v_vars, period;
                ang = line(angv, bnd))
        end
        ds.attrib[gname * "_info"] = "indices for $cname in $pname"
    end
    println("wrote ", edata_file)
end

## run
if abspath(PROGRAM_FILE) == @__FILE__
    isfile(edata_file) && error("$edata_file exists; remove it first")
    make_bry_edata(parent_file, child_file, edata_file, gname;
        obc = (west = obc_west, east = obc_east, south = obc_south, north = obc_north),
        period, rho_vars, u_vars, v_vars)
end
