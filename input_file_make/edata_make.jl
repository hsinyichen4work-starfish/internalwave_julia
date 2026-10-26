# edata_make.jl
#
# Ported from ucla-tools/edata/mooring_extract_obj.m + add_object.m — writes
# the extraction-point (edata) file read by extract_data.opt: one
# <name>_mor(three, <name>_np) variable per mooring holding the fractional
# parent i, j index and the velocity rotation angle.
#   - Index space as in crop_parent.m: rho-point k (1-based) sits at k - 1.5,
#     so rho-points run from -0.5 to nx + 0.5.
#   - The fractional index comes from inverting the bilinear lon/lat map of
#     the grid cell holding the mooring (griddata in the MATLAB version).
#   - The angle is the grid angle interpolated to the mooring, which is what
#     the 900 m file holds (-0.597 rad).
#
# Run, e.g.
#     julia --project=/home/hsinyi/Documents/Julia input_file_make/edata_make.jl

using NCDatasets

## settings
grid_file  = "/home/hsinyi/roms_data/grid/roms_cmems_grd_2700m.nc"
edata_file = "/home/hsinyi/roms_data/grid/roms_cmems_grd_2700m_mor_edata.nc"

period = 360.0
mooring_vars = "zeta, temp, salt, u, v"

# (name, lon, lat) — same moorings as roms_grd_900m_mor_edata.nc
moorings = [
    ("french", -45.13,    3.95),
    ("M1",     -45.4802,  2.5415),
    ("M2",     -45.2785,  2.8378),
    ("M3",     -44.6732,  3.7267),
    ("M4",     -45.2312,  2.5888),
    ("CPIES1", -45.5275,  2.7905),
    ("CPIES2", -45.0767,  3.1342),
    ("CPIES3", -44.8748,  3.4303),
    ("CPIES4", -44.4713,  4.023),
    ("CPIES5", -44.2695,  4.3193),
    ("CPIES6", -44.0678,  4.6157),
    ("CPIES7", -43.8157,  4.986),
    ("CPIES8", -43.6974,  4.3635),
    ("CPIES9", -44.4382,  4.8677),
]

## functions
# Fractional (i, j) of (x0, y0) in the bilinear map of cell (i, j)..(i+1, j+1);
# returns nothing if the point is outside that cell.
function cell_frac(x, y, i, j, x0, y0)
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
        a += ( dyb*ex - dxb*ey) / det
        b += (-dya*ex + dxa*ey) / det
    end
    (-1e-9 <= a <= 1 + 1e-9 && -1e-9 <= b <= 1 + 1e-9) ? (a, b) : nothing
end

# Fractional rho-point index (i, j) of (lon0, lat0), 1-based; nothing if outside.
function find_index(lon, lat, lon0, lat0)
    d = @. (lon - lon0)^2 + (lat - lat0)^2
    ic, jc = Tuple(argmin(d))
    nx, ny = size(lon)
    for i in max(ic-1, 1):min(ic, nx-1), j in max(jc-1, 1):min(jc, ny-1)
        ab = cell_frac(lon, lat, i, j, lon0, lat0)
        ab === nothing || return (i + ab[1], j + ab[2])
    end
    return nothing
end

function bilin(f, fi, fj)
    i, j = floor(Int, fi), floor(Int, fj)
    a, b = fi - i, fj - j
    (1-a)*(1-b)*f[i,j] + a*(1-b)*f[i+1,j] + (1-a)*b*f[i,j+1] + a*b*f[i+1,j+1]
end

function make_edata(grid_file, edata_file, moorings; period = 360.0,
                    mooring_vars = "zeta, temp, salt, u, v")
    lon, lat, ang = NCDataset(grid_file) do ds
        mod.(Array(ds["lon_rho"]), 360), Array(ds["lat_rho"]), Array(ds["angle"])
    end
    nxp, nyp = size(lon)
    gname = basename(grid_file)

    NCDataset(edata_file, "c") do ds
        for (name, lon0, lat0) in moorings
            ij = find_index(lon, lat, mod(lon0, 360), lat0)   # grids use 0..360
            ij === nothing && error("$name ($lon0, $lat0) is outside $gname")
            fi, fj = ij
            obj = [fi - 1.5, fj - 1.5, bilin(ang, fi, fj)]
            (0 <= obj[1] <= nxp - 2 && 0 <= obj[2] <= nyp - 2) ||
                println("$name is borderline: i = $(obj[1]), j = $(obj[2])")

            vname = name * "_mor"
            defDim(ds, name * "_np", 1)
            haskey(ds.dim, "three") || defDim(ds, "three", 3)
            # Julia order is reversed from ncdump: this is (three, <name>_np) there
            v = defVar(ds, vname, Float64, (name * "_np", "three"), attrib = [
                "long_name"     => "index coordinates and angle of data object",
                "units"         => "non-dimensional and radian",
                "output_period" => period,
                "output_vars"   => mooring_vars])
            v[1, :] = obj
            ds.attrib[vname * "_info"] =
                "indices for $vname in $gname , location:($lon0;$lat0)"
            println(rpad(vname, 12), round.(obj; digits = 4))
        end
    end
    println("wrote ", edata_file)
end

## run
if abspath(PROGRAM_FILE) == @__FILE__
    isfile(edata_file) && error("$edata_file exists; remove it first")
    make_edata(grid_file, edata_file, moorings; period, mooring_vars)
end
