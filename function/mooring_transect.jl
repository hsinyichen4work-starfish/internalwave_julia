# mooring_transect.jl
#
# Shared helpers for the mooring / transect extraction scripts
# (extract_mooring_GLOBAL.jl, mooring_NCOM_extraction.jl, extract_mooring_HYCOM.jl):
#   read_moorings       mooring names + locations from the grid file attributes
#   transect_points     equally spaced points on a straight line between two moorings
#   save_transect       write <model>_<name>.nc for a transect, same layout for every model
#
# Every model is interpolated onto the SAME transect points (dx_km apart), so the
# transect files of the different models can be compared point by point.

using NCDatasets, Dates

const R_earth_km = 6371.0

"""
    read_moorings(mooring_file) -> (names, lons, lats)

Mooring locations (lon -180..180, lat) from the global attributes of mooring_file, e.g.
  :M1_mor_info = "indices for M1_mor in roms_grd_900m.nc , location:(-45.4802;2.5415)"
(order = variable order in the file: french, M1..M4, CPIES1..9).
"""
function read_moorings(mooring_file)
    NCDataset(mooring_file) do ds
        names = [replace(v, "_mor" => "") for v in keys(ds) if endswith(v, "_mor")]
        locs  = [match(r"location:\(([-\d.]+);([-\d.]+)\)", ds.attrib["$(n)_mor_info"]) for n in names]
        names, [parse(Float64, m[1]) for m in locs], [parse(Float64, m[2]) for m in locs]
    end
end

"""
    transect_points(names, lons, lats, from, to; dx_km=1.0) -> tr

Straight line from mooring `from` to mooring `to` (names in `names`), sampled
every ~dx_km (both end points included). Distances use a local flat-earth
approximation, which is fine for a few hundred km near the equator.

`tr` is a NamedTuple with
  lon, lat, dist (km from `from`)   - the transect points
  angle                             - direction from `from` to `to`, degrees counterclockwise from east
  moor, moor_dist, moor_off         - moorings within 2 km of the line, their along-line
                                      distance and their distance off the line (km)
"""
function transect_points(names, lons, lats, from, to; dx_km=1.0, tol_km=2.0)
    a, b = findfirst(==(from), names), findfirst(==(to), names)
    (a === nothing || b === nothing) && error("transect end $from / $to not in moorings: $(join(names, ", "))")
    coslat = cosd((lats[a] + lats[b]) / 2)
    km(lon, lat) = (deg2rad(lon - lons[a]) * coslat * R_earth_km, deg2rad(lat - lats[a]) * R_earth_km)
    xb, yb = km(lons[b], lats[b])
    L = hypot(xb, yb)
    n = max(round(Int, L / dx_km), 1)
    s = range(0, 1, length = n + 1)
    tr_lon = lons[a] .+ s .* (lons[b] - lons[a])
    tr_lat = lats[a] .+ s .* (lats[b] - lats[a])
    ex, ey = xb / L, yb / L

    # moorings on the line, sorted along it
    on = Tuple{String,Float64,Float64}[]
    for k in eachindex(names)
        x, y = km(lons[k], lats[k])
        d, off = x * ex + y * ey, -x * ey + y * ex
        abs(off) <= tol_km && -tol_km <= d <= L + tol_km && push!(on, (names[k], d, off))
    end
    sort!(on, by = o -> o[2])

    return (name = "$(from)_$(to)", from = from, to = to,
            lon = collect(tr_lon), lat = collect(tr_lat), dist = collect(s .* L),
            angle = atand(ey, ex),
            moor = first.(on), moor_dist = [o[2] for o in on], moor_off = [o[3] for o in on])
end

"""
    save_transect(fname, tr, z, time, vars; ssh=nothing, vars2d=[], h=nothing, title="", source="", note="")

Write one transect file. `z` (m, positive up, ascending), `time` (DateTime),
`vars` a list of `name => (array, long_name, units)` with arrays (z, x, time).
`vars2d`: same, with arrays (x, time) (e.g. several sea level fields).
For a file with only 2D fields pass `z = nothing` and `vars = []`.
If "u" and "v" are in `vars`, the along/cross-transect velocities are added:
  u_along  toward tr.to                      = u cosθ + v sinθ
  u_cross  90° to the left of u_along        = -u sinθ + v cosθ
with θ = tr.angle. `ssh` is (x, time), `h` (x).
"""
function save_transect(fname, tr, z, time, vars; ssh=nothing, vars2d=[], h=nothing, title="", source="", note="")
    vars = collect(Pair{String,<:Any}, vars)
    names = first.(vars)
    if "u" in names && "v" in names
        u = last(vars[findfirst(==("u"), names)])[1]
        v = last(vars[findfirst(==("v"), names)])[1]
        c, s = cosd(tr.angle), sind(tr.angle)
        push!(vars, "u_along" => (u .* c .+ v .* s, "velocity along the transect, positive toward $(tr.to)", "m/s"))
        push!(vars, "u_cross" => (-u .* s .+ v .* c,
                                  "velocity across the transect, positive 90 deg to the left of u_along", "m/s"))
    end

    mkpath(dirname(fname))
    NCDataset(fname, "c") do ds
        ds.attrib["title"]  = title
        ds.attrib["source"] = source
        isempty(note) || (ds.attrib["note"] = note)
        ds.attrib["transect"] = "straight line from $(tr.from) to $(tr.to)"
        z === nothing || defDim(ds, "z", length(z))
        defDim(ds, "x", length(tr.dist)); defDim(ds, "time", length(time))
        defDim(ds, "moor", length(tr.moor))

        defVar(ds, "lon", tr.lon, ("x",), attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", tr.lat, ("x",), attrib = ["units" => "degrees_north"])
        defVar(ds, "dist", tr.dist, ("x",), attrib = ["long_name" => "distance from $(tr.from) along the transect", "units" => "km"])
        defVar(ds, "angle", tr.angle, (), attrib = ["long_name" => "transect direction ($(tr.from) -> $(tr.to)), counterclockwise from east", "units" => "degrees"])
        z === nothing || defVar(ds, "z", Float64.(z), ("z",), attrib = ["units" => "m", "positive" => "up"])
        defVar(ds, "time", time, ("time",), attrib = ["units" => "hours since 2022-01-01 00:00:00"])
        defVar(ds, "mooring", tr.moor, ("moor",), attrib = ["long_name" => "moorings on the transect"])
        defVar(ds, "mooring_dist", tr.moor_dist, ("moor",), attrib = ["long_name" => "along-transect distance of the mooring", "units" => "km"])
        defVar(ds, "mooring_offset", tr.moor_off, ("moor",), attrib = ["long_name" => "distance of the mooring off the line (positive left)", "units" => "km"])
        h === nothing || defVar(ds, "h", h, ("x",), attrib = ["long_name" => "water depth", "units" => "m", "positive" => "up"])

        for (name, (A, long_name, units)) in vars
            defVar(ds, name, A, ("z", "x", "time"), attrib = ["long_name" => long_name, "units" => units])
        end
        ssh === nothing || defVar(ds, "ssh", ssh, ("x", "time"), attrib = ["long_name" => "sea surface height", "units" => "m"])
        for (name, (A, long_name, units)) in vars2d
            defVar(ds, name, A, ("x", "time"), attrib = ["long_name" => long_name, "units" => units])
        end
    end
    println("$(tr.name): saved $fname  ($(length(tr.dist)) points, $(time[1]) to $(time[end]), $(length(time)) records)")
    return fname
end
