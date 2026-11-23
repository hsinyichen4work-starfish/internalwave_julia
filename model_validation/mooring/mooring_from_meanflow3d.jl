#= mooring_from_meanflow3d.jl
Hsin-Yi Chen, USM DMS
Half-monthly / monthly / total mean profiles at the moorings and along the M1 -> CPIES7 transect, taken directly
from the 3-D time-mean files of meanflow/mean_flow_avg_loop.jl (<model>_meanflow_avg_3d.nc, 12 bins) instead of
extracting every time step (extract_mooring_all.jl) and averaging afterwards (mooring_halfmonth_avg.jl).

Why the result is the same: the interpolation is linear in the field, and for HYCOM (fixed depths) and NCOM
(layer thickness constant in time, checked: Aug 22 vs Nov 30 identical everywhere) the level depths do not change
in time, so  mean(interp(field)) = interp(mean(field)). The only difference is where a point is dry in some records
and wet in others (not the case at fixed depths / constant layers).

Method: the same as extract_mooring_all.jl (functions copied from there):
  horizontal: inverse bilinear cell + 4 bilinear weights on the native grid (regular or curvilinear)
  vertical:   per corner column linear in z onto ZOUT (m, positive up); above the top level: top value; between
              the deepest valid level and the column bottom (NCOM: -sum(lthick); HYCOM: deepest valid level):
              deepest value; deeper: NaN. Corners combined with their weights, dry corners dropped.
  u, v: east / north (as stored in the mean files).

Input : HALFMONTH/<model>_<yyyymmdd>_<yyyymmdd>/<model>_meanflow_avg_3d.nc (+ <model>_meanflow_avg.nc for nrec)
Output: OUT_DIR/<SRC>/halfmonth_avg/   same names and layout as mooring_halfmonth_avg.jl writes
  <SRC>_<name>_avg.nc        lon, lat, h, z(z); temp, salt, u, v (z, win); ssh (win)
  <SRC>_M1_CPIES7_avg.nc     lon, lat, dist, h (x), angle, mooring*; temp, salt, u, v, u_along, u_cross (z, x, win);
                             ssh (x, win)
  every file: win_label ("2022-09-01_15", "2022-09-16_30", "2022-09", ..., "total"), win_kind (half / month / total),
              win_t0, win_t1 ([t0, t1)), n_time (records averaged, from the meanflow file)
  SRC = HYCOM, NCOM2700 (the extract_mooring_all.jl names).

Settings (environment): MX_SRC (default HYCOM,NCOM2700), MX_OUTDIR (default /home/hsinyi/roms_data/mooring_extract),
                        MX_ZOUT ("zmin:dz:zmax", default -1000:5:0)
Usage (from ~/Documents/Julia):  julia --project=. model_validation/mooring/mooring_from_meanflow3d.jl
=#

using NCDatasets, Dates, Printf

const FUNC      = "/home/hsinyi/Documents/Julia/function/"
include(joinpath(FUNC, "mooring_transect.jl"))          # read_moorings, transect_points
const HALFMONTH = "/home/hsinyi/roms_data/halfmonth"
const LOCS      = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # same as extract_mooring_all.jl on aurelia
const OUT_DIR   = get(ENV, "MX_OUTDIR", "/home/hsinyi/roms_data/mooring_extract")
const ZOUT      = let r = parse.(Float64, split(get(ENV, "MX_ZOUT", "-1000:5:0"), ":")); collect(r[1]:r[2]:r[3]) end
const SOURCES   = Dict("HYCOM" => (folder = "HYCOM", kind = :hycom), "NCOM2700" => (folder = "NCOM", kind = :ncom))
const SRC_RUN   = split(get(ENV, "MX_SRC", "HYCOM,NCOM2700"), ",")
const DX_KM     = 1.0                                   # transect spacing (as extract_mooring_all.jl)
const PIECE_KM  = 20.0
const TR_ENDS   = ("M1", "CPIES7")
for s in SRC_RUN; haskey(SOURCES, s) || error("MX_SRC: $s not one of $(collect(keys(SOURCES)))"); end

moor_name, moor_lon, moor_lat = read_moorings(LOCS)
const TR = transect_points(moor_name, moor_lon, moor_lat, TR_ENDS...; dx_km = DX_KM)
println("points: ", join(moor_name, ", "), " + transect $(TR.name) ($(length(TR.lon)) points); ",
        "zout $(ZOUT[1])..$(ZOUT[end]) m ($(length(ZOUT)) levels)")

## --- horizontal (from extract_mooring_all.jl)
lon180(x) = x > 180 ? x - 360 : x

function invbilin(X, Y, x, y)                          # corners 00, 10, 01, 11 -> (a, b)
    a, b = 0.5, 0.5
    for _ in 1:50
        fx = (1 - a) * (1 - b) * X[1] + a * (1 - b) * X[2] + (1 - a) * b * X[3] + a * b * X[4] - x
        fy = (1 - a) * (1 - b) * Y[1] + a * (1 - b) * Y[2] + (1 - a) * b * Y[3] + a * b * Y[4] - y
        xa = (1 - b) * (X[2] - X[1]) + b * (X[4] - X[3]); xb = (1 - a) * (X[3] - X[1]) + a * (X[4] - X[2])
        ya = (1 - b) * (Y[2] - Y[1]) + b * (Y[4] - Y[3]); yb = (1 - a) * (Y[3] - Y[1]) + a * (Y[4] - Y[2])
        det = xa * yb - xb * ya
        da = (yb * fx - xb * fy) / det; db = (-ya * fx + xa * fy) / det
        a -= da; b -= db
        abs(da) + abs(db) < 1e-12 && break
    end
    return a, b
end

function locate(lon, lat, x, y)
    c = cosd(y)
    d = ((lon .- x) .* c) .^ 2 .+ (lat .- y) .^ 2
    d[isnan.(d)] .= Inf
    I = argmin(d)
    nx, ny = size(lon)
    for ci in (I[1] - 1, I[1]), cj in (I[2] - 1, I[2])
        (1 <= ci < nx && 1 <= cj < ny) || continue
        X = c .* (lon[ci, cj], lon[ci+1, cj], lon[ci, cj+1], lon[ci+1, cj+1])
        Y = (lat[ci, cj], lat[ci+1, cj], lat[ci, cj+1], lat[ci+1, cj+1])
        a, b = invbilin(X, Y, c * x, y)
        if -1e-6 <= a <= 1 + 1e-6 && -1e-6 <= b <= 1 + 1e-6
            a, b = clamp(a, 0, 1), clamp(b, 0, 1)
            return (ci, cj, ((1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b))
        end
    end
    error("point ($x, $y) is outside the grid")
end

function make_pieces(cells, groups)
    map(groups) do pts
        is = [cells[p][1] for p in pts]; js = [cells[p][2] for p in pts]
        (i = minimum(is):maximum(is) + 1, j = minimum(js):maximum(js) + 1, pts = pts)
    end
end

## --- vertical (from extract_mooring_all.jl)
function vinterp(z, f, zout, zbot)
    ok = [k for k in eachindex(z) if isfinite(z[k]) && isfinite(f[k])]
    out = fill(NaN, length(zout))
    isempty(ok) && return out, NaN
    o = sort(ok; by = k -> z[k])
    zs = Float64[z[k] for k in o]; fs = Float64[f[k] for k in o]
    zb = isfinite(zbot) ? min(zbot, zs[1]) : zs[1]
    for (n, zo) in enumerate(zout)
        if zo >= zs[end];    out[n] = fs[end]
        elseif zo < zb;      continue
        elseif zo <= zs[1];  out[n] = fs[1]
        else
            m = searchsortedlast(zs, zo)
            r = (zo - zs[m]) / (zs[m+1] - zs[m]); out[n] = fs[m] + r * (fs[m+1] - fs[m])
        end
    end
    return out, zb
end

const CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))

function point_profile(F, Z, Zb, ci, cj, w)
    s = zeros(length(ZOUT)); ws = zeros(length(ZOUT)); hb = 0.0; wb = 0.0
    for (q, (di, dj)) in enumerate(CORNERS)
        col, zb = vinterp(view(Z, ci + di, cj + dj, :), view(F, ci + di, cj + dj, :), ZOUT, Zb[ci+di, cj+dj])
        isfinite(zb) || continue
        hb += w[q] * zb; wb += w[q]
        for n in eachindex(col)
            isfinite(col[n]) && (s[n] += w[q] * col[n]; ws[n] += w[q])
        end
    end
    p = Float32.(s ./ ws)
    wb > 0 && (p[ZOUT .< hb / wb] .= NaN32)
    return p, wb > 0 ? hb / wb : NaN
end

function point_value(F, ci, cj, w)
    s = 0.0; ws = 0.0
    for (q, (di, dj)) in enumerate(CORNERS)
        x = F[ci+di, cj+dj]
        isfinite(x) && (s += w[q] * x; ws += w[q])
    end
    return Float32(s / ws)
end

nanbig(x) = (x !== missing && isfinite(x) && abs(x) < 1f30) ? Float64(x) : NaN

## --- windows: meanflow bins -> mooring_halfmonth_avg.jl labels, in its order (per month: halves, month; total last)
function windows(names, kinds, t0, t1, nrec)
    W = map(eachindex(names)) do b
        k = kinds[b] == "halfmonthly" ? "half" : kinds[b] == "monthly" ? "month" : "total"
        lab = k == "total" ? "total" : k == "month" ? Dates.format(t0[b], "yyyy-mm") :
              Dates.format(t0[b], "yyyy-mm-dd") * "_" * Dates.format(t1[b] - Day(1), "dd")
        (b = b, label = lab, kind = k, t0 = t0[b], t1 = t1[b], n = nrec[b],
         order = (k == "total", firstdayofmonth(t0[b]), k == "month", t0[b]))   # total last; per month: halves, then the month
    end
    sort(W; by = w -> w.order)
end

## --- output (layout of mooring_halfmonth_avg.jl)
function def_win!(ds, W)
    defDim(ds, "win", length(W))
    defVar(ds, "win_label", [w.label for w in W], ("win",))
    defVar(ds, "win_kind", [w.kind for w in W], ("win",), attrib = ["long_name" => "half / month / total"])
    defVar(ds, "win_t0", [w.t0 for w in W], ("win",), attrib = ["units" => "hours since 2022-01-01 00:00:00"])
    defVar(ds, "win_t1", [w.t1 for w in W], ("win",), attrib = ["units" => "hours since 2022-01-01 00:00:00",
                                                                "long_name" => "window end (exclusive)"])
    defVar(ds, "n_time", Int32[w.n for w in W], ("win",), attrib = ["long_name" => "records of time in the window"])
end

const VARINFO = (("temp", "potential temperature", "degC"), ("salt", "salinity", "psu"),
                 ("u", "eastward velocity", "m/s"), ("v", "northward velocity", "m/s"))

function save_mooring_avg(fname, name, lon, lat, h, P, ssh, W, sname, note)
    mkpath(dirname(fname))
    NCDataset(fname, "c") do ds
        ds.attrib["title"]     = "$sname at mooring $name"
        ds.attrib["station"]   = name
        ds.attrib["source"]    = "mooring_from_meanflow3d.jl; $note"
        ds.attrib["averaging"] = "means per window taken from the 3-D time-mean file, then interpolated to the point"
        defDim(ds, "z", length(ZOUT))
        def_win!(ds, W)
        defVar(ds, "lon", lon, (), attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", lat, (), attrib = ["units" => "degrees_north"])
        defVar(ds, "h", h, (), attrib = ["long_name" => "water depth at the point", "units" => "m"])
        defVar(ds, "z", ZOUT, ("z",), attrib = ["units" => "m", "positive" => "up"])
        for (v, ln, u) in VARINFO
            defVar(ds, v, P[v], ("z", "win"); attrib = ["long_name" => ln, "units" => u], fillvalue = NaN32)
        end
        defVar(ds, "ssh", ssh, ("win",); attrib = ["long_name" => "sea surface height", "units" => "m"], fillvalue = NaN32)
    end
    println("  saved ", fname)
end

function save_transect_avg(fname, tr, P, ssh, h, W, sname, note)
    c, s = cosd(tr.angle), sind(tr.angle)
    vars = [(v, P[v], ln, u) for (v, ln, u) in VARINFO]
    push!(vars, ("u_along", P["u"] .* c .+ P["v"] .* s, "velocity along the transect, positive toward $(tr.to)", "m/s"))
    push!(vars, ("u_cross", -P["u"] .* s .+ P["v"] .* c,
                 "velocity across the transect, positive 90 deg to the left of u_along", "m/s"))
    mkpath(dirname(fname))
    NCDataset(fname, "c") do ds
        ds.attrib["title"]     = "$sname interpolated to the $(tr.from) - $(tr.to) transect"
        ds.attrib["source"]    = "mooring_from_meanflow3d.jl; $note"
        ds.attrib["averaging"] = "means per window taken from the 3-D time-mean file, then interpolated to the points"
        ds.attrib["transect"]  = "straight line from $(tr.from) to $(tr.to)"
        defDim(ds, "z", length(ZOUT)); defDim(ds, "x", length(tr.dist)); defDim(ds, "moor", length(tr.moor))
        def_win!(ds, W)
        defVar(ds, "lon", tr.lon, ("x",), attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", tr.lat, ("x",), attrib = ["units" => "degrees_north"])
        defVar(ds, "dist", tr.dist, ("x",), attrib = ["long_name" => "distance from $(tr.from) along the transect", "units" => "km"])
        defVar(ds, "angle", tr.angle, (), attrib = ["long_name" => "transect direction ($(tr.from) -> $(tr.to)), counterclockwise from east", "units" => "degrees"])
        defVar(ds, "z", ZOUT, ("z",), attrib = ["units" => "m", "positive" => "up"])
        defVar(ds, "mooring", tr.moor, ("moor",), attrib = ["long_name" => "moorings on the transect"])
        defVar(ds, "mooring_dist", tr.moor_dist, ("moor",), attrib = ["long_name" => "along-transect distance of the mooring", "units" => "km"])
        defVar(ds, "mooring_offset", tr.moor_off, ("moor",), attrib = ["long_name" => "distance of the mooring off the line (positive left)", "units" => "km"])
        defVar(ds, "h", h, ("x",), attrib = ["long_name" => "water depth", "units" => "m", "positive" => "up"])
        for (v, A, ln, u) in vars
            defVar(ds, v, A, ("z", "x", "win"); attrib = ["long_name" => ln, "units" => u], fillvalue = NaN32)
        end
        defVar(ds, "ssh", ssh, ("x", "win"); attrib = ["long_name" => "sea surface height", "units" => "m"], fillvalue = NaN32)
    end
    println("  saved ", fname)
end

## --- one source
function run_source(sname)
    t_start = time()
    src = SOURCES[sname]
    d = filter(n -> occursin(Regex("^$(src.folder)_\\d{8}_\\d{8}\$"), n), readdir(HALFMONTH))
    length(d) == 1 || error("$sname: expected one folder $(src.folder)_<start>_<end> in $HALFMONTH, found $d")
    f3 = joinpath(HALFMONTH, d[1], "$(src.folder)_meanflow_avg_3d.nc")
    f2 = joinpath(HALFMONTH, d[1], "$(src.folder)_meanflow_avg.nc")
    println("\n===== $sname: $f3")
    note = "$sname, $(basename(f3))"

    ds = NCDataset(f3)
    try
        names = ds["bin_name"][:]
        nrec = NCDataset(f2) do s2
            s2["bin_name"][:] == names || error("$f2: other bins than $f3")
            Int.(s2["nrec"][:])
        end
        W = windows(names, ds["bin_kind"][:], ds["bin_start"][:], ds["bin_end"][:], nrec)
        bsel = [w.b for w in W]                         # bins in output order
        lon = lon180.(Float64.(ds["lon"][:, :])); lat = Float64.(ds["lat"][:, :])
        dep = src.kind == :hycom ? Float64.(ds["depth"][:]) : nothing

        # points: moorings, then the transect in pieces with their own read box
        pts = [(moor_lon[k], moor_lat[k]) for k in eachindex(moor_name)]
        groups = [[k] for k in eachindex(moor_name)]
        ntr0 = length(pts)
        append!(pts, zip(TR.lon, TR.lat))
        append!(groups, [collect(r) for r in Iterators.partition(ntr0 + 1:length(pts), max(1, round(Int, PIECE_KM / DX_KM)))])
        cells = [locate(lon, lat, x, y) for (x, y) in pts]
        pieces = make_pieces(cells, groups)
        @printf("  %d windows; %d points in %d pieces (largest box %d x %d)\n", length(W), length(pts), length(pieces),
                maximum(length(p.i) for p in pieces), maximum(length(p.j) for p in pieces))

        nz, npt, nw = length(ZOUT), length(pts), length(W)
        F = Dict(v => fill(NaN32, nz, npt, nw) for v in ("temp", "salt", "u", "v"))
        S = fill(NaN32, npt, nw); hpt = fill(NaN, npt)
        for pc in pieces
            I, J = pc.i, pc.j; nb = (length(I), length(J))
            rd(v) = nanbig.(ds[v].var[I, J, :, bsel])   # (bx, by, nl, win)
            box = Dict(v => rd(v) for v in ("temp", "salt", "u", "v"))
            zeta = nanbig.(ds["zeta"].var[I, J, bsel])
            if src.kind == :hycom                         # fixed depths; bottom = deepest valid level
                Z = repeat(reshape(-dep, 1, 1, :), nb..., 1); Zb = fill(NaN, nb...)
            else                                          # NCOM: layer centres from lthick (constant in time: bin 1)
                th = nanbig.(ds["lthick"].var[I, J, :, bsel[1]]); th[isnan.(th)] .= 0
                cs = cumsum(th; dims = 3)
                Z = -(cs .- th ./ 2); Zb = -cs[:, :, end]
            end
            for p in pc.pts
                ci, cj, w = cells[p]; ci -= first(I) - 1; cj -= first(J) - 1
                for (jw, _) in enumerate(W)
                    for v in keys(F)
                        prof, hb = point_profile(view(box[v], :, :, :, jw), Z, Zb, ci, cj, w)
                        F[v][:, p, jw] = prof
                        v == "temp" && isnan(hpt[p]) && (hpt[p] = -hb)
                    end
                    S[p, jw] = point_value(view(zeta, :, :, jw), ci, cj, w)
                end
            end
        end

        out = joinpath(OUT_DIR, sname, "halfmonth_avg")
        for k in eachindex(moor_name)
            P = Dict(v => F[v][:, k, :] for v in keys(F))
            save_mooring_avg(joinpath(out, "$(sname)_$(moor_name[k])_avg.nc"), moor_name[k], moor_lon[k], moor_lat[k],
                             hpt[k], P, S[k, :], W, sname, note)
        end
        r = ntr0 + 1:npt
        save_transect_avg(joinpath(out, "$(sname)_$(TR.name)_avg.nc"), TR, Dict(v => F[v][:, r, :] for v in keys(F)),
                          S[r, :], hpt[r], W, sname, note)
    finally
        close(ds)
    end
    @printf("===== %s done, %.1f min\n", sname, (time() - t_start) / 60)
end

foreach(run_source, SRC_RUN)
