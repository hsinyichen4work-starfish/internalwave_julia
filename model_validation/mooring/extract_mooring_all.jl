#= extract_mooring_all.jl
Hsin-Yi Chen, USM DMS
One extraction for every model: values at the moorings and along the mooring line transect (M1 -> CPIES7,
every dx_km), all with the SAME method and the SAME output grid, so the models compare point by point.
Replaces, for new extractions, extract_mooring_GLOBAL.jl / extract_mooring_HYCOM.jl / mooring_NCOM_extraction.jl
(those stay as they are).

Sources (MX_SRC picks some, comma separated; default per machine, see P):
  design_1113_hycom_2700, design_1116_cmems_2700, amazon_900m_dbry_0927   ROMS runs
      moorings : the ROMS station output <run>/mooring_merged/<name>.nc (~10 min, at the station point,
                 s-levels -> z with zlevs3(h, zeta); u, v rotated by the station angle)
      transect : roms_his.*.nc (hourly snapshots; after_joint/ for the design runs, run folder for dbry)
  CMEMS     CMEMS_DATA/cmems_*_0731_1130_big.nc: u, v, T, S 6-hourly; ssh hourly on its own axis (time_ssh)
  HYCOM     HYCOM_DATA_NC_2700m_1104/hycom_<tag>_{grd,ssh,ts,uv}.nc, 3-hourly
  NCOM2700  NCOM_DATA_NC_2700/<tag>_{lthick,ssh,ts,uv}.nc, hourly
  NCOM900   NCOM_DATA_NC/<tag>_{lthick,ssh,ts,uv}.nc, hourly
  HYCOM / NCOM: the daily files overlap by one record; the NEWER file's record is kept.

Method (every gridded source; the ROMS moorings only do step 2):
  1. horizontal: the grid cell around each point (inverse bilinear, works on regular and curvilinear grids),
     4 bilinear weights. Only small boxes are read: the transect is cut into pieces of ~piece_km, each with
     its own box (as in mooring_NCOM_extraction.jl).
  2. vertical, per corner column: linear in z onto zout (m, positive up, ascending = bottom -> surface).
     Above the top level: top value. Between the deepest valid level and the column bottom (ROMS -h,
     NCOM -sum(layer thickness); HYCOM / CMEMS: the deepest valid level): deepest value. Deeper: NaN.
     Level depths: ROMS zlevs3(h, zeta, theta_s, theta_b, hc, N, "r", "new2008") every record (attributes of
     roms_his); NCOM layer centres from the daily layer thickness; HYCOM / CMEMS fixed depths.
  3. corners combined with their weights; a dry corner (land / below its bottom) is dropped and the others are
     rescaled; zout below the weighted bottom of the point -> NaN.
  u, v: east / north for every model (ROMS u, v moved to rho points and rotated by angle).

Output (OUT_DIR/<SRC>/), same layout as the old scripts:
  <SRC>_<name>.nc      lon, lat, h, z (z), time; temp, salt, u, v (z, time); ssh (time) or (time_ssh)
  <SRC>_M1_CPIES7.nc   save_transect: temp, salt, u, v, u_along, u_cross (z, x, time), ssh (x, time), h (x)
  <SRC>_M1_CPIES7_ssh.nc  only when ssh has its own time axis (CMEMS)

Settings (environment):
  MX_SRC      sources                     MX_HOST   expanse / aurelia (auto)
  MX_T0, MX_T1  time window [T0, T1) (default: everything)
  MX_ZOUT     "zmin:dz:zmax" (default -1000:5:0)
  MX_OUTDIR   output root
Arguments pick the points: mooring names and/or "transect" (default: all moorings + transect).

Usage:
  expanse:  sbatch sbatch_files/julia_extract_mooring_all.sh                   # 3 ROMS runs + CMEMS
            MX_SRC=CMEMS julia --project=. model_validation/mooring/extract_mooring_all.jl M2 transect
  aurelia:  MX_SRC=HYCOM,NCOM2700,NCOM900 nohup julia --project=. model_validation/mooring/extract_mooring_all.jl > mx.log 2>&1 &
=#

using NCDatasets, Dates, Printf, Statistics

## --- machine
const HOST = get(ENV, "MX_HOST", isdir("/expanse/lustre") ? "expanse" : "aurelia")
const P = HOST == "expanse" ? (
        func  = "/home/hchen54/internalwave_julia/function/",
        base  = "/expanse/lustre/projects/uso102/hchen54/",
        pdata = "/home/hchen54/aurelia/roms_data/",                       # HYCOM / NCOM over sshfs: slow, use aurelia
        locs  = "/expanse/lustre/projects/uso102/hchen54/input/grid/roms_grd_900m_mor_edata.nc",
        out   = "/expanse/lustre/projects/uso102/hchen54/mooring_extract",
        default = "design_1113_hycom_2700,design_1116_cmems_2700,amazon_900m_dbry_0927,CMEMS") :
    HOST == "aurelia" ? (
        func  = "/home/hsinyi/Documents/Julia/function/",
        base  = "/home/hsinyi/roms_data/",
        pdata = "/home/hsinyi/roms_data/",
        locs  = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc",
        out   = "/home/hsinyi/roms_data/mooring_extract",
        default = "HYCOM,NCOM2700,NCOM900") :
    error("MX_HOST = $HOST, not expanse / aurelia")
println("host ", HOST)
for f in ("mooring_transect.jl", "zlevs3.jl", "time_dir.jl")   # read_moorings, transect_points, save_transect; zlevs3; time_dir
    include(joinpath(P.func, f))
end

## --- sources
const B = P.base
const G27 = B * "new_design_fix/"
const SOURCES = [
    "design_1113_hycom_2700" => (kind = :roms, data = B * "design_1113_hycom_2700/after_joint/",
                                 station = B * "design_1113_hycom_2700/mooring_merged/",
                                 grid = G27 * "roms_grd_2700m_1104.nc", mor = G27 * "roms_grd_2700m_1104_mor_edata.nc"),
    "design_1116_cmems_2700" => (kind = :roms, data = B * "design_1116_cmems_2700/after_joint/",
                                 station = B * "design_1116_cmems_2700/mooring_merged/",
                                 grid = G27 * "roms_grd_2700m_1104.nc", mor = G27 * "roms_grd_2700m_1104_mor_edata.nc"),
    "amazon_900m_dbry_0927"  => (kind = :roms, data = B * "amazon_900m_dbry_0927/",
                                 station = B * "amazon_900m_dbry_0927/mooring_merged/",
                                 grid = B * "input/grid/roms_grd_900m.nc", mor = B * "input/grid/roms_grd_900m_mor_edata.nc"),
    "CMEMS"    => (kind = :cmems, data = B * "CMEMS_DATA/"),
    "HYCOM"    => (kind = :hycom, data = P.pdata * "HYCOM_DATA_NC_2700m_1104/", prefix = "hycom_"),
    "NCOM2700" => (kind = :ncom,  data = P.pdata * "NCOM_DATA_NC_2700/", prefix = ""),
    "NCOM900"  => (kind = :ncom,  data = P.pdata * "NCOM_DATA_NC/", prefix = ""),
]

## --- settings
const SRC_RUN  = split(get(ENV, "MX_SRC", P.default), ",")
const T0       = haskey(ENV, "MX_T0") ? DateTime(ENV["MX_T0"]) : typemin(DateTime)
const T1       = haskey(ENV, "MX_T1") ? DateTime(ENV["MX_T1"]) : typemax(DateTime)
const ZOUT     = let r = parse.(Float64, split(get(ENV, "MX_ZOUT", "-1000:5:0"), ":")); collect(r[1]:r[2]:r[3]) end
const OUT_DIR  = get(ENV, "MX_OUTDIR", P.out)
const T_REF    = DateTime(1994, 1, 1)                  # ROMS ocean_time
const DX_KM    = 1.0                                   # transect spacing (same as the old scripts)
const PIECE_KM = 20.0                                  # transect piece with its own read box
const TR_ENDS  = ("M1", "CPIES7")
inwin(t) = T0 <= t < T1
for s in SRC_RUN; any(==(s), first.(SOURCES)) || error("MX_SRC: $s not one of $(first.(SOURCES))"); end

moor_name, moor_lon, moor_lat = read_moorings(P.locs)
const TR = transect_points(moor_name, moor_lon, moor_lat, TR_ENDS...; dx_km = DX_KM)
const DO_TR = isempty(ARGS) || "transect" in ARGS
function find_moor(a)
    k = findfirst(==(a), moor_name)
    k === nothing && error("unknown point $a; choose from $(join(moor_name, ", ")), transect")
    return k
end
const MOOR_SEL = isempty(ARGS) ? collect(eachindex(moor_name)) : [find_moor(a) for a in ARGS if a != "transect"]
println("points: ", join(moor_name[MOOR_SEL], ", "), DO_TR ? " + transect $(TR.name) ($(length(TR.lon)) points)" : "",
        "; zout $(ZOUT[1])..$(ZOUT[end]) m ($(length(ZOUT)) levels)")

## --- horizontal: cell + bilinear weights on any (regular or curvilinear) grid
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

# lower-left corner (i, j) of the cell holding (x, y) and the weights of corners 00, 10, 01, 11
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

# points -> pieces: consecutive points sharing one read box (i, j ranges)
function make_pieces(cells, groups)
    map(groups) do pts
        is = [cells[p][1] for p in pts]; js = [cells[p][2] for p in pts]
        (i = minimum(is):maximum(is) + 1, j = minimum(js):maximum(js) + 1, pts = pts)
    end
end

## --- vertical
# column (levels in any order) -> zout; zbot: column bottom (NaN = deepest valid level)
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

# profile at one point from a box: F, Z (bx, by, nl), Zb (bx, by); (ci, cj) local lower-left corner
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
    p = Float32.(s ./ ws)                              # 0 / 0 -> NaN
    wb > 0 && (p[ZOUT .< hb / wb] .= NaN32)
    return p, wb > 0 ? hb / wb : NaN
end

function point_value(F, ci, cj, w)                     # 2-D field (ssh)
    s = 0.0; ws = 0.0
    for (q, (di, dj)) in enumerate(CORNERS)
        x = F[ci+di, cj+dj]
        isfinite(x) && (s += w[q] * x; ws += w[q])
    end
    return Float32(s / ws)
end

## --- readers: each gives grid (lon, lat 2-D), records, and per record + piece a box of
##     (; z, zb, temp, salt, u, v, ssh) with u, v east / north and NaN on land
nanbig(x) = (isfinite(x) && abs(x) < 1f30) ? Float64(x) : NaN

# ROMS his
function roms_setup(src)
    lon, lat, h, mask, angle = NCDataset(src.grid) do g
        lon180.(g["lon_rho"][:, :]), g["lat_rho"][:, :], g["h"][:, :], g["mask_rho"][:, :], g["angle"][:, :]
    end
    td = time_dir(src.data; prefix = "roms_his.", dt = 3600,
                  tstart = T0 == typemin(DateTime) ? nothing : T0, tend = T1 == typemax(DateTime) ? nothing : T1 - Second(1))
    vc = NCDataset(joinpath(src.data, td.files[1])) do ds
        (theta_s = ds.attrib["theta_s"], theta_b = ds.attrib["theta_b"], hc = ds.attrib["hc"], N = ds.dim["s_rho"])
    end
    return (; lon, lat, h, mask, angle, td, vc)
end

function roms_records(f, src, st, pieces)
    nx, ny = size(st.lon)
    for (fi, recs) in records_by_file(st.td)
        NCDataset(joinpath(src.data, st.td.files[fi])) do ds
            for (k, n) in recs
                t = st.td.tdate[n]; inwin(t) || continue
                boxes = map(pieces) do pc
                    I, J = pc.i, pc.j
                    zeta = Float64.(ds["zeta"].var[I, J, k])
                    temp = Float64.(ds["temp"].var[I, J, :, k]); salt = Float64.(ds["salt"].var[I, J, :, k])
                    ua = max(first(I) - 1, 1):min(last(I), nx - 1); u = ds["u"].var[ua, J, :, k]
                    va = max(first(J) - 1, 1):min(last(J), ny - 1); v = ds["v"].var[I, va, :, k]
                    ur = similar(temp); vr = similar(temp)
                    for (ii, i) in enumerate(I)          # u, v -> rho (edge rows: one neighbour)
                        lo, hi = max(i - 1, 1) - first(ua) + 1, min(i, nx - 1) - first(ua) + 1
                        ur[ii, :, :] .= 0.5 .* (u[lo, :, :] .+ u[hi, :, :])
                    end
                    for (jj, j) in enumerate(J)
                        lo, hi = max(j - 1, 1) - first(va) + 1, min(j, ny - 1) - first(va) + 1
                        vr[:, jj, :] .= 0.5 .* (v[:, lo, :] .+ v[:, hi, :])
                    end
                    a = st.angle[I, J]
                    ue = ur .* cos.(a) .- vr .* sin.(a); vn = ur .* sin.(a) .+ vr .* cos.(a)
                    hb = st.h[I, J]
                    z = permutedims(zlevs3(hb, zeta, st.vc.theta_s, st.vc.theta_b, st.vc.hc, st.vc.N, "r", "new2008")[1], (2, 3, 1))
                    land = st.mask[I, J] .== 0
                    for F in (temp, salt, ue, vn); for kk in axes(F, 3); view(F, :, :, kk)[land] .= NaN; end; end
                    zeta[land] .= NaN
                    (; z, zb = -hb, temp, salt, u = ue, v = vn, ssh = zeta)
                end
                f(t, boxes)
            end
        end
    end
end

# HYCOM / NCOM: tags with all needed kinds; per record the newer tag wins
function hn_setup(src)
    kinds = src.kind == :hycom ? ("grd", "ssh", "ts", "uv") : ("lthick", "ssh", "ts", "uv")
    re = Regex("^" * src.prefix * raw"(\d{10})_(" * join(kinds, "|") * raw")\.nc$")
    have = Dict{String, Set{String}}()
    for n in readdir(src.data)
        m = match(re, n); m === nothing && continue
        push!(get!(have, m[1], Set{String}()), m[2])
    end
    tags = sort([t for (t, k) in have if all(in(k), kinds)])
    isempty(tags) && error("no complete file sets in $(src.data)")
    fname(tag, kind) = joinpath(src.data, "$(src.prefix)$(tag)_$kind.nc")
    lon, lat, dep = NCDataset(fname(tags[1], kinds[1])) do ds
        lon180.(Float64.(ds["Longitude"].var[:, :])), Float64.(ds["Latitude"].var[:, :]),
        src.kind == :hycom ? Float64.(ds["depth"].var[:]) : nothing
    end
    mt2dt(mt) = round(DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000)), Minute(1))
    rec = Dict{DateTime, Tuple{String, Int}}()
    for tag in tags                                     # ascending: a later tag overwrites a shared time
        for (k, t) in enumerate(NCDataset(ds -> mt2dt.(ds["MT"].var[:]), fname(tag, "ssh")))
            inwin(t) && (rec[t] = (tag, k))
        end
    end
    times = sort(collect(keys(rec)))
    @printf("%d file sets %s .. %s, %d records\n", length(tags), tags[1], tags[end], length(times))
    return (; lon, lat, dep, tags, fname, rec, times)
end

function hn_records(f, src, st, pieces)
    bytag = Dict{String, Vector{Tuple{DateTime, Int}}}()
    for t in st.times; (tag, k) = st.rec[t]; push!(get!(bytag, tag, Tuple{DateTime, Int}[]), (t, k)); end
    for tag in sort(collect(keys(bytag)))
        ds = Dict(k => NCDataset(st.fname(tag, k)) for k in ("ssh", "ts", "uv"))
        dl = src.kind == :ncom ? NCDataset(st.fname(tag, "lthick")) : nothing
        try
            zz = map(pieces) do pc                     # level depths (bx, by, nl) and bottoms of this tag
                if src.kind == :hycom
                    nb = (length(pc.i), length(pc.j))
                    (z = repeat(reshape(-st.dep, 1, 1, :), nb..., 1), zb = fill(NaN, nb...))
                else
                    th = nanbig.(dl["layer_thickness"].var[pc.i, pc.j, :, 1]); th[isnan.(th)] .= 0
                    cs = cumsum(th; dims = 3)
                    (z = -(cs .- th ./ 2), zb = -cs[:, :, end])
                end
            end
            for (t, k) in sort(bytag[tag])
                boxes = map(zip(pieces, zz)) do (pc, Z)
                    I, J = pc.i, pc.j
                    (; z = Z.z, zb = Z.zb,
                     temp = nanbig.(ds["ts"]["layer_temperature"].var[I, J, :, k]),
                     salt = nanbig.(ds["ts"]["layer_salinity"].var[I, J, :, k]),
                     u = nanbig.(ds["uv"]["u_velocity"].var[I, J, :, k]),
                     v = nanbig.(ds["uv"]["v_velocity"].var[I, J, :, k]),
                     ssh = nanbig.(ds["ssh"]["ssh"].var[I, J, k]))
                end
                f(t, boxes)
            end
        finally
            foreach(close, values(ds)); dl === nothing || close(dl)
        end
    end
end

# CMEMS: one big file per variable; ssh (zos) hourly on its own time axis
const CM = (temp = ("temp", "thetao"), salt = ("salt", "so"), u = ("uv", "uo"), v = ("uv", "vo"), ssh = ("zos", "zos"))
cm_file(src, key) = joinpath(src.data, only(filter(n -> startswith(n, "cmems_$(key)_") && endswith(n, ".nc"), readdir(src.data))))
function cmems_setup(src)
    lo, la, dep, t3 = NCDataset(cm_file(src, "uv")) do ds
        Float64.(ds["longitude"][:]), Float64.(ds["latitude"][:]), Float64.(ds["depth"][:]), round.(ds["time"][:], Minute(1))
    end
    for key in ("temp", "salt")
        NCDataset(ds -> round.(ds["time"][:], Minute(1)), cm_file(src, key)) == t3 || error("CMEMS $key: other time axis than uv")
    end
    tz = NCDataset(ds -> round.(ds["time"][:], Minute(1)), cm_file(src, "zos"))
    lon = repeat(lon180.(lo), 1, length(la)); lat = repeat(la', length(lo), 1)
    return (; lon, lat, dep, k3 = findall(inwin, t3), t3, kz = findall(inwin, tz), tz)
end

function cmems_records(f, src, st, pieces; ssh_only = false)
    keys_ = ssh_only ? (:ssh,) : (:temp, :salt, :u, :v)
    ds = Dict(k => NCDataset(cm_file(src, CM[k][1])) for k in keys_)
    try
        for k in (ssh_only ? st.kz : st.k3)
            boxes = map(pieces) do pc
                I, J = pc.i, pc.j
                if ssh_only
                    (; ssh = nanbig.(coalesce.(ds[:ssh]["zos"][I, J, 1, k], NaN)))
                else
                    nb = (length(I), length(J))
                    rd(s) = nanbig.(coalesce.(ds[s][CM[s][2]][I, J, :, k], NaN))
                    (; z = repeat(reshape(-st.dep, 1, 1, :), nb..., 1), zb = fill(NaN, nb...),
                     temp = rd(:temp), salt = rd(:salt), u = rd(:u), v = rd(:v), ssh = fill(NaN, nb...))
                end
            end
            f(ssh_only ? st.tz[k] : st.t3[k], boxes)
        end
    finally
        foreach(close, values(ds))
    end
end

## --- ROMS moorings from the station output
function roms_station(src, name, st)
    f = joinpath(src.station, "$name.nc")
    isfile(f) || (@warn "$name: no station file $f"; return nothing)
    ii, jj, ang = NCDataset(ds -> ds["$(name)_mor"][1, :], src.mor)  # fractional object index, angle [rad]
    x, y = ii + 1.5, jj + 1.5                           # -> Julia rho index (as in mooring_comp.jl)
    i0, j0 = floor(Int, x), floor(Int, y); cx, cy = x - i0, y - j0
    bil(A) = (1 - cx) * (1 - cy) * A[i0, j0] + cx * (1 - cy) * A[i0+1, j0] + (1 - cx) * cy * A[i0, j0+1] + cx * cy * A[i0+1, j0+1]
    h = bil(st.h)
    NCDataset(f) do ds
        t = [T_REF + Millisecond(round(Int, 1000 * s)) for s in ds["ocean_time"].var[:]]
        it = findall(inwin, t)
        zeta = Float64.(ds["zeta"].var[1, it])
        rd(v) = Float64.(ds[v].var[1, :, it])           # (N, nt)
        temp, salt, u, v = rd("temp"), rd("salt"), rd("u"), rd("v")
        ue = u .* cos(ang) .- v .* sin(ang); vn = u .* sin(ang) .+ v .* cos(ang)
        nt = length(it)
        z = zlevs3(fill(h, 1, nt), reshape(zeta, 1, nt), st.vc.theta_s, st.vc.theta_b, st.vc.hc, size(temp, 1), "r", "new2008")[1][:, 1, :]
        prof(F) = reduce(hcat, [Float32.(first(vinterp(view(z, :, n), view(F, :, n), ZOUT, -h))) for n in 1:nt])
        (lon = bil(st.lon), lat = bil(st.lat), h, time = t[it], temp = prof(temp), salt = prof(salt),
         u = prof(ue), v = prof(vn), ssh = Float32.(zeta), note = "ROMS station output $f (s-levels -> zout with zlevs3)")
    end
end

## --- output
function save_mooring(fname, m, sname, src_note; time_ssh = nothing)
    mkpath(dirname(fname))
    NCDataset(fname, "c") do ds
        ds.attrib["title"]   = "$sname at mooring $(m.name)"
        ds.attrib["station"] = m.name
        ds.attrib["source"]  = "extract_mooring_all.jl; $src_note"
        defDim(ds, "z", length(ZOUT)); defDim(ds, "time", length(m.time))
        defVar(ds, "lon", m.lon, (), attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", m.lat, (), attrib = ["units" => "degrees_north"])
        defVar(ds, "h", m.h, (), attrib = ["long_name" => "water depth at the point", "units" => "m"])
        defVar(ds, "z", ZOUT, ("z",), attrib = ["units" => "m", "positive" => "up"])
        defVar(ds, "time", m.time, ("time",), attrib = ["units" => "hours since 2022-01-01 00:00:00"])
        for (v, ln, u) in (("temp", "potential temperature", "degC"), ("salt", "salinity", "psu"),
                           ("u", "eastward velocity", "m/s"), ("v", "northward velocity", "m/s"))
            defVar(ds, v, getfield(m, Symbol(v)), ("z", "time"); attrib = ["long_name" => ln, "units" => u])
        end
        if time_ssh === nothing
            defVar(ds, "ssh", m.ssh, ("time",), attrib = ["long_name" => "sea surface height", "units" => "m"])
        else
            defDim(ds, "time_ssh", length(time_ssh))
            defVar(ds, "time_ssh", time_ssh, ("time_ssh",), attrib = ["units" => "hours since 2022-01-01 00:00:00"])
            defVar(ds, "ssh", m.ssh, ("time_ssh",), attrib = ["long_name" => "sea surface height", "units" => "m"])
        end
    end
    println("  saved ", fname, "  ($(length(m.time)) records)")
end

## --- one source
function run_source(sname, src)
    t_start = time()
    out = joinpath(OUT_DIR, sname)
    st = src.kind == :roms ? roms_setup(src) : src.kind == :cmems ? cmems_setup(src) : hn_setup(src)
    note = "$sname, data $(src.data)"

    # ROMS moorings: station output
    if src.kind == :roms
        for k in MOOR_SEL
            m = roms_station(src, moor_name[k], st); m === nothing && continue
            save_mooring(joinpath(out, "$(sname)_$(moor_name[k]).nc"), merge((name = moor_name[k],), m), sname, m.note)
        end
    end

    # gridded points: moorings (non-ROMS) + transect
    pts = Tuple{Float64, Float64}[]; groups = Vector{Vector{Int}}()
    mk = src.kind == :roms ? Int[] : MOOR_SEL
    for k in mk; push!(pts, (moor_lon[k], moor_lat[k])); push!(groups, [length(pts)]); end
    ntr0 = length(pts)
    if DO_TR
        append!(pts, zip(TR.lon, TR.lat))
        np_piece = max(1, round(Int, PIECE_KM / DX_KM))
        append!(groups, [collect(r) for r in Iterators.partition(ntr0 + 1:length(pts), np_piece)])
    end
    if isempty(pts)
        @printf("===== %s done, %.1f min\n", sname, (time() - t_start) / 60); return
    end
    cells = [locate(st.lon, st.lat, x, y) for (x, y) in pts]
    pieces = make_pieces(cells, groups)
    @printf("  %d gridded points in %d pieces (largest box %d x %d)\n", length(pts), length(pieces),
            maximum(length(p.i) for p in pieces), maximum(length(p.j) for p in pieces))

    nz, npt = length(ZOUT), length(pts)
    time_out = DateTime[]
    acc = Dict(v => Array{Float32, 2}[] for v in ("temp", "salt", "u", "v"))
    ssh = Vector{Float32}[]; hpt = fill(NaN, npt)
    function take(t, boxes)
        push!(time_out, t)
        rec = Dict(v => fill(NaN32, nz, npt) for v in keys(acc)); s = fill(NaN32, npt)
        for (pc, bx) in zip(pieces, boxes), p in pc.pts
            ci, cj, w = cells[p]; ci -= first(pc.i) - 1; cj -= first(pc.j) - 1
            for v in keys(acc)
                prof, hb = point_profile(getfield(bx, Symbol(v)), bx.z, bx.zb, ci, cj, w)
                rec[v][:, p] = prof
                v == "temp" && isnan(hpt[p]) && (hpt[p] = -hb)
            end
            s[p] = point_value(bx.ssh, ci, cj, w)
        end
        for v in keys(acc); push!(acc[v], rec[v]); end
        push!(ssh, s)
        length(time_out) % 200 == 0 && @printf("  %s  %d records, %.1f min\n", t, length(time_out), (time() - t_start) / 60)
    end
    src.kind == :roms ? roms_records(take, src, st, pieces) :
    src.kind == :cmems ? cmems_records(take, src, st, pieces) : hn_records(take, src, st, pieces)
    isempty(time_out) && error("$sname: no records in the window")

    # CMEMS: ssh again on its own (hourly) axis
    time_ssh = nothing
    if src.kind == :cmems
        time_ssh = DateTime[]; ssh = Vector{Float32}[]
        cmems_records(src, st, pieces; ssh_only = true) do t, boxes
            push!(time_ssh, t); s = fill(NaN32, npt)
            for (pc, bx) in zip(pieces, boxes), p in pc.pts
                ci, cj, w = cells[p]
                s[p] = point_value(bx.ssh, ci - first(pc.i) + 1, cj - first(pc.j) + 1, w)
            end
            push!(ssh, s)
        end
    end

    o = sortperm(time_out)                              # records come file by file
    time_out = time_out[o]
    F = Dict(v => stack(acc[v])[:, :, o] for v in keys(acc))   # (z, point, time)
    S = stack(ssh); time_ssh === nothing && (S = S[:, o])      # (point, time)
    acc = nothing; GC.gc()

    for (n, k) in enumerate(mk)
        m = (name = moor_name[k], lon = moor_lon[k], lat = moor_lat[k], h = hpt[n], time = time_out,
             temp = F["temp"][:, n, :], salt = F["salt"][:, n, :], u = F["u"][:, n, :], v = F["v"][:, n, :], ssh = S[n, :])
        save_mooring(joinpath(out, "$(sname)_$(moor_name[k]).nc"), m, sname, note; time_ssh)
    end
    if DO_TR
        r = ntr0 + 1:npt
        title = "$sname interpolated to the $(TR.from) - $(TR.to) transect"
        save_transect(joinpath(out, "$(sname)_$(TR.name).nc"), TR, ZOUT, time_out,
                      ["temp" => (F["temp"][:, r, :], "potential temperature", "degC"),
                       "salt" => (F["salt"][:, r, :], "salinity", "psu"),
                       "u"    => (F["u"][:, r, :], "eastward velocity", "m/s"),
                       "v"    => (F["v"][:, r, :], "northward velocity", "m/s")];
                      ssh = time_ssh === nothing ? S[r, :] : nothing, h = hpt[r],
                      title, source = "extract_mooring_all.jl; $note")
        time_ssh === nothing ||
            save_transect(joinpath(out, "$(sname)_$(TR.name)_ssh.nc"), TR, nothing, time_ssh, [];
                          vars2d = ["ssh" => (S[r, :], "sea surface height", "m")],
                          title = "$sname sea surface height on the $(TR.from) - $(TR.to) transect",
                          source = "extract_mooring_all.jl; $note")
    end
    @printf("===== %s done, %.1f min\n", sname, (time() - t_start) / 60)
end

for sname in SRC_RUN
    src = Dict(SOURCES)[sname]
    println("\n===== ", sname, ": ", src.data)
    isdir(src.data) || (println("  skipped, no data folder on $HOST"); continue)
    run_source(sname, src)
end
