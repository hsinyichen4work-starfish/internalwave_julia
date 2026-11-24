#= m2_modes_fun.jl
Hsin-Yi Chen, USM DMS
Settings, readers and column functions for m2_modes_models.jl (loaded on every worker). Modal decomposition and
tidal harmonic fit of the baroclinic pressure and velocity for several models, the same method as
DFF/bcef_modes_puv.jl + DFF/mode_bcef_harmonic.jl, but one code for every model:
  - density from T, S with TEOS-10 for every model (GibbsSeaWater: SA = SP 35.16504 / 35, CT from potential T,
    in-situ rho at the level's actual depth), so the models differ only in their fields, not in their equations
  - columns ordered bottom -> top (k = 1 bottom), levels below the model bottom dropped per column
  - u, v east / north for every model (ROMS rotated by angle; HYCOM, NCOM, GLORYS already east / north)

Settings (environment):
  M2_HOST    expanse / aurelia (default: expanse if /expanse/lustre exists)
  M2_SRC     sources, comma separated (default per machine: the data that lives there)
             ROMS_HYCOMparent, ROMS_GLORYSparent, GLORYS (expanse);  HYCOM, NCOM2700 (aurelia)
             a new ROMS run on the 2700 m grid: M2_SRC=ROMS:<run folder>  (e.g. ROMS:design_1201_hycomdetide_2700)
  M2_T0, M2_T1   window (default 2022-08-25 .. 2022-11-30, as the SSH tide comparison)
  M2_NMODE   modes (default 3)
  M2_CONS    constituents fitted (default M2,S2,N2,K1,O1; GLORYS drops S2: 6-hourly samples hit its Nyquist frequency)
  M2_STRIDE  every n-th horizontal point (tests); files get _s<n>
  M2_OUT     output folder
=#

using NCDatasets, Dates, Statistics, LinearAlgebra, Printf, GibbsSeaWater

const HOST = get(ENV, "M2_HOST", isdir("/expanse/lustre") ? "expanse" : "aurelia")
const P = HOST == "expanse" ? (
        func  = "/home/hchen54/internalwave_julia/function/",
        base  = "/expanse/lustre/projects/uso102/hchen54/",
        pdata = "/home/hchen54/aurelia/roms_data/",                    # HYCOM / NCOM over sshfs: run those on aurelia
        grid  = "/expanse/lustre/projects/uso102/hchen54/new_design_fix/roms_grd_2700m_1104.nc",
        out   = "/expanse/lustre/projects/uso102/hchen54/figure/m2_modes_models/",
        default = "ROMS_HYCOMparent,ROMS_GLORYSparent,GLORYS") :
    HOST == "aurelia" ? (
        func  = "/home/hsinyi/Documents/Julia/function/",
        base  = "/home/hsinyi/roms_data/",
        pdata = "/home/hsinyi/roms_data/",
        grid  = "/home/hsinyi/roms_data/new_design_fix/roms_grd_2700m_1104.nc",
        out   = "/home/hsinyi/roms_data/m2_modes_models/",
        default = "HYCOM,NCOM2700") :
    error("M2_HOST = $HOST, not expanse / aurelia")
include(P.func * "harm_fit.jl")                           # TIDE_FREQ

const SOURCES = Dict(
    "ROMS_HYCOMparent"  => (kind = :roms, data = P.base * "design_1113_hycom_2700/after_joint/", label = "ROMS 2700 m (HYCOM parent)"),
    "ROMS_GLORYSparent" => (kind = :roms, data = P.base * "design_1116_cmems_2700/after_joint/", label = "ROMS 2700 m (GLORYS parent)"),
    "HYCOM"    => (kind = :hycom, data = P.pdata * "HYCOM_DATA_NC_2700m_1104/", prefix = "hycom_", label = "HYCOM"),
    "NCOM2700" => (kind = :ncom,  data = P.pdata * "NCOM_DATA_NC_2700/", prefix = "", label = "NCOM"),
    "GLORYS"   => (kind = :cmems, data = P.base * "CMEMS_DATA/", label = "GLORYS (CMEMS GLO12)"),
)
function source(name)
    startswith(name, "ROMS:") && return (kind = :roms, data = P.base * name[6:end] * "/after_joint/", label = "ROMS 2700 m ($(name[6:end]))")
    haskey(SOURCES, name) || error("M2_SRC: $name not one of $(join(keys(SOURCES), ", ")) or ROMS:<run>")
    return SOURCES[name]
end
srcname(name) = replace(name, "ROMS:" => "ROMS_")          # file name of a ROMS:<run> source

const T0     = DateTime(get(ENV, "M2_T0", "2022-08-25T00:00:00"))
const T1     = DateTime(get(ENV, "M2_T1", "2022-11-30T00:00:00"))
const NMODE  = parse(Int, get(ENV, "M2_NMODE", "3"))
const STRIDE = parse(Int, get(ENV, "M2_STRIDE", "1"))
const SFX    = STRIDE > 1 ? "_s$STRIDE" : ""
const OUT    = get(ENV, "M2_OUT", P.out)
const T_REF  = DateTime(1994, 1, 1)
const RHO0   = 1025.0                                      # kg m-3, for HKE and APE
const GRAV   = 9.81
const MARGIN = 0.3                                         # deg around the 2700 m grid read from the other models
const NMIN   = 8                                           # columns with fewer valid levels are skipped
const HMIN   = 20.0                                        # m, columns shallower than this are skipped
const P1_DT  = 6.0                                         # h, record spacing used for the mean state (pass 1)
cons_for(kind) = (c = String.(split(get(ENV, "M2_CONS", "M2,S2,N2,K1,O1"), ",")); kind == :cmems ? filter(!=("S2"), c) : c)
lon180(x) = x > 180 ? x - 360 : x
nanbig(x) = (isfinite(x) && abs(x) < 1e10) ? Float64(x) : NaN
tosec(t) = Dates.value(Millisecond(t - T_REF)) / 1000

## target box: the 2700 m ROMS grid (lon -180..180)
const BOX = NCDataset(P.grid) do g
    lo = lon180.(g["lon_rho"][:, :]); la = g["lat_rho"][:, :]
    (lon = (minimum(lo) - MARGIN, maximum(lo) + MARGIN), lat = (minimum(la) - MARGIN, maximum(la) + MARGIN))
end
function boxindex(lon, lat)                                # index ranges of the source points inside BOX
    p = findall((BOX.lon[1] .<= lon .<= BOX.lon[2]) .& (BOX.lat[1] .<= lat .<= BOX.lat[2]))
    I = minimum(first.(Tuple.(p))):maximum(first.(Tuple.(p))); J = minimum(last.(Tuple.(p))):maximum(last.(Tuple.(p)))
    return first(I):STRIDE:last(I), first(J):STRIDE:last(J)
end

## =====================================================================================================
## setup: grid, time line and work units of a source. A unit is one file (or file set) and its records:
##   (; file = <id>, recs = [(record in file, time index n), ...])
## Every reader then gives, per record, bottom -> top columns (nx, ny, nk), Float64:
##   z (level centre depth, m, negative), Hz (thickness, m), PT (potential temperature), SP (practical salinity),
##   u, v (east, north); invalid levels: NaN in PT. kb[i, j] = lowest valid level (0 = no column).
## =====================================================================================================
function setup(name)
    src = source(name); k = src.kind
    st = k == :roms ? setup_roms(src) : k == :cmems ? setup_cmems(src) : setup_hn(src)
    tsec = tosec.(st.times)
    return merge(st, (; name, src, tsec, cons = cons_for(k)))
end

## ROMS: roms_his.*.nc hourly, s-levels (Vtransform 2) moving with zeta
function setup_roms(src)
    files = sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(src.data)))
    isempty(files) && error("no roms_his files in $(src.data)")
    rec = Dict{DateTime, Tuple{Int, Int}}()
    for (fi, f) in enumerate(files)
        for (kk, s) in enumerate(NCDataset(ds -> ds["ocean_time"].var[:], src.data * f))
            t = T_REF + Second(round(Int, s)); T0 <= t <= T1 && (rec[t] = (fi, kk))
        end
    end
    times = sort(collect(keys(rec)))
    units = group_units(times, rec)
    lon, lat, h, mask, ang = NCDataset(P.grid) do g
        g["lon_rho"][:, :], g["lat_rho"][:, :], Float64.(g["h"][:, :]), g["mask_rho"][:, :] .== 1, Float64.(g["angle"][:, :])
    end
    nxf, nyf = size(lon); ix = 1:STRIDE:nxf; jy = 1:STRIDE:nyf
    nk, hc, Cs_r, Cs_w = NCDataset(src.data * files[1]) do ds
        ds.dim["s_rho"], Float64(ds.attrib["hc"]), Float64.(ds.attrib["Cs_r"]), Float64.(ds.attrib["Cs_w"])
    end
    hh = h[ix, jy]
    sc_r = ((1:nk) .- nk .- 0.5) ./ nk; sc_w = ((0:nk) .- nk) ./ nk
    z0_r = (hc .* reshape(sc_r, 1, 1, :) .+ reshape(Cs_r, 1, 1, :) .* hh) ./ (hh .+ hc)
    z0_w = (hc .* reshape(sc_w, 1, 1, :) .+ reshape(Cs_w, 1, 1, :) .* hh) ./ (hh .+ hc)
    kb = ifelse.(mask[ix, jy] .& (hh .>= HMIN), 1, 0)
    return (; lon = lon180.(lon[ix, jy]), lat = lat[ix, jy], times, units, files, nk, kb, ix, jy, nxf, nyf,
              h = hh, z0_r, z0_w, ang = ang[ix, jy])
end

## HYCOM (3-hourly, fixed z-levels) / NCOM (hourly, fixed layers): daily file sets <prefix><tag>_{ts,uv}.nc;
## a record in two files (overlap) is taken from the newer one
function setup_hn(src)
    re = Regex("^" * src.prefix * raw"(\d{10})_ts\.nc$")
    tags = sort([m[1] for m in (match(re, n) for n in readdir(src.data)) if m !== nothing && isfile(src.data * "$(src.prefix)$(m[1])_uv.nc")])
    isempty(tags) && error("no ts / uv file pairs in $(src.data)")
    mt2dt(mt) = round(DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000)), Minute(1))
    rec = Dict{DateTime, Tuple{Int, Int}}()
    for (fi, tag) in enumerate(tags), (kk, t) in enumerate(NCDataset(ds -> mt2dt.(ds["MT"].var[:]), src.data * "$(src.prefix)$(tag)_ts.nc"))
        T0 <= t <= T1 && (rec[t] = (fi, kk))
    end
    times = sort(collect(keys(rec))); units = group_units(times, rec)
    gfile = src.data * "$(src.prefix)$(tags[1])_" * (src.kind == :hycom ? "grd.nc" : "lthick.nc")
    lon, lat = NCDataset(ds -> (lon180.(Float64.(ds["Longitude"].var[:, :])), Float64.(ds["Latitude"].var[:, :])), gfile)
    I, J = boxindex(lon, lat)
    # level depths (top -> bottom, positive) and valid levels per column (from the first record's temperature)
    T1st = NCDataset(ds -> nanbig.(ds["layer_temperature"].var[I, J, :, 1]), src.data * "$(src.prefix)$(tags[1])_ts.nc")
    nx, ny, nk = size(T1st)
    if src.kind == :hycom
        d = NCDataset(ds -> Float64.(ds["depth"].var[:]), gfile)
        nv = [count(isfinite, view(T1st, i, j, :)) for i in 1:nx, j in 1:ny]
        e = zeros(nx, ny, nk + 1)                              # interfaces (positive depth), top -> bottom
        for j in 1:ny, i in 1:nx
            n = nv[i, j]; n < 2 && continue
            for kk in 1:n - 1; e[i, j, kk + 1] = (d[kk] + d[kk + 1]) / 2; end
            e[i, j, n + 1] = d[n] + (d[n] - d[n - 1]) / 2       # half a spacing below the deepest valid level
        end
        Hz_td = max.(diff(e; dims = 3), 0.0); zc_td = repeat(-reshape(d, 1, 1, :), nx, ny, 1)
    else                                                       # NCOM: fixed layer thickness (same every day)
        th = NCDataset(ds -> nanbig.(ds["layer_thickness"].var[I, J, :, 1]), gfile); th[.!isfinite.(th)] .= 0
        nv = [count(kk -> th[i, j, kk] > 0 && isfinite(T1st[i, j, kk]), 1:nk) for i in 1:nx, j in 1:ny]
        cs = cumsum(th; dims = 3); Hz_td = th; zc_td = -(cs .- th ./ 2)
    end
    rev = nk:-1:1                                              # top -> bottom  =>  bottom -> top
    Hz = Hz_td[:, :, rev]; zc = zc_td[:, :, rev]
    kb = [(nv[i, j] >= NMIN && sum(view(Hz_td, i, j, 1:nv[i, j])) >= HMIN) ? nk - nv[i, j] + 1 : 0 for i in 1:nx, j in 1:ny]
    return (; lon = lon[I, J], lat = lat[I, J], times, units, tags, I, J, nk, kb, Hz, zc)
end

## GLORYS / CMEMS GLO12: one file per variable, 6-hourly instantaneous, fixed z-levels, regular lon / lat
cm_file(src, key) = src.data * only(filter(n -> startswith(n, "cmems_$(key)_6h") && endswith(n, ".nc"), readdir(src.data)))
function setup_cmems(src)
    lo, la, d, t = NCDataset(ds -> (Float64.(ds["longitude"][:]), Float64.(ds["latitude"][:]), Float64.(ds["depth"][:]),
                                     round.(ds["time"][:], Minute(1))), cm_file(src, "uv"))
    for key in ("temp", "salt")
        NCDataset(ds -> round.(ds["time"][:], Minute(1)), cm_file(src, key)) == t || error("CMEMS $key: other time axis than uv")
    end
    kt = findall(x -> T0 <= x <= T1, t); times = t[kt]
    units = [(file = 0, recs = [(kt[n], n) for n in r]) for r in Iterators.partition(eachindex(kt), 4)]
    lon = repeat(lon180.(lo), 1, length(la)); lat = repeat(la', length(lo), 1)
    I, J = boxindex(lon, lat)
    T1st = NCDataset(ds -> nanbig.(ds["thetao"].var[I, J, :, kt[1]]), cm_file(src, "temp"))
    nx, ny, nk = size(T1st)
    nv = [count(isfinite, view(T1st, i, j, :)) for i in 1:nx, j in 1:ny]
    e = zeros(nx, ny, nk + 1)
    for j in 1:ny, i in 1:nx
        n = nv[i, j]; n < 2 && continue
        for kk in 1:n - 1; e[i, j, kk + 1] = (d[kk] + d[kk + 1]) / 2; end
        e[i, j, n + 1] = d[n] + (d[n] - d[n - 1]) / 2
    end
    rev = nk:-1:1
    Hz = max.(diff(e; dims = 3), 0.0)[:, :, rev]; zc = repeat(-reshape(d, 1, 1, :), nx, ny, 1)[:, :, rev]
    kb = [(nv[i, j] >= NMIN && sum(view(Hz, i, j, :)) >= HMIN) ? nk - nv[i, j] + 1 : 0 for i in 1:nx, j in 1:ny]
    return (; lon = lon[I, J], lat = lat[I, J], times, units, I, J, nk, kb, Hz, zc)
end

function group_units(times, rec)                             # records grouped by file, in file order
    by = Dict{Int, Vector{Tuple{Int, Int}}}()
    for (n, t) in enumerate(times); (fi, kk) = rec[t]; push!(get!(by, fi, Tuple{Int, Int}[]), (kk, n)); end
    return [(file = fi, recs = sort(by[fi])) for fi in sort(collect(keys(by)))]
end

## =====================================================================================================
## readers: f(n, rec) for every record of a unit, rec = (; z, Hz, PT, SP, u, v) bottom -> top
## =====================================================================================================
function read_unit(f, st, u)
    k = st.src.kind
    k == :roms ? read_roms(f, st, u) : k == :cmems ? read_cmems(f, st, u) : read_hn(f, st, u)
end

function read_roms(f, st, u)
    NCDataset(st.src.data * st.files[u.file]) do ds
        nx, ny = length(st.ix), length(st.jy)
        for (kk, n) in u.recs
            zeta = Float64.(ds["zeta"].var[st.ix, st.jy, kk])
            z = zeta .+ (zeta .+ st.h) .* st.z0_r
            Hz = diff(zeta .+ (zeta .+ st.h) .* st.z0_w; dims = 3)
            PT = Float64.(ds["temp"].var[st.ix, st.jy, :, kk]); SP = Float64.(ds["salt"].var[st.ix, st.jy, :, kk])
            # u, v to rho points (mean of the two neighbours; edge: one), then east / north
            ua = Float64.(ds["u"].var[:, :, :, kk]); va = Float64.(ds["v"].var[:, :, :, kk])
            ur = similar(PT); vr = similar(PT)
            for (jj, j) in enumerate(st.jy), (ii, i) in enumerate(st.ix)
                il, ih = max(i - 1, 1), min(i, st.nxf - 1); jl, jh = max(j - 1, 1), min(j, st.nyf - 1)
                for kz in 1:st.nk
                    ur[ii, jj, kz] = 0.5 * (ua[il, j, kz] + ua[ih, j, kz]); vr[ii, jj, kz] = 0.5 * (va[i, jl, kz] + va[i, jh, kz])
                end
            end
            ca = cos.(st.ang); sa = sin.(st.ang)
            ue = ur .* ca .- vr .* sa; vn = ur .* sa .+ vr .* ca
            f(n, (; z, Hz, PT, SP, u = ue, v = vn))
        end
    end
end

function read_hn(f, st, u)
    tag = st.tags[u.file]; pre = st.src.data * st.src.prefix * tag
    ks = first.(u.recs); r = minimum(ks):maximum(ks); rev = st.nk:-1:1
    NCDataset(pre * "_ts.nc") do dt
        NCDataset(pre * "_uv.nc") do du
            for (kk, n) in u.recs
                rd(ds, v) = nanbig.(ds[v].var[st.I, st.J, :, kk])[:, :, rev]
                f(n, (; z = st.zc, Hz = st.Hz, PT = rd(dt, "layer_temperature"), SP = rd(dt, "layer_salinity"),
                        u = rd(du, "u_velocity"), v = rd(du, "v_velocity")))
            end
        end
    end
end

function read_cmems(f, st, u)
    rev = st.nk:-1:1
    ds = Dict(k => NCDataset(cm_file(st.src, k)) for k in ("temp", "salt", "uv"))
    try
        for (kk, n) in u.recs
            rd(key, v) = nanbig.(ds[key][v].var[st.I, st.J, :, kk])[:, :, rev]
            f(n, (; z = st.zc, Hz = st.Hz, PT = rd("temp", "thetao"), SP = rd("salt", "so"), u = rd("uv", "uo"), v = rd("uv", "vo")))
        end
    finally
        foreach(close, values(ds))
    end
end

## =====================================================================================================
## pass 1: sums of in-situ rho, SA, CT, z, Hz over the records (every column's valid levels)
## =====================================================================================================
sa_of(SP) = SP * 35.16504 / 35
function pass1_unit(st, u)
    nx, ny = size(st.kb); nk = st.nk
    S = (rho = zeros(nx, ny, nk), SA = zeros(nx, ny, nk), CT = zeros(nx, ny, nk), z = zeros(nx, ny, nk), Hz = zeros(nx, ny, nk), n = zeros(1))
    read_unit(st, u) do n, rec
        @inbounds for j in 1:ny, i in 1:nx
            kb = st.kb[i, j]; kb == 0 && continue
            for k in kb:nk
                sa = sa_of(rec.SP[i, j, k]); ct = gsw_ct_from_pt(sa, rec.PT[i, j, k])
                S.rho[i, j, k] += gsw_rho(sa, ct, -rec.z[i, j, k]); S.SA[i, j, k] += sa; S.CT[i, j, k] += ct
                S.z[i, j, k] += rec.z[i, j, k]; S.Hz[i, j, k] += rec.Hz[i, j, k]
            end
        end
        S.n[1] += 1
    end
    return S
end
addsums(a, b) = map(+, a, b)

## =====================================================================================================
## modes of  W'' + (N^2 / c^2) W = 0, W = 0 at bottom and surface (as modes() in DFF/bcef_modes_puv.jl), from the
## mean column: zr, Hz bottom -> top, N2 on the interior interfaces (from GSW, held >= N2min).
## phi_n = dW_n/dz on the levels, (1/H) int phi_m phi_n dz = delta_mn, positive at the surface.
## =====================================================================================================
function modes_N2(zr, Hz, N2; nmode = NMODE, N2min = 1e-7)
    nz = length(zr); dzc = diff(zr)
    B = max.(N2, N2min) .* dzc
    d = (1 ./ Hz[2:nz] .+ 1 ./ Hz[1:nz-1]) ./ B
    e = (-1 ./ Hz[2:nz-1]) ./ sqrt.(B[1:end-1] .* B[2:end])
    F = eigen(SymTridiagonal(d, e), 1:nmode)
    c = 1 ./ sqrt.(F.values); φ = zeros(nz, nmode)
    for n in 1:nmode
        W = [0; F.vectors[:, n] ./ sqrt.(B); 0]
        φn = diff(W) ./ Hz
        φn ./= sqrt(sum(φn .^ 2 .* Hz) / sum(Hz))
        φn[end] < 0 && (φn .*= -1)
        φ[:, n] .= φn
    end
    return c, φ
end

## mean state -> (; rhom, zm, H, c, phi): rhom, zm (nx, ny, nk) mean in-situ density and level depth; phi (nx, ny, nk, NMODE)
function mean_state(st, S)
    nx, ny = size(st.kb); nk = st.nk; n = S.n[1]
    rhom = S.rho ./ n; zm = S.z ./ n; Hzm = S.Hz ./ n; SA = S.SA ./ n; CT = S.CT ./ n
    c = fill(NaN, nx, ny, NMODE); φ = zeros(nx, ny, nk, NMODE); H = fill(NaN, nx, ny)
    kbm = copy(st.kb)
    Threads.@threads for j in 1:ny
        for i in 1:nx
            kb = st.kb[i, j]; kb == 0 && continue
            r = kb:nk
            try
                N2 = gsw_nsquared(SA[i, j, r], CT[i, j, r], -zm[i, j, r], fill(st.lat[i, j], length(r)))[1]
                cc, ph = modes_N2(zm[i, j, r], Hzm[i, j, r], N2)
                c[i, j, :] .= cc; φ[i, j, r, :] .= ph; H[i, j] = sum(Hzm[i, j, r])
            catch
                kbm[i, j] = 0                                   # degenerate column: skipped
            end
        end
    end
    return (; rhom, zm, H, c, φ, kb = kbm)
end

## =====================================================================================================
## pass 2: per record, p', u', v' -> modal amplitudes pn, un, vn, accumulated into the normal equations
##   B[f, q, i, j] += y_f(t) a_q(t),  a(t) = [1, (t - tc)/span, cos(w_k th), sin(w_k th), ...]  (as harm_fit)
## f = (p1, u1, v1, p2, u2, v2, ...); one rho' profile per column: rho(SA, CT, p at the actual depth) minus the mean
## in-situ profile at that depth (linear in the mean level depths), hydrostatic p' from the surface, depth means removed.
## =====================================================================================================
function design_row(t, tc, span, cons)
    th = (t - tc) / 3600
    a = zeros(2 + 2length(cons)); a[1] = 1; a[2] = (t - tc) / span
    for (k, c) in enumerate(cons); ph = deg2rad(TIDE_FREQ[c]) * th; a[2k + 1] = cos(ph); a[2k + 2] = sin(ph); end
    return a
end

function pass2_unit(st, ms, u, tc, span)
    nx, ny = size(st.kb); nk = st.nk; nf = 3NMODE; np = 2 + 2length(st.cons)
    B = zeros(nf, np, nx, ny); SS = zeros(nf, nx, ny)       # also sum of y^2 (residual variance)
    NB = zeros(nx, ny)                                      # records with a missing value (column dropped later)
    rp = zeros(nk); pp = zeros(nk); y = zeros(nf)
    read_unit(st, u) do n, rec
        a = design_row(st.tsec[n], tc, span, st.cons)
        @inbounds for j in 1:ny, i in 1:nx
            kb = ms.kb[i, j]; kb == 0 && continue
            ok = true
            for k in kb:nk
                isfinite(rec.PT[i, j, k]) && isfinite(rec.SP[i, j, k]) && isfinite(rec.u[i, j, k]) && isfinite(rec.v[i, j, k]) || (ok = false; break)
            end
            ok || (NB[i, j] += 1; continue)
            # rho' at the actual depth
            m = kb
            for k in kb:nk
                zz = rec.z[i, j, k]
                while m < nk - 1 && ms.zm[i, j, m + 1] < zz; m += 1; end
                w = clamp((zz - ms.zm[i, j, m]) / (ms.zm[i, j, m + 1] - ms.zm[i, j, m]), 0.0, 1.0)
                rbar = ms.rhom[i, j, m] + w * (ms.rhom[i, j, m + 1] - ms.rhom[i, j, m])
                sa = sa_of(rec.SP[i, j, k]); ct = gsw_ct_from_pt(sa, rec.PT[i, j, k])
                rp[k] = gsw_rho(sa, ct, -zz) - rbar
            end
            # hydrostatic p' from the surface down, then depth means removed and projections
            ptop = 0.0; Hc = 0.0
            for k in nk:-1:kb
                dp = GRAV * rp[k] * rec.Hz[i, j, k]; pp[k] = ptop + 0.5dp; ptop += dp; Hc += rec.Hz[i, j, k]
            end
            mp = 0.0; mu = 0.0; mv = 0.0
            for k in kb:nk; hz = rec.Hz[i, j, k]; mp += pp[k] * hz; mu += rec.u[i, j, k] * hz; mv += rec.v[i, j, k] * hz; end
            mp /= Hc; mu /= Hc; mv /= Hc
            for q in 1:NMODE
                sp = 0.0; su = 0.0; sv = 0.0
                for k in kb:nk
                    w = ms.φ[i, j, k, q] * rec.Hz[i, j, k]
                    sp += (pp[k] - mp) * w; su += (rec.u[i, j, k] - mu) * w; sv += (rec.v[i, j, k] - mv) * w
                end
                y[3q - 2] = sp / Hc; y[3q - 1] = su / Hc; y[3q] = sv / Hc
            end
            for f in 1:nf
                SS[f, i, j] += y[f]^2
                for q in 1:np; B[f, q, i, j] += y[f] * a[q]; end
            end
        end
    end
    return (B = B, SS = SS, NB = NB)
end
