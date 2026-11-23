#= mean_flow_avg_loop.jl
Hsin-Yi Chen, USM DMS
Monthly and half-monthly means (full 3-D fields + surface maps) for three ROMS runs and the three parent models,
in one loop.
Same method as mean_flow_avg.jl (running sums, every file read once), extended to
  - two averaging windows:  monthly      one bin per calendar month
                            halfmonthly  days 1-15 and 16-end of each month
    (+ the whole-record "total" in both). Both windows come out of one pass over the data.
  - six sources (MF_SRC picks some, comma separated; default = all, in this order):
      design_1113_hycom_2700  ROMS 2700 m (grid 1104), HYCOM bry, daily averages
      design_1116_cmems_2700  ROMS 2700 m (grid 1104), CMEMS bry, daily averages
      amazon_900m_dbry_0927   ROMS 900 m, hourly averages (files in the run folder itself, no after_joint)
      CMEMS                   CMEMS_DATA/cmems_*_0731_1130_big.nc: uo/vo/thetao/so 6-hourly, zos hourly,
                              regular lon/lat, depth level 1 (0.49 m)
      HYCOM                   HYCOM_DATA_NC_2700m_1104/hycom_<yyyymmddHH>_{ssh,uv,ts}.nc: 3-hourly, z = 1 (0 m),
                              lon/lat from hycom_*_grd.nc
      NCOM                    NCOM_DATA_NC_2700/<yyyymmddHH>_{ssh,uv,ts}.nc: hourly (tides included; they
                              average out over >= 15 d), top layer, lon/lat from <tag>_lthick.nc
    HYCOM / NCOM files overlap by one record (last of a day = first of the next): every time is used once.
    Parent u, v are east / north already; ROMS u, v are moved to rho points and rotated by angle.
  Only records in [T0, T1) are used (default Aug 1 - Dec 1 2022), so the calendar bins line up across sources;
  a bin a source only partly covers shows its own day count in the panel title.

Running sums are kept per grid point and per half-month only (Float64 sum, UInt16 count of finite values);
a month = its two halves, total = all halves (sum of sums / sum of counts, so exactly the same as averaging the
records directly). NaN land / below-bottom / fill (~1e37) in the parent files never enters a mean; ROMS land
(stored as 0) -> NaN by mask_rho/u/v. Variables are done one at a time (peak memory ~ 10 B x 3-D field x halves:
~3.5 GB 2700 m runs, ~2.5 GB 900 m run, ~7 GB NCOM).
3-D = every native level: ROMS s-levels (s_rho, bottom -> top; z from zlevs3 with h, mean zeta and the
vertical-coordinate attributes copied from roms_avg), HYCOM / CMEMS fixed depths (depth), NCOM layers (mean
layer thickness lthick per bin, from the daily <tag>_lthick.nc). MF_3D=0: surface only (fast, as before).

Output per source (paths per machine, see P below):
  OUT_DIR/<src>_meanflow_avg_3d.nc              all bins, native grids: zeta, u, v, temp, salt (+ lthick for NCOM);
                                                  ROMS u / v on their own u / v points along xi / eta (not rotated),
                                                  parent u / v east / north
  OUT_DIR/<src>_meanflow_avg.nc                         all bins: zeta, u_east, v_north, temp, salt on rho points,
                                                  lon, lat, bin name / start / end / record count
  FIG_ROOT/<src>/mean_flow/<window>/<view>/meanflow_avg_surf_<field>.png   (ssh, temp, salt, speed)
    views: surface/       this source's map extent (parents: the 2700 m 1104 ROMS grid extent)
           surface_ncom/  fixed NCOM parent box (same for every source)
  copies in CHECK_DIR as <src>_<window>_<view>_meanflow_avg_surf_<field>.png

Machines (auto-detected, or MF_HOST=expanse / aurelia):
  expanse  ROMS runs + CMEMS on lustre; default MF_SRC = the three runs + CMEMS
           (HYCOM / NCOM are reachable only through the slow sshfs mount ~/aurelia: run them on aurelia)
  aurelia  HYCOM / NCOM on local disk (/home/hsinyi/roms_data); default MF_SRC = HYCOM,NCOM
  A source whose data folder is missing on this machine is skipped with a message.

Usage:
  expanse:  sbatch sbatch_files/julia_meanflow_avg_loop.sh            # 3 runs + CMEMS, job array 0-3
            MF_SRC=CMEMS MF_WIN=halfmonthly julia --project=. meanflow/mean_flow_avg_loop.jl
  aurelia:  (copy this file to ~/Documents/Julia/meanflow/, then from ~/Documents/Julia)
            nohup julia --project=. meanflow/mean_flow_avg_loop.jl > meanflow_loop_parents.log 2>&1 &
            MF_SRC=NCOM julia --project=. meanflow/mean_flow_avg_loop.jl
  test:     MF_T0=2022-09-14 MF_T1=2022-09-17 MF_OUTDIR=/tmp/x MF_FIGROOT=/tmp/x MF_NOCHECK=1 MF_SRC=HYCOM julia ...
Expanse login node: 8 GB limit; use the batch job.
=#

using NCDatasets, Printf, Statistics, Dates

## --- machine: paths
const HOST = get(ENV, "MF_HOST", isdir("/expanse/lustre") ? "expanse" : "aurelia")
const P = HOST == "expanse" ? (
        func  = "/home/hchen54/internalwave_julia/function/",
        base  = "/expanse/lustre/projects/uso102/hchen54/",               # ROMS runs, CMEMS_DATA, grids
        pdata = "/home/hchen54/aurelia/roms_data/",                       # HYCOM / NCOM (sshfs: slow)
        grid27 = "/expanse/lustre/projects/uso102/hchen54/new_design_fix/",
        out   = "/expanse/lustre/projects/uso102/hchen54/meanflow_avg",
        fig   = "/expanse/lustre/projects/uso102/hchen54/figure",
        check = expanduser("~/figure_check"),
        default = "design_1113_hycom_2700,design_1116_cmems_2700,amazon_900m_dbry_0927,CMEMS") :
    HOST == "aurelia" ? (
        func  = "/home/hsinyi/Documents/Julia/function/",
        base  = "/home/hsinyi/roms_data/",                                 # no ROMS runs / CMEMS here
        pdata = "/home/hsinyi/roms_data/",
        grid27 = "/home/hsinyi/roms_data/new_design_fix/",
        out   = "/home/hsinyi/roms_data/meanflow_avg",
        fig   = "/home/hsinyi/figure",
        check = nothing,
        default = "HYCOM,NCOM") :
    error("MF_HOST = $HOST, not expanse / aurelia")
println("host ", HOST)

for f in ("fig_setup.jl", "uvp_masks.jl", "uv2rho.jl", "time_dir.jl",    # figkw, figsave, cm_to_pt; masks; u2rho; time_dir
          "plotting_fun.jl", "mooring_transect.jl")                        # CairoMakie, topdown_axis, *_curvilinear!; read_moorings
    include(joinpath(P.func, f))
end

## --- sources
const BASE = P.base
const GRID_2700 = P.grid27 * "roms_grd_2700m_1104.nc"                    # parents: map extent, 200 m isobath, outline
const MOOR_2700 = P.grid27 * "roms_grd_2700m_1104_mor_edata.nc"
const SOURCES = [
    "design_1113_hycom_2700" => (kind = :roms, data = BASE * "design_1113_hycom_2700/after_joint/",
                                 grid = GRID_2700, moor = MOOR_2700, dt = 86400, what = "daily ROMS averages"),
    "design_1116_cmems_2700" => (kind = :roms, data = BASE * "design_1116_cmems_2700/after_joint/",
                                 grid = GRID_2700, moor = MOOR_2700, dt = 86400, what = "daily ROMS averages"),
    "amazon_900m_dbry_0927"  => (kind = :roms, data = BASE * "amazon_900m_dbry_0927/",
                                 grid = BASE * "input/grid/roms_grd_900m.nc",
                                 moor = BASE * "input/grid/roms_grd_900m_mor_edata.nc", dt = 3600, what = "hourly ROMS averages"),
    "CMEMS" => (kind = :cmems, data = BASE * "CMEMS_DATA/", moor = MOOR_2700, dt = 6 * 3600,
                what = "CMEMS 6-hourly u, v, T, S and hourly ssh"),
    "HYCOM" => (kind = :hycom, data = P.pdata * "HYCOM_DATA_NC_2700m_1104/", prefix = "hycom_", moor = MOOR_2700,
                dt = 3 * 3600, what = "HYCOM 3-hourly fields"),
    "NCOM"  => (kind = :ncom, data = P.pdata * "NCOM_DATA_NC_2700/", prefix = "", moor = MOOR_2700,
                dt = 3600, what = "NCOM hourly fields (tides averaged out)"),
]

## --- settings
const SRC_RUN  = split(get(ENV, "MF_SRC", P.default), ",")
const WINDOWS  = split(get(ENV, "MF_WIN", "monthly,halfmonthly"), ",")
const T0       = DateTime(get(ENV, "MF_T0", "2022-08-01"))   # records used: T0 <= t < T1
const T1       = DateTime(get(ENV, "MF_T1", "2022-12-01"))
const t_ref    = DateTime(1994, 1, 1)                  # ROMS ocean_time reference
const VARS     = ["zeta", "u", "v", "temp", "salt"]    # fields averaged (NCOM 3-D: + lthick)
const FULL3D   = get(ENV, "MF_3D", "1") != "0"         # false: surface level only, no 3-D file
const OUT_DIR  = get(ENV, "MF_OUTDIR", P.out)          # means NetCDF
const FIG_ROOT = get(ENV, "MF_FIGROOT", P.fig)         # figures in FIG_ROOT/<src>/mean_flow/...
const CHECK_DIR = haskey(ENV, "MF_NOCHECK") ? nothing : P.check   # copy of every figure; nothing = no copies
for s in SRC_RUN; any(==(s), first.(SOURCES)) || error("MF_SRC: $s not one of $(first.(SOURCES))"); end
for w in WINDOWS; w in ("monthly", "halfmonthly") || error("MF_WIN: $w not monthly / halfmonthly"); end

## --- averaging bins: every month and half-month in [T0, T1) + total (empty ones are dropped later)
function all_bins()
    bins = NamedTuple{(:name, :kind, :t0, :t1), Tuple{String, Symbol, DateTime, DateTime}}[]
    m = DateTime(year(T0), month(T0))
    while m < T1
        e = m + Month(1)
        push!(bins, (name = Dates.format(m, "yyyy-mm"), kind = :monthly, t0 = m, t1 = e))
        push!(bins, (name = Dates.format(m, "mm-dd") * "–15", kind = :halfmonthly, t0 = m, t1 = m + Day(15)))
        push!(bins, (name = Dates.format(m + Day(15), "mm-dd") * "–" * string(daysinmonth(m)), kind = :halfmonthly,
                     t0 = m + Day(15), t1 = e))
        m = e
    end
    push!(bins, (name = "total", kind = :total, t0 = T0, t1 = T1))
    return bins
end

## --- accumulation: one running sum per half-month; months and total are combined from the halves
mutable struct Half{N}
    S::Array{Float64, N}; C::Array{UInt16, N}      # per point sum and count of finite values
    nrec::Int; tfirst::DateTime; tlast::DateTime
end
Half(sz, t) = Half(zeros(sz...), zeros(UInt16, sz...), 0, t, t)
halfkey(t) = (year(t), month(t), day(t) <= 15 ? 1 : 2)
halfstart(k) = DateTime(k[1], k[2], k[3] == 1 ? 1 : 16)

function addrec!(h::Half, x::AbstractArray, t::DateTime)
    S, C = h.S, h.C
    @inbounds for I in eachindex(S, x)
        xi = x[I]
        if isfinite(xi) && abs(xi) < 1f30
            S[I] += xi; C[I] += 0x0001
        end
    end
    h.nrec += 1; h.tfirst = min(h.tfirst, t); h.tlast = max(h.tlast, t)
end

# halves that make up bin b (a half lies wholly inside its month)
members(halves, b) = [h for (k, h) in sort(collect(halves); by = first)
                      if b.kind == :total || b.t0 <= halfstart(k) < b.t1]

function combine(hs)                                   # mean over all records of the halves hs
    S = copy(hs[1].S); C = Int32.(hs[1].C)
    for h in hs[2:end]; S .+= h.S; C .+= h.C; end
    return Float32.(S ./ C)                            # count 0 -> NaN
end

function masknan!(m, mk)                               # ROMS land -> NaN, on every level
    land = mk .== 0
    for k in 1:size(m, 3); view(m, :, :, k)[land] .= NaN32; end
    return m
end

# f(times, x) for every block of records of variable v; x is (x, y, record), times in [T0, T1)
inwin(t) = T0 <= t < T1

function roms_blocks(f, src, v, st)
    for (fi, recs) in st.byfile
        kr = minimum(first.(recs)):maximum(first.(recs))
        x = NCDataset(joinpath(src.data, st.td.files[fi])) do ds
            v == "zeta" ? ds[v].var[:, :, kr] : ds[v].var[:, :, st.lev, kr]
        end
        ks = first.(recs) .- first(kr) .+ 1
        f(st.tmid[last.(recs)], selectdim(x, ndims(x), ks))
    end
end

const CM_VAR = Dict("zeta" => ("zos", "zos"), "u" => ("uv", "uo"), "v" => ("uv", "vo"),
                    "temp" => ("temp", "thetao"), "salt" => ("salt", "so"))
function cmems_blocks(f, src, v, st; nblock = FULL3D ? 8 : 48)
    key, name = CM_VAR[v]
    file = only(filter(n -> startswith(n, "cmems_$(key)_") && endswith(n, ".nc"), readdir(src.data)))
    NCDataset(joinpath(src.data, file)) do ds
        t  = round.(ds["time"][:], Minute(1))
        ks = findall(inwin, t)
        for r in Iterators.partition(ks, nblock)                    # ks contiguous (time sorted)
            kr = first(r):last(r)
            x = v == "zeta" ? ds[name][:, :, 1, kr] : ds[name][:, :, st.lev, kr]   # (lon, lat, depth, time)
            f(t[kr], coalesce.(x, NaN32))
        end
    end
end

const HN_VAR = Dict("zeta" => ("ssh", "ssh", false), "u" => ("uv", "u_velocity", true), "v" => ("uv", "v_velocity", true),
                    "temp" => ("ts", "layer_temperature", true), "salt" => ("ts", "layer_salinity", true),
                    "lthick" => ("lthick", "layer_thickness", true))      # NCOM only; one record per file, no MT
mt2datetime(mt) = round(DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000)), Minute(1))   # MT: days since 1900-12-31
hn_tags(src, suf) = sort([chopsuffix(n, "_$suf.nc") for n in readdir(src.data)
                          if startswith(n, src.prefix) && endswith(n, "_$suf.nc")])
function hn_blocks(f, src, v, st)                     # one record at a time (an NCOM 3-D record is ~0.7 GB)
    suf, name, has_z = HN_VAR[v]
    seen = Set{DateTime}()
    for tag in hn_tags(src, suf)
        NCDataset(joinpath(src.data, "$(tag)_$suf.nc")) do ds
            t = haskey(ds, "MT") ? mt2datetime.(ds["MT"].var[:]) : [DateTime(tag[end-9:end], "yyyymmddHH")]
            for k in eachindex(t)
                (inwin(t[k]) && !(t[k] in seen)) || continue
                x = has_z ? ds[name].var[:, :, st.lev, k:k] : ds[name].var[:, :, k:k]
                push!(seen, t[k])
                f(t[k:k], x)
            end
        end
    end
end

blocks(f, src, v, st) = src.kind == :roms ? roms_blocks(f, src, v, st) :
                        src.kind == :cmems ? cmems_blocks(f, src, v, st) : hn_blocks(f, src, v, st)

## --- grids: lon, lat (rho / parent points, -180..180), field sizes, and what ROMS needs to finish u, v
lon180(x) = ifelse(x > 180, x - 360, x)
function load_source(src)
    if src.kind == :roms
        lon, lat, h, mask_rho, angle = NCDataset(src.grid) do g
            g["lon_rho"][:, :], g["lat_rho"][:, :], g["h"][:, :], g["mask_rho"][:, :], g["angle"][:, :]
        end
        td = time_dir(src.data; prefix = "roms_avg.", dt = src.dt, t_ref, tstart = T0 - Day(1), tend = T1 + Day(1))   # exact cut by tmid below
        N, gatt = NCDataset(joinpath(src.data, td.files[1])) do ds
            ds.dim["s_rho"], Dict(k => v for (k, v) in ds.attrib)       # theta_s, theta_b, ... for zlevs3
        end
        lev = FULL3D ? (1:N) : (N:N)                                    # s-levels read; surface = last
        byfile = records_by_file(td)
        tmid = Vector{DateTime}(undef, td.nt)                        # centre time of each average
        for (fi, recs) in byfile
            t = NCDataset(ds -> ds["ocean_time"].var[:], joinpath(src.data, td.files[fi]))
            for (k, n) in recs; tmid[n] = t_ref + Millisecond(round(Int, 1000 * t[k])); end
        end
        mask_u, mask_v = uvp_masks(mask_rho)
        nl = length(lev)
        sz = Dict("zeta" => size(mask_rho), "temp" => (size(mask_rho)..., nl), "salt" => (size(mask_rho)..., nl),
                  "u" => (size(mask_u)..., nl), "v" => (size(mask_v)..., nl))
        return (; lon = lon180.(lon), lat, h, angle, mask_rho, mask_u, mask_v, sz, td, N, lev, isurf = nl,
                gatt, byfile, tmid)
    elseif src.kind == :cmems
        f = only(filter(n -> startswith(n, "cmems_uv_"), readdir(src.data)))
        lo, la = NCDataset(ds -> (Float64.(ds["longitude"][:]), Float64.(ds["latitude"][:])), joinpath(src.data, f))
        lon = repeat(lon180.(lo), 1, length(la)); lat = repeat(la', length(lo), 1)
        depth = NCDataset(ds -> Float64.(ds["depth"][:]), joinpath(src.data, f))
    else
        gsuf = src.kind == :hycom ? "grd" : "lthick"                 # NCOM lon/lat live in the layer-thickness files
        lon, lat = NCDataset(ds -> (ds["Longitude"].var[:, :], ds["Latitude"].var[:, :]),
                             joinpath(src.data, "$(first(hn_tags(src, gsuf)))_$gsuf.nc"))
        lon = lon180.(lon)
        depth = src.kind == :hycom ?                                    # NCOM: layers, see lthick
            NCDataset(ds -> ds["depth"].var[:], joinpath(src.data, "$(first(hn_tags(src, "grd")))_grd.nc")) : nothing
    end
    nz  = NCDataset(ds -> ds.dim[src.kind == :cmems ? "depth" : "z"],
                    joinpath(src.data, src.kind == :cmems ? only(filter(n -> startswith(n, "cmems_uv_"), readdir(src.data))) :
                                                            "$(first(hn_tags(src, "uv")))_uv.nc"))
    lev = FULL3D ? (1:nz) : (1:1)                                        # surface = first
    sz  = Dict(v => v == "zeta" ? size(lon) : (size(lon)..., length(lev)) for v in [VARS; "lthick"])
    return (; lon, lat, sz, lev, isurf = 1, depth)
end

# mean grid spacing [km] (along the first index)
dx_km(lon, lat) = mean(filter(isfinite, hypot.(diff(lon; dims = 1) .* cosd.(lat[1:end-1, :]), diff(lat; dims = 1)))) * 111.195

## --- map settings (as mean_flow_avg.jl)
const CLIM = (temp = (26.0, 30.0), salt = (32.5, 36.5), speed = (0.0, 1.5))
const UV_KM = 108.0; const UV_SCALE = 0.75; const REF_W = 14.6
const MOOR_GROUPS = [("french", :star5, 7), ("M", :diamond, 5), ("CPIES", :circle, 4)]
moor_group(n) = n == "french" ? 1 : startswith(n, "CPIES") ? 3 : 2
const NCOM_LON = [-57.39, -37.68, -27.47, -46.99, -57.39]
const NCOM_LAT = [  1.32,  -8.63,  11.42,  21.37,   1.32]
function crange(x; q = (0.01, 0.99), step = 0.1)
    lo, hi = quantile(filter(!isnan, vec(x)), q)
    return (floor(lo / step) * step, ceil(hi / step) * step)
end

# ROMS 2700 m grid: map extent, 200 m isobath and outline on the parent maps
const G27 = NCDataset(GRID_2700) do g
    lo = lon180.(g["lon_rho"][:, :]); la = g["lat_rho"][:, :]
    (lon = lo, lat = la, h = g["h"][:, :], edge = grid_boundary(lo, la))
end

function mean_map(F, uv, field, view, M; title, label, colormap, colorrange, uv_color = :black, lowclip = nothing)
    bins, sel = M.bins, M.sel
    nb   = length(sel)
    ncol = nb <= 5 ? 3 : 4
    nrow = cld(nb, ncol)
    W  = view.lonlim[2] - view.lonlim[1]
    pw = (18 * cm_to_pt - 96 - 60 - ncol * 4) / ncol
    ph = pw * (view.latlim[2] - view.latlim[1]) / W
    skip  = max(1, round(Int, UV_KM * W / REF_W / M.dx))
    scale = UV_SCALE * W / REF_W
    fig = Figure(; figkw(18, 14)..., figure_padding = (2, 6, 2, 4))
    Label(fig[0, 1:ncol], "$title, $(M.name) ($(M.window))"; fontsize = 12, font = :bold, padding = (0, 0, 0, 0))
    local hm
    for (q, b) in enumerate(sel)
        r, c = fldmod1(q, ncol)
        days = M.nrec[b] * M.dt / 86400                             # coverage of bin b [d]
        ax = topdown_axis(fig[r, c]; title = @sprintf("%s (%.0f d)", bins[b].name, days),
                          xlabel = q + ncol > nb ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "",
                          backgroundcolor = view.bg, width = pw, height = ph, titlegap = 2,
                          xlabelpadding = 1, ylabelpadding = 1)
        hm = plot_curvilinear!(ax, M.lon, M.lat, F[:, :, b]; colormap, colorrange, nan_color = :gray80,
                               lowclip = something(lowclip, Makie.Automatic()))
        contour!(ax, M.hlon, M.hlat, M.h; levels = [200], color = :gray40, linewidth = 0.6)
        uv === nothing || quiver_curvilinear!(ax, M.lon, M.lat, uv[1][:, :, b], uv[2][:, :, b]; skip,
                                              xlim = view.lonlim, ylim = view.latlim,
                                              lengthscale = scale, color = uv_color, shaftwidth = 1.2,
                                              tipwidth = 5, tiplength = 5)
        (view.ncom || M.parent) && lines!(ax, M.edge...; color = :black, linewidth = 0.6)
        view.ncom && lines!(ax, NCOM_LON, NCOM_LAT; color = :black, linewidth = 0.9, linestyle = :dash)
        for (g, (_, mk, ms)) in enumerate(MOOR_GROUPS)
            s = moor_group.(M.moor[1]) .== g
            scatter!(ax, M.moor[2][s], M.moor[3][s]; marker = mk, markersize = ms, color = :red,
                     strokecolor = :black, strokewidth = 0.5)
        end
        limits!(ax, view.lonlim..., view.latlim...)
        q + ncol <= nb && hidexdecorations!(ax; grid = false, ticks = false)   # a panel below: no x labels
        c > 1 && hideydecorations!(ax; grid = false, ticks = false)
    end
    Colorbar(fig[1:nrow, ncol + 1], hm; label, vertical = true, height = Relative(nrow == 1 ? 0.9 : 0.6), width = 10)
    edge = M.parent ? "solid: ROMS 2700 m grid edge" : "solid: model grid edge"
    note = @sprintf("%s, %s .. %s\ngrey line: 200 m isobath%s%s\narrows: mean surface velocity; red: moorings (★ french, ◆ M1–M4, ● CPIES1–9)",
                    M.what, Date(M.tfirst), Date(M.tlast), M.parent ? " (ROMS 2700 m grid)" : "",
                    view.ncom ? "; $edge; dashed: NCOM domain" : M.parent ? "; $edge" : "")
    Label(fig[nrow + 1, 1:ncol + 1], note; fontsize = 8, halign = :left, justification = :left, tellwidth = false)
    rowgap!(fig.layout, 4); colgap!(fig.layout, 4)
    resize_to_layout!(fig)
    fname = "meanflow_avg_surf_$field.png"
    out = joinpath(mkpath(joinpath(FIG_ROOT, M.name, "mean_flow", M.window, view.sub)), fname)
    figsave(out, fig)
    CHECK_DIR === nothing || cp(out, joinpath(mkpath(CHECK_DIR), "$(M.name)_$(M.window)_$(view.name)_$fname"); force = true)
    println("saved ", out)
    empty!(fig); GC.gc()
end

## --- 3-D output file
function open3d(name, src, st, bins, keep)
    mkpath(OUT_DIR)
    ds = NCDataset(joinpath(OUT_DIR, "$(name)_meanflow_avg_3d.nc"), "c")
    nx, ny = size(st.lon); nl = length(st.lev)
    if src.kind == :roms
        defDim(ds, "xi_rho", nx); defDim(ds, "eta_rho", ny); defDim(ds, "xi_u", nx - 1); defDim(ds, "eta_v", ny - 1)
        defDim(ds, "s_rho", nl)
        for (k, v) in st.gatt; ds.attrib["roms_" * k] = v; end        # theta_s, theta_b, grid_file, ...
        ds.attrib["vertical"] = "ROMS s-levels 1 (bottom) .. $nl (top); z from zlevs3(h, zeta, theta_s, theta_b, hc, N, ...)"
        defVar(ds, "lon_rho", st.lon, ("xi_rho", "eta_rho")); defVar(ds, "lat_rho", st.lat, ("xi_rho", "eta_rho"))
        defVar(ds, "h", st.h, ("xi_rho", "eta_rho")); defVar(ds, "mask_rho", st.mask_rho, ("xi_rho", "eta_rho"))
        defVar(ds, "angle", st.angle, ("xi_rho", "eta_rho"))
    else
        defDim(ds, "x", nx); defDim(ds, "y", ny); defDim(ds, "z", nl)
        defVar(ds, "lon", st.lon, ("x", "y")); defVar(ds, "lat", st.lat, ("x", "y"))
        if st.depth === nothing
            ds.attrib["vertical"] = "NCOM layers 1 (top) .. $nl; layer thickness: lthick (mean per bin)"
        else
            ds.attrib["vertical"] = "fixed depths, level 1 = top"
            defVar(ds, "depth", Float64.(st.depth[st.lev]), ("z",); attrib = ["units" => "m", "positive" => "down"])
        end
    end
    ds.attrib["source"] = "$name: $(src.what), $(src.data)"
    ds.attrib["method"] = "mean of all records in [bin_start, bin_end) at every point; NaN = land / no data"
    defDim(ds, "bin", length(keep))
    defVar(ds, "bin_name", getfield.(bins[keep], :name), ("bin",))
    defVar(ds, "bin_kind", string.(getfield.(bins[keep], :kind)), ("bin",))
    defVar(ds, "bin_start", getfield.(bins[keep], :t0), ("bin",))
    defVar(ds, "bin_end", getfield.(bins[keep], :t1), ("bin",))
    return ds
end

const UNITS = Dict("zeta" => "m", "u" => "m s-1", "v" => "m s-1", "temp" => "degC", "salt" => "psu", "lthick" => "m")
function write3d!(ds, v, m, ib)
    if !haskey(ds, v)
        roms = haskey(ds.dim, "xi_rho")
        h2 = roms ? (v == "u" ? ("xi_u", "eta_rho") : v == "v" ? ("xi_rho", "eta_v") : ("xi_rho", "eta_rho")) : ("x", "y")
        dims = ndims(m) == 2 ? (h2..., "bin") : (h2..., roms ? "s_rho" : "z", "bin")
        defVar(ds, v, Float32, dims; fillvalue = NaN32, deflatelevel = 1, shuffle = true,
               chunksizes = (size(m, 1), size(m, 2), ntuple(_ -> 1, length(dims) - 2)...),
               attrib = ["units" => UNITS[v],
                         "note" => roms && v in ("u", "v") ? "ROMS grid-relative (xi / eta) component on its own points" :
                                   v in ("u", "v") ? "east / north component" : ""])
    end
    ds[v].var[ntuple(_ -> Colon(), ndims(m))..., ib] = m
end

## --- loop over sources
for name in SRC_RUN
    src = Dict(SOURCES)[name]
    if !isdir(src.data)
        println("\n===== ", name, ": skipped, no data folder on $HOST (", src.data, ")"); continue
    end
    println("\n===== ", name, ": ", src.data)
    t0 = time()
    st = load_source(src)
    bins = all_bins(); nb = length(bins)

    # Section 2: half-month sums, variable by variable; months / total from the halves; 3-D file + surface slices
    vlist = src.kind == :ncom && FULL3D ? [VARS; "lthick"] : VARS
    means = Dict{String, Array{Float32, 3}}()                           # surface (x, y, bin), all bins
    nrec = zeros(Int, nb); tfirst = fill(T0, nb); tlast = fill(T0, nb)
    keep = Int[]
    nc3 = nothing
    for v in vlist
        halves = Dict{NTuple{3, Int}, Half}()
        blocks(src, v, st) do times, x
            for k in eachindex(times)
                t = times[k]; inwin(t) || continue
                h = get!(() -> Half(st.sz[v], t), halves, halfkey(t))
                addrec!(h, selectdim(x, ndims(x), k), t)
            end
        end
        isempty(halves) && error("$name: no $v records in [$T0, $T1)")
        if v == first(vlist)                                            # bins with data (from zeta)
            keep = [b for b in eachindex(bins) if !isempty(members(halves, bins[b]))]
            FULL3D && (nc3 = open3d(name, src, st, bins, keep))
        end
        surf = fill(NaN32, size(st.lon)..., nb)
        for (ib, b) in enumerate(keep)
            hs = members(halves, bins[b]); isempty(hs) && continue
            m = combine(hs)
            src.kind == :roms && masknan!(m, v == "u" ? st.mask_u : v == "v" ? st.mask_v : st.mask_rho)
            nc3 === nothing || write3d!(nc3, v, m, ib)
            ms = ndims(m) == 2 ? m : view(m, :, :, st.isurf)
            if src.kind == :roms && v in ("u", "v")                     # to rho points for the maps
                ms = v == "u" ? u2rho(ms) : v2rho(ms)
            end
            v in VARS && (surf[:, :, b] = ms)
            if v == "u"
                nrec[b] = sum(h.nrec for h in hs)
                tfirst[b] = minimum(h.tfirst for h in hs); tlast[b] = maximum(h.tlast for h in hs)
            end
        end
        v in VARS && (means[v] = surf)
        @printf("  %-6s %6d records, %.1f min\n", v, sum(h.nrec for h in values(halves)), (time() - t0) / 60)
        halves = nothing; GC.gc()
    end
    if nc3 !== nothing
        close(nc3); println("  saved ", joinpath(OUT_DIR, "$(name)_meanflow_avg_3d.nc"))
    end
    # u, v -> rho points, east / north
    if src.kind == :roms                                                # already on rho points
        u_r = means["u"]; v_r = means["v"]
        means["u"] = u_r .* cos.(st.angle) .- v_r .* sin.(st.angle)
        means["v"] = u_r .* sin.(st.angle) .+ v_r .* cos.(st.angle)
    end
    spd = sqrt.(means["u"] .^ 2 .+ means["v"] .^ 2)

    for b in keep
        @printf("  %-9s %-11s %5d recs  %s .. %s   ssh %7.3f  u %7.3f  v %7.3f  T %6.2f  S %6.2f (domain mean)\n",
                bins[b].name, bins[b].kind, nrec[b], tfirst[b], tlast[b],
                (mean(filter(!isnan, view(means[v], :, :, b))) for v in VARS)...)
    end

    # save the means (non-empty bins)
    mkpath(OUT_DIR)
    ncout = joinpath(OUT_DIR, "$(name)_meanflow_avg.nc")
    NCDataset(ncout, "c") do ds
        nx, ny = size(st.lon)
        defDim(ds, "x", nx); defDim(ds, "y", ny); defDim(ds, "bin", length(keep))
        ds.attrib["source"] = "$name: $(src.what), $(src.data)"
        ds.attrib["method"] = "surface means of all records in [bin_start, bin_end); u_east / v_north on rho / parent points"
        defVar(ds, "lon", st.lon, ("x", "y")); defVar(ds, "lat", st.lat, ("x", "y"))
        defVar(ds, "bin_name", getfield.(bins[keep], :name), ("bin",))
        defVar(ds, "bin_kind", string.(getfield.(bins[keep], :kind)), ("bin",))
        defVar(ds, "bin_start", getfield.(bins[keep], :t0), ("bin",))
        defVar(ds, "bin_end", getfield.(bins[keep], :t1), ("bin",))
        defVar(ds, "nrec", nrec[keep], ("bin",))
        for (v, nm, u) in (("zeta", "zeta", "m"), ("u", "u_east", "m s-1"), ("v", "v_north", "m s-1"),
                           ("temp", "temp", "degC"), ("salt", "salt", "psu"))
            defVar(ds, nm, means[v][:, :, keep], ("x", "y", "bin"); attrib = ["units" => u, "_FillValue" => NaN32])
        end
    end
    println("  saved ", ncout)

    # Section 3: maps for each window
    moor = read_moorings(src.moor)
    parent = src.kind != :roms
    hsrc = parent ? G27 : (lon = st.lon, lat = st.lat, h = st.h)
    edge = parent ? G27.edge : grid_boundary(st.lon, st.lat)
    views = [
        (name = "domain", lonlim = extrema(parent ? G27.lon : st.lon), latlim = extrema(parent ? G27.lat : st.lat),
         sub = "surface", bg = parent ? :white : :gray80, ncom = false),
        (name = "ncom", lonlim = (-57.39, -27.47), latlim = (-8.63, 21.37), sub = "surface_ncom", bg = :white, ncom = true),
    ]
    zrange = crange(means["zeta"][:, :, keep]; step = 0.05)
    for window in WINDOWS
        sel = [b for b in keep if bins[b].kind == Symbol(window) || bins[b].kind == :total]
        M = (; name, window, bins, sel, nrec, dt = src.dt, what = src.what, lon = st.lon, lat = st.lat,
             hlon = hsrc.lon, hlat = hsrc.lat, h = hsrc.h, edge, moor, parent,
             dx = dx_km(st.lon, st.lat), tfirst = tfirst[end], tlast = tlast[end])
        for view in views
            mean_map(means["zeta"], (means["u"], means["v"]), "ssh", view, M; title = "Mean sea-surface height",
                     label = "ssh [m]", colormap = Reverse(:RdBu), colorrange = zrange)
            mean_map(means["temp"], (means["u"], means["v"]), "temp", view, M; title = "Mean surface temperature",
                     label = "T [°C]", colormap = :thermal, colorrange = CLIM.temp)
            mean_map(means["salt"], (means["u"], means["v"]), "salt", view, M; title = "Mean surface salinity",
                     label = "S [psu]", colormap = :haline, colorrange = CLIM.salt, lowclip = :black)
            mean_map(spd, (means["u"], means["v"]), "speed", view, M; title = "Mean surface speed |ū| and velocity",
                     label = "speed [m s⁻¹]", colormap = :viridis, colorrange = CLIM.speed, uv_color = :white)
        end
    end
    means = nothing; spd = nothing; st = nothing; GC.gc()
    @printf("===== %s done, %.1f min\n", name, (time() - t0) / 60)
end
