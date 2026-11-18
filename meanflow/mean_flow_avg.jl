#= mean_flow_avg.jl
Hsin-Yi Chen, USM DMS
Mean (sub-tidal) surface flow from ROMS average output, monthly + whole-record means, for several runs.
The run is picked with MF_RUN (keys of RUNS below):
  amazon_2700m_cmems_0801   2700 m, CMEMS parent, daily averages
  amazon_900m_3mon_1008     900 m (data in amazon_900m_3mon_1008_2), hourly averages
  design_1113_hycom_2700    2700 m new design (grid 1104), HYCOM bry, daily averages
  design_1116_cmems_2700    2700 m new design (grid 1104), CMEMS ini/bry, daily averages

Section 1: timeline of roms_avg.*.nc (4 records per file; daily (86400 s) or hourly (3600 s) averaging window)
  surface fields: u, v [m/s] (top s-level, own u / v grids), ssh = zeta [m], temp [degC], salt [psu]
  tmid = centre time of each average (from ocean_time).
  READ_ALL = true also reads the full (xi, eta, time) arrays (not needed by Section 2).

Section 2: running means without holding the time series in memory.
  Bins: one per calendar month in the record + "total" (whole record).
  Each file is read once (all its records of one variable in one block), every record is added to the sum
  of every bin it falls in (by tmid), then dropped; cnt[b] counts the records of bin b; mean = sum / cnt,
  land -> NaN. Memory = one file block + the sums (nbin x field size, Float64), independent of record length.
  Output: means["u"] etc. as (xi, eta, bin) or (xi, eta, s, bin); bin names in periods[b].name.

Section 3: maps of the surface means, one figure per field, 2 x 3 panels (Aug, Sep, Oct / Nov, total, colour bar):
  ssh, temp, salt, speed |mean u|; mean surface velocity arrows on all (u, v moved to rho points, rotated to
  east / north by angle; black, white on the speed map); moorings as red markers; 200 m isobath grey.
  Two views, each its own folder:
    surface/       axes = this run's grid extent
    surface_ncom/  axes fixed to the NCOM parent box (same for every run, for comparing runs);
                   dashed = rotated NCOM domain, solid = this run's grid edge
  Arrows are spaced and scaled per view so they look the same on paper (UV_KM / UV_SCALE per 14.6 deg of map).
  Colour ranges fixed for temp / salt / speed (same in every run and view); ssh from this run's 1-99 %
  (reference level differs between parents).
  Copies go to CHECK_DIR as <run>_<view>_meanflow_avg_surf_<field>.png.

Usage:
  MF_RUN=design_1116_cmems_2700 julia --project=. --heap-size-hint=3G meanflow/mean_flow_avg.jl
  sbatch sbatch_files/julia_meanflow_avg.sh          # all four runs as a job array
=#

using NCDatasets, Printf, Statistics, Dates
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")   # time_dir, records_by_file, data_load, ...
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # CairoMakie, topdown_axis, plot_curvilinear!, quiver_curvilinear!
include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings

## --- runs
const BASE = "/expanse/lustre/projects/uso102/hchen54/"
const RUNS = Dict(
    "amazon_2700m_cmems_0801" => (data = BASE * "amazon_2700m_cmems_0801/after_joint/",
                                  grid = BASE * "input/grid/roms_cmems_grd_2700m.nc",
                                  moor = BASE * "input/grid/roms_cmems_grd_2700m_mor_edata.nc", dt = 86400),
    "amazon_900m_3mon_1008"   => (data = BASE * "amazon_900m_3mon_1008_2/after_joint/",
                                  grid = BASE * "input/grid/roms_grd_900m.nc",
                                  moor = BASE * "input/grid/roms_grd_900m_mor_edata.nc", dt = 3600),
    "design_1113_hycom_2700"  => (data = BASE * "design_1113_hycom_2700/after_joint/",
                                  grid = BASE * "new_design_fix/roms_grd_2700m_1104.nc",
                                  moor = BASE * "new_design_fix/roms_grd_2700m_1104_mor_edata.nc", dt = 86400),
    "design_1116_cmems_2700"  => (data = BASE * "design_1116_cmems_2700/after_joint/",
                                  grid = BASE * "new_design_fix/roms_grd_2700m_1104.nc",
                                  moor = BASE * "new_design_fix/roms_grd_2700m_1104_mor_edata.nc", dt = 86400),
)

## --- settings
const RUN         = get(ENV, "MF_RUN", "amazon_2700m_cmems_0801")
haskey(RUNS, RUN) || error("MF_RUN = $RUN, not one of $(sort(collect(keys(RUNS))))")
const cfg         = RUNS[RUN]
const data_folder = cfg.data
const grid_file   = cfg.grid
const moor_file   = cfg.moor                      # mooring locations (attributes)
const DT          = cfg.dt                        # averaging window = record spacing [s]
const t_ref       = DateTime(1994, 1, 1)          # ocean_time reference ("Time since 1994/01/01")
const SPINUP      = 0.0                           # days dropped at the start of the run
const TSTART      = nothing                       # e.g. DateTime(2022, 9, 1); nothing = from the start
const TEND        = nothing                       # e.g. DateTime(2022, 11, 30); nothing = to the end
const READ_ALL    = false                         # Section 1: also read full (xi, eta, time) arrays
const figure_path = BASE * "figure/$RUN/mean_flow"
const CHECK_DIR   = expanduser("~/figure_check")  # copy of every figure for checking; nothing = no copy
println("run ", RUN, ": ", data_folder)

## --- grid (Julia order: xi, eta)
lon, lat, h, mask_rho, pm, pn, angle = NCDataset(grid_file) do g
    g["lon_rho"][:, :], g["lat_rho"][:, :], g["h"][:, :], g["mask_rho"][:, :],
    g["pm"][:, :], g["pn"][:, :], g["angle"][:, :]
end
lon = @. ifelse(lon > 180, lon - 360, lon)        # degrees east -> -180..180 (west negative)
mask_u, mask_v = uvp_masks(mask_rho)
const DX_KM = mean(1 ./ pm) / 1000                # mean grid spacing [km]

## --- Section 1: timeline of the averages
td = time_dir(data_folder; prefix = "roms_avg.", dt = DT, t_ref, spinup = SPINUP, tstart = TSTART, tend = TEND)
N  = NCDataset(ds -> ds.dim["s_rho"], joinpath(data_folder, td.files[1]))   # top s-level = surface

# centre time of each average (time_dir rounds ocean_time to whole DT, which lands on the next step)
tmid = Vector{DateTime}(undef, td.nt)
for (fi, recs) in records_by_file(td)
    t = NCDataset(ds -> ds["ocean_time"].var[:], joinpath(data_folder, td.files[fi]))
    for (k, n) in recs; tmid[n] = t_ref + Millisecond(round(Int, 1000 * t[k])); end
end

if READ_ALL
    ssh  = data_load(data_folder, "zeta", td; grid_file)              # (xi_rho, eta_rho, t) [m]
    u_s  = data_load(data_folder, "u",    td; iz = N, grid_file)      # (xi_u,   eta_rho, t) [m/s]
    v_s  = data_load(data_folder, "v",    td; iz = N, grid_file)      # (xi_rho, eta_v,   t) [m/s]
    temp = data_load(data_folder, "temp", td; iz = N, grid_file)      # (xi_rho, eta_rho, t) [degC]
    salt = data_load(data_folder, "salt", td; iz = N, grid_file)      # (xi_rho, eta_rho, t) [psu]
end

## --- Section 2: monthly + whole-record means, accumulated file by file
# variable => level: nothing for 2-D fields, N for the surface, 1:N (or any range) for full 3-D
const VARS = ["zeta" => nothing, "u" => N, "v" => N, "temp" => N, "salt" => N]

# averaging bins [t0, t1): each calendar month present in tmid, then the whole record
periods = [(name = Dates.format(DateTime(y, m), "yyyy-mm"), t0 = DateTime(y, m), t1 = DateTime(y, m) + Month(1))
           for (y, m) in unique(yearmonth.(tmid))]
push!(periods, (name = "total", t0 = typemin(DateTime), t1 = typemax(DateTime)))
nb     = length(periods)
binsof = [findall(p -> p.t0 <= t < p.t1, periods) for t in tmid]   # bins each time step belongs to

# sums (Float64, so 1000+ adds of Float32 lose nothing) and record counts
acc = NCDataset(joinpath(data_folder, td.files[1])) do ds
    Dict(v => zeros(Float64, size(lev === nothing ? ds[v].var[:, :, 1] : ds[v].var[:, :, lev, 1])..., nb)
         for (v, lev) in VARS)
end
cnt = zeros(Int, nb)

t0 = time()
byfile = records_by_file(td)
for (q, (fi, recs)) in enumerate(byfile)
    kr = minimum(first.(recs)):maximum(first.(recs))               # records of this file, read as one block
    NCDataset(joinpath(data_folder, td.files[fi])) do ds
        for (v, lev) in VARS
            x = lev === nothing ? ds[v].var[:, :, kr] : ds[v].var[:, :, lev, kr]   # raw (land = 0), time last
            a = acc[v]; c = ntuple(_ -> Colon(), ndims(a) - 1)
            for (k, n) in recs, b in binsof[n]
                view(a, c..., b) .+= selectdim(x, ndims(x), k - first(kr) + 1)
            end
        end
    end
    for (_, n) in recs; cnt[binsof[n]] .+= 1; end
    q % 50 == 0 && @printf("  %d / %d files, %.1f min\n", q, length(byfile), (time() - t0) / 60)
end

# mean = sum / count; land -> NaN on each variable's own grid
means = Dict{String, Array{Float32}}()
for (v, _) in VARS
    a  = acc[v]; nd = ndims(a)
    m  = Float32.(a ./ reshape(cnt, ntuple(_ -> 1, nd - 1)..., nb))
    mk = size(a)[1:2] == size(mask_u) ? mask_u : size(a)[1:2] == size(mask_v) ? mask_v : mask_rho
    m[findall(mk .== 0), ntuple(_ -> Colon(), nd - 2)...] .= NaN32
    means[v] = m
end
acc = nothing                                                       # free the sums

const AVG_WORD = DT == 86400 ? "daily" : DT == 3600 ? "hourly" : "$(DT) s"
cnt_days(b) = cnt[b] * DT / 86400                                   # length of bin b [days]
@printf("Section 2: %d bins, %.1f min\n", nb, (time() - t0) / 60)
for (b, p) in enumerate(periods)
    ts = tmid[findall(n -> b in binsof[n], 1:td.nt)]
    @printf("  %-8s %5d %s means (%5.1f d)  %s .. %s   ssh %7.3f  u %7.3f  v %7.3f  T %6.2f  S %6.2f (domain mean)\n",
            p.name, cnt[b], AVG_WORD, cnt_days(b), Date(ts[1]), Date(ts[end]),
            (mean(filter(!isnan, selectdim(means[v], ndims(means[v]), b))) for v in first.(VARS))...)
end

## --- Section 3: monthly / total surface maps, 2 x 3 panels per field, two views
# velocity on rho points, rotated from grid (xi, eta) to east / north
u_r = u2rho(means["u"]); v_r = v2rho(means["v"])
ue  = u_r .* cos.(angle) .- v_r .* sin.(angle)
vn  = u_r .* sin.(angle) .+ v_r .* cos.(angle)
spd = sqrt.(ue .^ 2 .+ vn .^ 2)                                     # |mean vector| [m/s]

# colour range over all bins from quantiles, rounded to step
function crange(x; q = (0.01, 0.99), step = 0.1)
    lo, hi = quantile(filter(!isnan, vec(x)), q)
    return (floor(lo / step) * step, ceil(hi / step) * step)
end

# fixed colour ranges, same for every run and view (ssh: per run)
const CLIM = (temp = (26.0, 30.0), salt = (32.5, 36.5), speed = (0.0, 1.5))

const PANEL = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]             # bins in periods order; colour bar in (2, 3)
const UV_KM    = 108.0                                              # arrow spacing [km] on a 14.6 deg wide map
const UV_SCALE = 0.75                                               # arrow length [deg] per 1 m/s on a 14.6 deg wide map
const REF_W    = 14.6                                               # map width [deg] the two above are tuned for

# moorings: french (star), M1-M4 (diamond), CPIES1-9 (circle)
moor_name, moor_lon, moor_lat = read_moorings(moor_file)
const MOOR_GROUPS = [("french", :star5, 7), ("M", :diamond, 5), ("CPIES", :circle, 4)]
moor_group(n) = n == "french" ? 1 : startswith(n, "CPIES") ? 3 : 2

# NCOM parent domain: corners (j, i) = (1,1), (1,end), (end,end), (end,1), closed
const NCOM_LON = [-57.39, -37.68, -27.47, -46.99, -57.39]
const NCOM_LAT = [  1.32,  -8.63,  11.42,  21.37,   1.32]
grid_bx, grid_by = grid_boundary(lon, lat)

# views: map limits, sub-folder, background outside the grid, outlines
const VIEWS = [
    (name = "domain", lonlim = extrema(lon), latlim = extrema(lat), sub = "surface", bg = :gray80, ncom = false),
    (name = "ncom", lonlim = (-57.39, -27.47), latlim = (-8.63, 21.37), sub = "surface_ncom", bg = :white, ncom = true),
]

const PANEL_W = (18 * cm_to_pt - 96 - 2 * 4) / 3                    # pt; 96 pt for y ticks + label + padding, 4 pt gaps

function mean_map(F, field, view; title, label, colormap, colorrange, uv = (ue, vn), uv_color = :black, lowclip = nothing)
    W  = view.lonlim[2] - view.lonlim[1]
    ph = PANEL_W * (view.latlim[2] - view.latlim[1]) / W            # DataAspect
    skip  = max(1, round(Int, UV_KM * W / REF_W / DX_KM))           # same arrow density on paper in every view / run
    scale = UV_SCALE * W / REF_W
    fig = Figure(; figkw(18, 14)..., figure_padding = (2, 6, 2, 4))
    Label(fig[0, 1:3], "$title, $RUN"; fontsize = 12, font = :bold, padding = (0, 0, 0, 0))
    local hm
    for (b, p) in enumerate(periods)
        r, c = PANEL[b]
        ax = topdown_axis(fig[r, c]; title = @sprintf("%s (%.0f d)", p.name, cnt_days(b)),
                          xlabel = r == 2 ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "",
                          backgroundcolor = view.bg, width = PANEL_W, height = ph, titlegap = 2,
                          xlabelpadding = 1, ylabelpadding = 1)
        hm = plot_curvilinear!(ax, lon, lat, F[:, :, b]; colormap, colorrange, nan_color = :gray80,
                               lowclip = something(lowclip, Makie.Automatic()))
        contour!(ax, lon, lat, h; levels = [200], color = :gray40, linewidth = 0.6)
        uv === nothing || quiver_curvilinear!(ax, lon, lat, uv[1][:, :, b], uv[2][:, :, b]; skip,
                                              lengthscale = scale, color = uv_color, shaftwidth = 1.2,
                                              tipwidth = 5, tiplength = 5)
        if view.ncom
            lines!(ax, grid_bx, grid_by; color = :black, linewidth = 0.6)                      # this run's grid
            lines!(ax, NCOM_LON, NCOM_LAT; color = :black, linewidth = 0.9, linestyle = :dash)  # NCOM domain
        end
        for (g, (_, mk, ms)) in enumerate(MOOR_GROUPS)
            sel = moor_group.(moor_name) .== g
            scatter!(ax, moor_lon[sel], moor_lat[sel]; marker = mk, markersize = ms, color = :red,
                     strokecolor = :black, strokewidth = 0.5)
        end
        limits!(ax, view.lonlim..., view.latlim...)                  # arrows past the edge are clipped
        r == 1 && hidexdecorations!(ax; grid = false, ticks = false)
        c > 1 && hideydecorations!(ax; grid = false, ticks = false)
    end
    Colorbar(fig[2, 3], hm; label, vertical = true, height = Relative(0.85), width = 10,
             tellwidth = false, tellheight = false, halign = :left)
    note = @sprintf("%s means %s .. %s; grey line: 200 m isobath%s\narrows: mean surface velocity; red: moorings (★ french, ◆ M1–M4, ● CPIES1–9)",
                    AVG_WORD, Date(tmid[1]), Date(tmid[end]),
                    view.ncom ? "; solid: model grid edge; dashed: NCOM domain" : "")
    Label(fig[3, 1:3], note; fontsize = 8, halign = :left, justification = :left, tellwidth = false)
    rowgap!(fig.layout, 4); colgap!(fig.layout, 4)
    resize_to_layout!(fig)                                          # figure height = what the panels need
    fname = "meanflow_avg_surf_$field.png"
    out = figpath(view.sub, fname)
    figsave(out, fig)
    CHECK_DIR === nothing || cp(out, joinpath(mkpath(CHECK_DIR), "$(RUN)_$(view.name)_$fname"); force = true)
    println("saved ", out)
    empty!(fig); GC.gc()                                            # free the figure (login node: 8 GB limit)
end

for view in VIEWS
    mean_map(means["zeta"], "ssh", view; title = "Mean sea-surface height",
             label = "ssh [m]", colormap = Reverse(:RdBu), colorrange = crange(means["zeta"]; step = 0.05))
    mean_map(means["temp"], "temp", view; title = "Mean surface temperature",
             label = "T [°C]", colormap = :thermal, colorrange = CLIM.temp)
    mean_map(means["salt"], "salt", view; title = "Mean surface salinity",
             label = "S [psu]", colormap = :haline, colorrange = CLIM.salt,
             lowclip = :black)                                      # plume < range in black, so the open ocean keeps contrast
    mean_map(spd, "speed", view; title = "Mean surface speed |ū| and velocity",
             label = "speed [m s⁻¹]", colormap = :viridis, colorrange = CLIM.speed, uv_color = :white)
end
