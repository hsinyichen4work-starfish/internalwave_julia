using Distributed

# worker processes for the hourly frames (section 6), the only slow part (~19 s per frame, 2353 frames); everything
# else runs on this main process. Added only once, so re-running this block in the same session is safe. Each worker
# is told how much memory it may use before it must clean up: its share of the job's memory after leaving ~6 GB for
# this main process (which holds the Hovmöller lines and the amplitude sums).
n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "3")) - 1)   # -1 reserves a CPU for this main process
job_mem_mb = parse(Int, get(ENV, "SLURM_MEM_PER_NODE", "16000"))
heap_mb = floor(Int, (job_mem_mb - 6000) * 0.85 / n_workers)
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: several processes read the same files; workers inherit this
nprocs() == 1 && addprocs(n_workers; exeflags = ["--project=$(Base.active_project())", "--heap-size-hint=$(heap_mb)M"])
println("running with $(nprocs() - 1) worker processes")

# ============================================================================
# (1) read the grid and the DDF output (ssh_north/south, ssh_north/south_M2, ssh_north/south_bc)
# ============================================================================
# Everything in the @everywhere block is set up on every process (main and workers): packages, file names, grid,
# moorings and the reading helpers — the workers need them to draw frames, and they share no memory with this process.
# The DDF file is kept open and the ssh fields are read only when asked for: each is [xi, eta, time] ≈ 5.5 GB
# for the full run (≈ 33 GB for all six), too much to load at once.
@everywhere begin
    using NCDatasets, Dates, Statistics

    grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
    ddf_fname = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/ddf_ssh/ssh_ddf_north_south.nc"   # made by DDF_on_ssh.jl
    ds_ddf = NCDataset(ddf_fname)
    ssh_names = ["ssh_north", "ssh_south", "ssh_north_M2", "ssh_south_M2", "ssh_north_bc", "ssh_south_bc"]

    # DDF_on_ssh.jl dropped the last xi and eta row so nx, ny are odd — trim the grid fields the same way
    nx_f, ny_f = ds_ddf.dim["xi_rho"], ds_ddf.dim["eta_rho"]
    nt = ds_ddf.dim["time"]
    lon_rho, lat_rho, mask_rho = ds_ddf["lon_rho"][:, :], ds_ddf["lat_rho"][:, :], ds_ddf["mask_rho"][:, :]
    h, pm, pn, grid_angle = NCDataset(grid_fname) do ds
        ds["h"][1:nx_f, 1:ny_f], ds["pm"][1:nx_f, 1:ny_f], ds["pn"][1:nx_f, 1:ny_f], ds["angle"][1:nx_f, 1:ny_f]
    end
    date = ds_ddf["ocean_time"][:]   # DateTime (decoded from seconds since 1994-01-01)

    # moorings (lon -180..180, lat), same styles as energyflux_tile_working_hpc.jl. The mooring file stores ROMS
    # 0-based fractional grid indices, so Julia index = ROMS index + 1 (checked against the moorings' lon/lat). The
    # main mooring line M1 -> CPIES7 is the grid column xi = 343 — the middle of the trimmed DDF grid — running grid
    # south (shelf) to grid north (offshore) along eta.
    include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings
    mooring_file = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m_mor_edata.nc"   # french, M1..M4, CPIES1..9
    moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
    moor_ij = NCDataset(mooring_file) do ds
        [(ds["$(n)_mor"][1] + 1, ds["$(n)_mor"][2] + 1) for n in moor_name]   # fractional Julia (xi, eta)
    end
    moor_group = [replace(n, r"\d+$" => "") for n in moor_name]   # "M3" -> "M", "CPIES7" -> "CPIES"
    moor_style = Dict("french" => (marker = :star5,   markersize = 11, color = :magenta),
                      "M"      => (marker = :diamond, markersize = 7,  color = :cyan),
                      "CPIES"  => (marker = :circle,  markersize = 6,  color = :yellow))
    ix_tr = round(Int, moor_ij[findfirst(==("M1"), moor_name)][1])   # mooring-line column (343)
    # three grid columns that cut the domain into 4 equal pieces along xi (quarter points 0.5 + k·nx_f/4); the middle
    # one is the mooring line. These are the Hovmöller transects in section 5.
    tr_lines = [("west", round(Int, 0.5 + nx_f / 4)), ("mooring", round(Int, 0.5 + nx_f / 2)), ("east", round(Int, 0.5 + 3nx_f / 4))]

    # marks every mooring (and the three Hovmöller lines, dashed) on a map; Point3f at z = 0 + overdraw keeps them on
    # top of plot_curvilinear!'s flat surface
    function add_mooring!(ax; transect = true)
        if transect
            for (_, ix) in tr_lines
                lines!(ax, Point3f.(lon_rho[ix, :], lat_rho[ix, :], 0); color = :black, linestyle = :dash,
                       linewidth = 0.8, overdraw = true)
            end
        end
        for g in unique(moor_group)
            k = findall(==(g), moor_group)
            st = get(moor_style, g, (marker = :rect, markersize = 6, color = :white))
            scatter!(ax, Point3f.(moor_lon[k], moor_lat[k], 0); marker = st.marker, markersize = st.markersize,
                     color = st.color, strokecolor = :black, strokewidth = 0.75, overdraw = true)
        end
    end

    # reading helpers — files are chunked one snapshot per chunk:
    #   read_map is fast (one chunk); read_series reads every chunk (slow on the full file, minutes);
    #   read_all loads a whole field (≈ 5.5 GB, use on a compute node)
    read_map(name, t) = nomissing(ds_ddf[name][:, :, t], NaN32)  # [xi, eta] at time index t (fill/land -> NaN)
    read_series(name, i, j) = ds_ddf[name][i, j, :]              # [time] at grid point (i, j)
    read_all(name) = ds_ddf[name][:, :, :]                       # [xi, eta, time]
    tindex(d::DateTime) = argmin(abs.(date .- d))                # nearest time index to a date
end   # @everywhere

println("DDF file: ", ddf_fname)
for name in ssh_names
    println("  $name: ", ds_ddf[name].attrib["long_name"])
end

# check the trimmed grid lines up with the DDF file (same points, lon in -180..180 in both)
lon_grid = NCDataset(ds -> ds["lon_rho"][1:nx_f, 1:ny_f], grid_fname)
lon_grid[lon_grid .> 180] .-= 360
maximum(abs, lon_grid .- lon_rho) < 1e-6 || error("grid file and DDF file lon_rho do not match — wrong grid or trimming")

fband_M2 = ds_ddf.attrib["M2_band_cph"]
println("grid: $nx_f x $ny_f, time: $nt records, ", date[1], " -> ", date[end], ", dt = ", Dates.canonicalize(date[2] - date[1]))
println("M2 band: periods $(round(1 / fband_M2[2], digits = 1))–$(round(1 / fband_M2[1], digits = 1)) h")

on_tr = findall(k -> abs(moor_ij[k][1] - ix_tr) < 1, eachindex(moor_name))   # moorings on the mooring line
println("mooring line: xi = $ix_tr, moorings on it: $(join(moor_name[on_tr], ", "))")
tr_lines[2][2] == ix_tr || @warn "middle quarter line xi = $(tr_lines[2][2]) is not the mooring line xi = $ix_tr"
println("Hovmöller lines: ", join(["$n xi = $i" for (n, i) in tr_lines], ", "))

# ============================================================================
# (2) M2 ssh variance against wavelength: where is mode 1, and is the wavenumber cut in the right place?
# ============================================================================
# Saved by DDF_on_ssh.jl from the M2-band spectrum before the cut. Expect the barotropic tide (whatever the plane did
# not remove) at the longest wavelengths, and a mode-1 internal-tide peak near λ1. The cut (dashed red) should sit
# between the two; if the mode-1 peak is not near the assumed λ1 (dotted), change λ1 in DDF_on_ssh.jl and re-run.
@everywhere using CairoMakie
@everywhere figure_path = "/expanse/lustre/projects/uso101/hchen54/figure/amazon_900m_3mon_1008/DFF"
mkpath(figure_path)

k_center = ds_ddf["k_center"][:]   # [rad per meter]
spec_north, spec_south = ds_ddf["spec_M2_north"][:], ds_ddf["spec_M2_south"][:]
λ1, λcut = ds_ddf.attrib["M2_lambda1_m"], ds_ddf.attrib["M2_kcut_wavelength_m"]
wavelength_km = 2π ./ k_center ./ 1e3

let fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xreversed = true,
              xlabel = "wavelength [km]  (shorter →)", ylabel = "M2 ssh variance per |k| bin [m²]",
              title = "M2-band (periods $(round(1 / fband_M2[2], digits = 1))–$(round(1 / fband_M2[1], digits = 1)) h) ssh spectrum, before the wavenumber cut")
    # log axis: skip empty bins; stop at the shortest resolvable wavelength 2dx — bins beyond |k| = π only cover the
    # corners of the wavenumber grid and drop off artificially
    ok = (spec_north .> 0) .& (spec_south .> 0) .& (wavelength_km .>= 2 * mean(1 ./ pm) / 1e3)
    lines!(ax, wavelength_km[ok], spec_north[ok]; label = "grid-north going")
    lines!(ax, wavelength_km[ok], spec_south[ok]; label = "grid-south going")
    vlines!(ax, [λcut / 1e3]; color = :red, linestyle = :dash, label = "cut: 3λ1 = $(λcut / 1e3) km")
    vlines!(ax, [λ1 / 1e3]; color = :gray40, linestyle = :dot, label = "assumed λ1 = $(λ1 / 1e3) km")
    axislegend(ax; position = :rt)
    fname = joinpath(figure_path, "M2_ssh_k_spectrum.png")
    save(fname, fig)
    println("saved $fname")
end

# share of the M2 variance the cut removes (wavelengths ≥ 3λ1), per sector
cut = wavelength_km .>= λcut / 1e3
println("M2 variance removed by the cut: north $(round(100 * sum(spec_north[cut]) / sum(spec_north), digits = 1))%, ",
        "south $(round(100 * sum(spec_south[cut]) / sum(spec_south), digits = 1))%")

# ============================================================================
# (3) one pass over the record: amplitude sums, the three Hovmöller lines, and the raw-ssh spatial mean
# ============================================================================
# Every hourly map of the six DDF fields (and the raw ssh) is read once, and everything sections 4–5 need is collected
# on the way: running sums for the amplitude maps (for each period in `periods`), the ssh along the three Hovmöller
# lines, and the spatial mean of the raw ssh. Reading once matters: the full DDF file is ~33 GB.
# Memory stays small: the sums are a few [xi, eta] maps, the lines are 3 x 7 x [eta, time] (~170 MB for the full record).
@everywhere include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # topdown_axis, plot_curvilinear!
@everywhere ssh_raw_fname = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/ddf_ssh/roms_ssh_hourly.nc"   # made by DDF_on_ssh.jl (1b)

check_day = Date(2022, 9, 4)   # the one-day zoom (amplitude maps, Hovmöller, movie) next to the full period
day_tidx = findfirst(d -> Date(d) == check_day, date):findlast(d -> Date(d) == check_day, date)
periods = [("full", 1:nt), ("day", day_tidx)]   # (name, time indices) — every output below is made for each
plabel(tidx) = "$(Dates.format(date[tidx[1]], "yyyymmddHH"))-$(Dates.format(date[tidx[end]], "yyyymmddHH"))"
pstr(tidx) = "$(Dates.format(date[tidx[1]], "yyyy-mm-dd HH:MM")) – $(Dates.format(date[tidx[end]], "yyyy-mm-dd HH:MM"))"
println("periods: ", join(["$p $(pstr(t)) ($(length(t)) h)" for (p, t) in periods], ", "))

@everywhere begin   # every process: the frames read the raw ssh too
    ds_raw = NCDataset(ssh_raw_fname)
    date_raw = ds_raw["ocean_time"][:]
    raw_t0 = findfirst(==(date[1]), date_raw) - 1   # raw index = DDF index + raw_t0
    read_raw(t) = nomissing(ds_raw["zeta"][1:nx_f, 1:ny_f, raw_t0 + t], NaN32)   # [xi, eta] at DDF time index t, same trim
end
date_raw[raw_t0 .+ (1:nt)] == date || error("raw ssh file does not have the same hours as the DDF file")

function time_pass()
    S1 = Dict((p, name) => zeros(nx_f, ny_f) for (p, _) in periods for name in ssh_names)   # Σ a
    S2 = Dict((p, name) => zeros(nx_f, ny_f) for (p, _) in periods for name in ssh_names)   # Σ a²
    cols = Dict((l, name) => zeros(Float32, ny_f, nt) for (l, _) in tr_lines for name in [ssh_names; "raw_anom"])
    raw_mean = zeros(nt)
    for t in 1:nt
        for name in ssh_names
            a = read_map(name, t)
            for (p, ptidx) in periods
                t in ptidx || continue
                S1[(p, name)] .+= a
                S2[(p, name)] .+= Float64.(a) .^ 2
            end
            for (l, ix) in tr_lines
                cols[(l, name)][:, t] = a[ix, :]
            end
        end
        r = read_raw(t)
        raw_mean[t] = mean(filter(!isnan, r))   # spatial mean over the (trimmed) domain at this hour
        for (l, ix) in tr_lines
            cols[(l, "raw_anom")][:, t] = r[ix, :] .- raw_mean[t]
        end
        t % 200 == 0 && println("  time pass: $t / $nt hours")
    end
    # amplitude = √2 × std over time (variation about the time mean); max(…, 0) guards round-off below zero
    amp_all = Dict((p, name) => sqrt.(2 .* max.(S2[(p, name)] ./ length(ptidx) .- (S1[(p, name)] ./ length(ptidx)) .^ 2, 0))
                   for (p, ptidx) in periods for name in ssh_names)
    amp_all, cols, raw_mean
end
@time amp_all, cols, raw_mean = time_pass()

groups = [("total", "", "all frequencies"), ("M2", "_M2", "M2 band, λ < $(round(Int, λcut / 1e3)) km"),
          ("bc", "_bc", "all freq, λ < $(round(Int, λcut / 1e3)) km")]
edge = 30   # colour ranges come from the interior: skip this many points along the edges (FFT edge-leakage bands)

# ============================================================================
# (4) amplitude maps for each direction, and the south/north energy ratio — total, M2 and bc, for each period
# ============================================================================
# amplitude = √2 × std over time (variation about the time mean) — exact for one sine wave.
#   M2: the 9–15 h band also holds S2, N2, K2, so this is an effective amplitude averaged over the period; the full
#       record covers ~6 spring–neap cycles, one day only one phase of it.
#   total / bc: many frequencies, so not a tidal amplitude — read it as √2 × the ssh std. The time mean is taken out
#       first: a steady ssh pattern is not wave energy (and has no clear travel direction).
# Energy ∝ amplitude² with the same factor for both directions at one place (it depends on depth and stratification,
# not on direction), so (A_south / A_north)² is the south/north energy ratio there.
amp = Dict{Tuple{String, String}, NamedTuple}()   # amp[("full", "M2")].north, amp[("day", "bc")].ratio, ...  [xi, eta]
M2_band_str = "$(round(1 / fband_M2[2], digits = 1))–$(round(1 / fband_M2[1], digits = 1)) h"
amp_desc = Dict("total" => "all frequencies, all wavelengths",
                "M2" => "M2 band ($M2_band_str), λ < $(round(Int, λcut / 1e3)) km",
                "bc" => "all frequencies, λ < $(round(Int, λcut / 1e3)) km (approximate baroclinic)")

function plot_amp(p, ptidx, group, suffix)
    A_north, A_south = amp_all[(p, "ssh_north$suffix")], amp_all[(p, "ssh_south$suffix")]
    ratio = (A_south ./ A_north) .^ 2   # energy ratio south / north
    amp[(p, group)] = (north = A_north, south = A_south, ratio = ratio)
    println("$p $group amplitude ($(plabel(ptidx))): median north $(round(median(filter(!isnan, A_north)) * 100, digits = 2)) cm, ",
            "south $(round(median(filter(!isnan, A_south)) * 100, digits = 2)) cm; median energy ratio south/north ",
            "$(round(median(filter(isfinite, ratio)), digits = 2))")

    fig = Figure(size = (1500, 600))
    stride = 2   # every 2nd grid point: nothing visible is lost and drawing is ~4x faster
    sub(a) = a[1:stride:end, 1:stride:end]
    lon_s, lat_s, h_s = sub(lon_rho), sub(lat_rho), sub(h)
    inner(a) = vec(a[edge+1:end-edge, edge+1:end-edge])   # the edges then saturate
    amax = round(quantile(filter(!isnan, vcat(inner(A_north), inner(A_south))), 0.99) * 100, RoundUp; sigdigits = 2)   # cm
    alabel = group == "M2" ? "M2 amplitude [cm]" : "√2 × ssh std [cm]"
    # ratio on a log10 scale so "4x more south" and "4x more north" look equally strong; red = south dominates
    panels = [("A north (grid-north going)", A_north .* 100, :thermal, (0, amax), alabel),
              ("A south (grid-south going)", A_south .* 100, :thermal, (0, amax), alabel),
              ("energy ratio south / north", log10.(ratio), :balance, (-1, 1), "log₁₀(E_south / E_north)")]
    for (k, (title, data, cmap, crange, clabel)) in enumerate(panels)
        ax = topdown_axis(fig[1, k]; title = title, ylabel = k == 1 ? "Latitude" : "")
        hm = plot_curvilinear!(ax, lon_s, lat_s, sub(data); colormap = cmap, colorrange = crange)
        contour!(ax, lon_s, lat_s, h_s; levels = [200], color = :gray60, linewidth = 0.8)
        add_mooring!(ax)
        Colorbar(fig[2, k], hm; vertical = false, label = clabel, flipaxis = false)
    end
    Label(fig[0, :], "$group ssh ($(amp_desc[group])), $(pstr(ptidx))", fontsize = 15)
    fname = joinpath(figure_path, "amp_north_south_$(group)_$(plabel(ptidx)).png")
    save(fname, fig)
    println("saved $fname")
end
for (p, ptidx) in periods, (group, suffix, _) in groups
    plot_amp(p, ptidx, group, suffix)
end

# ============================================================================
# (5) Hovmöller along three grid columns (west, mooring line, east; grid south -> north): save and plot, each period
# ============================================================================
# y = distance along the line from its grid-south (shelf) end, x = time. "north" waves travel up the plot, so their
# phase lines tilt up to the right; "south" waves tilt down. Slope = phase speed along the line.
# Per line and time range (full, Aug–Sep, Oct, Nov, day), seven figures:
#   hovmoller_dir_<group>_*  north and south going (2 panels), for group = total, M2, bc
#   hovmoller_sum_<group>_*  north + south, for group = total, M2, bc
#   hovmoller_raw_*          raw ssh − spatial mean
# Colour ranges are shared by all panels of a figure type and by all three lines (per period), so everything compares.
hov = Dict{String, Any}()
for (lname, ix) in tr_lines
    dist = [0; cumsum(0.5 .* (1 ./ pn[ix, 1:end-1] .+ 1 ./ pn[ix, 2:end]))] ./ 1e3   # [km] along the line
    dist_at(j) = dist[floor(Int, j)] + (j - floor(j)) * (dist[floor(Int, j)+1] - dist[floor(Int, j)])   # fractional eta
    k_on = findall(k -> abs(moor_ij[k][1] - ix) < 1, eachindex(moor_name))   # moorings on this line (none for west/east)
    j200 = findfirst(>=(200), h[ix, :])   # shelf break: first point deeper than 200 m going offshore
    hov[lname] = (ix = ix, dist = dist, h = h[ix, :], d200 = j200 === nothing ? NaN : dist[j200],
                  moor = moor_name[k_on], moor_dist = [dist_at(moor_ij[k][2]) for k in k_on],
                  f = Dict(name => cols[(lname, name)] for name in [ssh_names; "raw_anom"]))   # [eta, time], all nt hours
    println("Hovmöller $lname line: xi = $ix, $(round(dist[end], digits = 1)) km, moorings: ",
            isempty(k_on) ? "none" : join(moor_name[k_on], ", "))
end

# save each line, full record (the day is a subset)
for (lname, ix) in tr_lines
    L = hov[lname]
    tr_fname = joinpath(dirname(ddf_fname), "ssh_ddf_transect_$(lname)_xi$(ix)_$(plabel(1:nt)).nc")
    isfile(tr_fname) && rm(tr_fname)
    NCDataset(tr_fname, "c") do ds
        defDim(ds, "eta_rho", ny_f)
        defDim(ds, "time", nt)
        defVar(ds, "ocean_time", date, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
        defVar(ds, "dist", L.dist, ("eta_rho",); attrib = ["long_name" => "distance along the line from its grid-south end", "units" => "km"])
        defVar(ds, "lon_rho", lon_rho[ix, :], ("eta_rho",); attrib = ["units" => "degree_east"])
        defVar(ds, "lat_rho", lat_rho[ix, :], ("eta_rho",); attrib = ["units" => "degree_north"])
        defVar(ds, "h", L.h, ("eta_rho",); attrib = ["units" => "meter"])
        for name in ssh_names
            defVar(ds, name, L.f[name], ("eta_rho", "time"); fillvalue = NaN32,
                   attrib = ["long_name" => ds_ddf[name].attrib["long_name"], "units" => "meter"])
        end
        defVar(ds, "ssh_raw_anom", L.f["raw_anom"], ("eta_rho", "time"); fillvalue = NaN32,
               attrib = ["long_name" => "raw ssh minus its spatial mean over the domain at each time", "units" => "meter"])
        ds.attrib["title"] = "DFF ssh along the $lname line (grid column xi = $ix, Julia 1-based index), from $ddf_fname"
        ds.attrib["moorings_on_line"] = isempty(L.moor) ? "none" :
            join(["$m at $(round(d, digits = 1)) km" for (m, d) in zip(L.moor, L.moor_dist)], "; ")
    end
    println("wrote $tr_fname")
end

# one Hovmöller panel: ssh [cm], moorings (magenta dashed) and the 200 m shelf break (grey); names only when label = true
function hov_panel!(ax, L, t_ax, data, cmax; label = false)
    hm = heatmap!(ax, t_ax, L.dist, permutedims(data .* 100); colormap = :balance, colorrange = (-cmax, cmax))
    isempty(L.moor_dist) || hlines!(ax, L.moor_dist; color = :magenta, linestyle = :dash, linewidth = 1.5)
    isnan(L.d200) || hlines!(ax, [L.d200]; color = :gray40, linewidth = 1)
    if label
        isempty(L.moor) || text!(ax, fill(t_ax[end], length(L.moor)), L.moor_dist; text = L.moor, align = (:right, :bottom),
                                 fontsize = 10, color = :magenta, font = :bold, offset = (-3, 1))
        isnan(L.d200) || text!(ax, t_ax[end], L.d200; text = "200 m", align = (:right, :top), fontsize = 10, color = :gray30, offset = (-3, -1))
    end
    hm
end

# time ranges for the Hovmöllers: the full record, the three months (the record starts 2022-08-24, so Aug–Sep is one
# piece) and the one-day zoom; a range with no hours in the record (e.g. in a short test file) is left out
hov_chunk(d1, d2) = (a = findfirst(>=(DateTime(d1)), date); b = findlast(<(DateTime(d2)), date);
                     a === nothing || b === nothing || b <= a ? nothing : a:b)
hov_periods = filter(x -> x[2] !== nothing,
                     [("full", 1:nt),
                      ("augsep", hov_chunk(Date(2022, 8, 1), Date(2022, 10, 1))),
                      ("oct", hov_chunk(Date(2022, 10, 1), Date(2022, 11, 1))),
                      ("nov", hov_chunk(Date(2022, 11, 1), Date(2022, 12, 1))),
                      ("day", day_tidx)])
println("Hovmöller periods: ", join(["$p $(pstr(t))" for (p, t) in hov_periods], ", "))

# one figure per group (total, M2, bc) and kind, one panel column each, so the time axis gets the whole width: the
# figure is ~1 unit per hour (900–2400 wide; the png is saved at 2x), enough to see single M2 crests even over 3 months
function plot_hov_period(p, ptidx)
    span_h = length(ptidx) - 1
    tunit, tdiv = span_h <= 72 ? ("hours", 1) : ("days", 24)   # hours for short ranges, days for long ones
    t_ax = (0:span_h) ./ tdiv
    xlab = "$tunit since $(Dates.format(date[ptidx[1]], "yyyy-mm-dd HH:MM"))"
    ylab = "distance along line [km]\n(grid south → north)"
    fsel(L, name) = L.f[name][:, ptidx]
    width = clamp(span_h + 200, 900, 2400)

    # shared colour ranges [cm] for this period: 99th percentile of |ssh| over the interior of all three lines
    lim(get) = round(quantile(abs.(filter(!isnan, reduce(vcat, [vec(get(hov[l])[edge+1:end-edge, :]) for (l, _) in tr_lines]))),
                     0.99) * 100, RoundUp; sigdigits = 2)
    lim_dir = maximum(lim(L -> fsel(L, name)) for name in ssh_names)
    lim_sum = maximum(lim(L -> fsel(L, "ssh_north$sfx") .+ fsel(L, "ssh_south$sfx")) for (_, sfx, _) in groups)
    lim_raw = lim(L -> fsel(L, "raw_anom"))
    println("$p Hovmöller ($(pstr(ptidx))) colour ranges ± [cm]: north/south $lim_dir, north + south $lim_sum, raw − mean $lim_raw")

    for (lname, ix) in tr_lines
        L = hov[lname]
        line_str = "$lname line (xi = $ix)"
        tag = "$(lname)_xi$(ix)_$(plabel(ptidx))"

        for (group, suffix, desc) in groups
            # north and south going
            fig = Figure(size = (width, 950))
            hm = nothing
            for (i, dir) in enumerate(["north", "south"])
                ax = Axis(fig[i, 1]; title = "$dir going", xlabel = i == 2 ? xlab : "", ylabel = ylab)
                hm = hov_panel!(ax, L, t_ax, fsel(L, "ssh_$(dir)$suffix"), lim_dir; label = true)
            end
            Colorbar(fig[3, 1], hm; vertical = false, label = "ssh [cm]", flipaxis = false, width = Relative(0.4))
            Label(fig[0, 1], "$group ssh ($desc) along the $line_str, $(pstr(ptidx))", fontsize = 16)
            save(joinpath(figure_path, "hovmoller_dir_$(group)_$tag.png"), fig)

            # north + south
            fig = Figure(size = (width, 560))
            ax = Axis(fig[1, 1]; title = "north + south", xlabel = xlab, ylabel = ylab)
            hm = hov_panel!(ax, L, t_ax, fsel(L, "ssh_north$suffix") .+ fsel(L, "ssh_south$suffix"), lim_sum; label = true)
            Colorbar(fig[2, 1], hm; vertical = false, label = "ssh [cm]", flipaxis = false, width = Relative(0.4))
            Label(fig[0, 1], "$group ssh ($desc) along the $line_str, $(pstr(ptidx))", fontsize = 16)
            save(joinpath(figure_path, "hovmoller_sum_$(group)_$tag.png"), fig)
        end

        # raw ssh − spatial mean
        fig = Figure(size = (width, 560))
        ax = Axis(fig[1, 1]; title = "raw ssh − spatial mean", xlabel = xlab, ylabel = ylab)
        hm = hov_panel!(ax, L, t_ax, fsel(L, "raw_anom"), lim_raw; label = true)
        Colorbar(fig[2, 1], hm; vertical = false, label = "ssh [cm]", flipaxis = false, width = Relative(0.4))
        Label(fig[0, 1], "raw ssh along the $line_str, $(pstr(ptidx))", fontsize = 16)
        save(joinpath(figure_path, "hovmoller_raw_$tag.png"), fig)
        println("saved hovmoller_{dir,sum}_{total,M2,bc}_$tag.png, hovmoller_raw_$tag.png")
    end
end
for (p, ptidx) in hov_periods
    plot_hov_period(p, ptidx)
    GC.gc()
end
GC.gc()

# ============================================================================
# (6) hourly snapshot frames: raw ssh − spatial mean vs the DFF pieces — the long part, so it comes last
# ============================================================================
# Each frame: column 1 raw ssh minus its spatial mean; columns 2–4 all frequencies / M2 / bc, with north + south on
# top, then north, then south. One colour range for every panel and every frame (from a sample of hours), so all
# columns and all frames compare directly. Frames are read one at a time (memory stays small) and a frame whose png
# exists is skipped, so an interrupted run just continues. ~15–20 s per frame: the full record is ~10–13 h on one
# process, so the frames are drawn by the workers in parallel (pmap hands each worker the next frame when it is free).
# north + south (all frequencies) must equal raw minus the plane removed in DDF_on_ssh.jl step 2b — printed per frame.
frame_tidx = 1:nt   # every hour of the record; e.g. 1:3:nt for every 3rd hour (3x faster), or day_tidx for one day
overwrite = false   # true: redraw frames that already exist

@everywhere begin
    plane_a, plane_b, plane_c = ds_ddf["plane_a"][:], ds_ddf["plane_b"][:], ds_ddf["plane_c"][:]
    plane(t) = plane_a[t] .+ plane_b[t] .* (1:nx_f) .+ plane_c[t] .* (1:ny_f)'   # [xi, eta], as removed in step 2b
    frame_fname(t) = joinpath(figure_path, "ssh_check_$(Dates.format(date[t], "yyyymmddHH")).png")
end
columns = [("raw ssh − spatial mean", nothing, nothing),
           ("all frequencies", "ssh_north", "ssh_south"),
           ("M2 (9–15 h, λ < $(round(Int, λcut / 1e3)) km)", "ssh_north_M2", "ssh_south_M2"),
           ("bc (all freq, λ < $(round(Int, λcut / 1e3)) km)", "ssh_north_bc", "ssh_south_bc")]

# one colour range for all frames: 99th percentile of |panel| over a sample of hours (every 24th frame, every 4th point)
frame_clim = let samp = Float32[]
    q4(a) = vec(abs.(a[1:4:end, 1:4:end]))
    for t in frame_tidx[1:24:end]
        append!(samp, q4(read_raw(t) .- raw_mean[t]))
        for (_, nn, sn) in columns[2:end]
            append!(samp, q4(read_map(nn, t) .+ read_map(sn, t)))
        end
    end
    round(quantile(filter(!isnan, samp), 0.99), RoundUp; sigdigits = 2)
end
println("frames: $(length(frame_tidx)) ($(pstr(frame_tidx))), colour range ± $frame_clim m")

# what the workers need from this process ($ copies the value over)
@everywhere frame_clim = $frame_clim
@everywhere columns = $columns
@everywhere raw_mean = $raw_mean

@everywhere function draw_frame(t)
    raw = read_raw(t)
    f = Dict(name => read_map(name, t) for name in ssh_names)
    err = maximum(abs, f["ssh_north"] .+ f["ssh_south"] .- (raw .- plane(t)))
    stride = 2   # every 2nd grid point: ~300 px per panel, nothing visible is lost and drawing is ~4x faster
    sub(a) = a[1:stride:end, 1:stride:end]
    lon_s, lat_s, h_s = sub(lon_rho), sub(lat_rho), sub(h)

    fig = Figure(size = (1500, 1350))
    hm = nothing
    for (j, (ctitle, nname, sname)) in enumerate(columns)
        rows = nname === nothing ? [("", raw .- raw_mean[t])] :
               [("north + south", f[nname] .+ f[sname]), ("north", f[nname]), ("south", f[sname])]
        for (i, (rtitle, data)) in enumerate(rows)
            ax = topdown_axis(fig[i, j]; title = isempty(rtitle) ? ctitle : (i == 1 ? "$ctitle\n$rtitle" : rtitle),
                              xlabel = i == length(rows) ? "Longitude" : "", ylabel = j == 1 ? "Latitude" : "")
            hm = plot_curvilinear!(ax, lon_s, lat_s, sub(data); colormap = :balance, colorrange = (-frame_clim, frame_clim))
            contour!(ax, lon_s, lat_s, h_s; levels = [200], color = :gray40, linewidth = 0.8)
            add_mooring!(ax)
        end
    end
    Colorbar(fig[4, 1:4], hm; vertical = false, label = "ssh [m]", flipaxis = false, width = Relative(0.5))
    Legend(fig[2, 1], [[MarkerElement(marker = moor_style[g].marker, color = moor_style[g].color, markersize = 12,
                                      strokecolor = :black, strokewidth = 0.75)] for g in ["french", "M", "CPIES"]] ∪
                      [[LineElement(color = :black, linestyle = :dash)], [LineElement(color = :gray40)]],
           ["french", "M1–M4", "CPIES1–9", "Hovmöller lines (xi = $(join(last.(tr_lines), ", ")))", "200 m isobath"];
           framevisible = false, tellwidth = false)   # tellwidth = false: the legend must not narrow the raw-ssh column
    Label(fig[0, :], "DFF ssh check   $(Dates.format(date[t], "yyyy-mm-dd HH:MM"))   (north/south = grid-north/south going)", fontsize = 16)
    save(frame_fname(t), fig)
    println("saved $(basename(frame_fname(t)))   max |north + south − (raw − plane)| = $(round(err, sigdigits = 2)) m")
    # free the figure before the next one (as free_memory() in energyflux_tile_working_hpc.jl): CairoMakie keeps
    # each figure's buffers until collected
    empty!(fig)
    GC.gc()
    Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)
end
frames_todo = [t for t in frame_tidx if overwrite || !isfile(frame_fname(t))]
println("$(length(frames_todo)) of $(length(frame_tidx)) frames to draw on $(nworkers()) workers")
@time pmap(draw_frame, frames_todo)
@everywhere close(ds_raw)

# ============================================================================
# (7) movies from the frames: the full record and the one-day zoom (same ffmpeg call as figure2movie/figure_make_movie.jl)
# ============================================================================
using FFMPEG_jll
movies = [(frame_tidx, 24),   # full record: 24 frames/s = one model day per second (~100 s for the 3 months)
          (day_tidx, 4)]      # one day: 4 frames/s, 6 s

for (mtidx, fps) in movies
    pngs = filter(isfile, frame_fname.(mtidx))   # in time order
    isempty(pngs) && (println("no frames for $(plabel(mtidx)), skipping movie"); continue)
    length(pngs) == length(mtidx) || @warn "$(length(pngs)) of $(length(mtidx)) frames found for $(plabel(mtidx)) — movie will have gaps"
    movie_fname = joinpath(figure_path, "ssh_check_$(plabel(mtidx))_movie.mp4")
    listfile = movie_fname * "_filelist.txt"
    open(io -> foreach(p -> println(io, "file '$p'"), pngs), listfile, "w")
    # pad to even width/height (libx264 needs it); yuv420p so the mp4 plays in browsers/PowerPoint. -threads caps the
    # png decoder (before -i) and the x264 encoder (after -vcodec) threads: by default ffmpeg starts one per CPU of the
    # node, which on a 128-core node runs out of memory
    FFMPEG_jll.ffmpeg() do ffmpeg_path
        run(`$ffmpeg_path -y -loglevel error -threads 2 -r $fps -f concat -safe 0 -i $listfile -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -threads 4 -pix_fmt yuv420p $movie_fname`)
    end
    rm(listfile)
    println("saved movie $movie_fname ($(length(pngs)) frames at $fps fps)")
end
