using NCDatasets, Dates, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
ddf_fname = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/ddf_ssh/ssh_ddf_north_south.nc"   # made by DDF_on_ssh.jl

# ============================================================================
# (1) read the grid and the DDF output (ssh_north/south, ssh_north/south_M2, ssh_north/south_bc)
# ============================================================================
# The DDF file is kept open and the ssh fields are read only when asked for: each is [xi, eta, time] ≈ 5.5 GB
# for the full run (≈ 33 GB for all six), too much to load at once on a login node.
ds_ddf = NCDataset(ddf_fname)
ssh_names = ["ssh_north", "ssh_south", "ssh_north_M2", "ssh_south_M2", "ssh_north_bc", "ssh_south_bc"]
println("DDF file: ", ddf_fname)
for name in ssh_names
    println("  $name: ", ds_ddf[name].attrib["long_name"])
end

# DDF_on_ssh.jl dropped the last xi and eta row so nx, ny are odd — trim the grid fields the same way
nx_f, ny_f = ds_ddf.dim["xi_rho"], ds_ddf.dim["eta_rho"]
nt = ds_ddf.dim["time"]
lon_rho, lat_rho, mask_rho = ds_ddf["lon_rho"][:, :], ds_ddf["lat_rho"][:, :], ds_ddf["mask_rho"][:, :]
h, pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["h"][1:nx_f, 1:ny_f], ds["pm"][1:nx_f, 1:ny_f], ds["pn"][1:nx_f, 1:ny_f], ds["angle"][1:nx_f, 1:ny_f]
end

# check the trimmed grid lines up with the DDF file (same points, lon in -180..180 in both)
lon_grid = NCDataset(ds -> ds["lon_rho"][1:nx_f, 1:ny_f], grid_fname)
lon_grid[lon_grid .> 180] .-= 360
maximum(abs, lon_grid .- lon_rho) < 1e-6 || error("grid file and DDF file lon_rho do not match — wrong grid or trimming")

date = ds_ddf["ocean_time"][:]   # DateTime (decoded from seconds since 1994-01-01)
fband_M2 = ds_ddf.attrib["M2_band_cph"]
println("grid: $nx_f x $ny_f, time: $nt records, ", date[1], " -> ", date[end], ", dt = ", Dates.canonicalize(date[2] - date[1]))
println("M2 band: periods $(round(1 / fband_M2[2], digits = 1))–$(round(1 / fband_M2[1], digits = 1)) h")

# moorings (lon -180..180, lat), same styles as energyflux_tile_working_hpc.jl. The mooring file stores ROMS 0-based
# fractional grid indices, so Julia index = ROMS index + 1 (checked against the moorings' lon/lat). The main mooring
# line M1 -> CPIES7 is the grid column xi = 343 — the middle of the trimmed DDF grid — running grid south (shelf) to
# grid north (offshore) along eta; it is the Hovmöller transect in section 6.
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
ix_tr = round(Int, moor_ij[findfirst(==("M1"), moor_name)][1])   # transect column (343)
on_tr = findall(k -> abs(moor_ij[k][1] - ix_tr) < 1, eachindex(moor_name))   # moorings on the transect
println("mooring line: xi = $ix_tr, moorings on it: $(join(moor_name[on_tr], ", "))")

# marks every mooring (and the transect, dashed) on a map; Point3f at z = 0 + overdraw keeps them on top of
# plot_curvilinear!'s flat surface
function add_mooring!(ax; transect = true)
    transect && lines!(ax, Point3f.(lon_rho[ix_tr, :], lat_rho[ix_tr, :], 0); color = :black, linestyle = :dash,
                       linewidth = 0.8, overdraw = true)
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

# ============================================================================
# (2) M2 ssh variance against wavelength: where is mode 1, and is the wavenumber cut in the right place?
# ============================================================================
# Saved by DDF_on_ssh.jl from the M2-band spectrum before the cut. Expect the barotropic tide (whatever the plane did
# not remove) at the longest wavelengths, and a mode-1 internal-tide peak near λ1. The cut (dashed red) should sit
# between the two; if the mode-1 peak is not near the assumed λ1 (dotted), change λ1 in DDF_on_ssh.jl and re-run.
using CairoMakie
figure_path = "/expanse/lustre/projects/uso101/hchen54/figure/amazon_900m_3mon_1008/DFF"
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


## ssh check.# ============================================================================
# (3) check one day of snapshots: raw ssh vs the DFF pieces, one png per hour
# ============================================================================
# Each frame: column 1 raw ssh minus its spatial mean (so the colour shows the pattern, not the basin-wide tide going
# up and down); columns 2–4 all frequencies / M2 / bc, with north + south on top, then north, then south.
# north + south (all frequencies) should equal raw minus the plane removed in DDF_on_ssh.jl step 2b — printed per hour.
# The 200 m isobath (grey) marks the shelf break: inshore of it the M2 and bc fields are less reliable (see DDF_on_ssh.jl).
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # topdown_axis, plot_curvilinear!

ssh_raw_fname = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/ddf_ssh/roms_ssh_hourly.nc"   # made by DDF_on_ssh.jl (1b)
check_day = Date(2022, 9, 4)
overwrite = true    # true: redraw all 24 frames (raw panel changed); set back to false so an interrupted run just continues

ds_raw = NCDataset(ssh_raw_fname)
date_raw = ds_raw["ocean_time"][:]
plane_a, plane_b, plane_c = ds_ddf["plane_a"][:], ds_ddf["plane_b"][:], ds_ddf["plane_c"][:]
plane(t) = plane_a[t] .+ plane_b[t] .* (1:nx_f) .+ plane_c[t] .* (1:ny_f)'   # [xi, eta], as removed in step 2b

# read the day: 24 hours x 7 fields x ~2.3 MB, small enough to hold
hours = DateTime(check_day) .+ Hour.(0:23)
snaps = map(hours) do d
    t, tr = tindex(d), findfirst(==(d), date_raw)
    (date[t] == d && tr !== nothing) || error("$d is not in the DDF or the raw ssh file")
    s = Dict(name => read_map(name, t) for name in ssh_names)
    s["raw"] = nomissing(ds_raw["zeta"][1:nx_f, 1:ny_f, tr], NaN32)   # same trim as the DDF fields
    s["raw_anom"] = s["raw"] .- mean(filter(!isnan, s["raw"]))         # minus the spatial mean (ocean points) at this hour
    s["plane"] = plane(t)
    s
end

# check: north + south = raw - plane at every hour
for (d, s) in zip(hours, snaps)
    err = maximum(abs, s["ssh_north"] .+ s["ssh_south"] .- (s["raw"] .- s["plane"]))
    println("  $(Dates.format(d, "yyyy-mm-dd HH:MM")): max |north + south - (raw - plane)| = $(round(err, sigdigits = 2)) m")
end

# fixed colour range per column for the whole day: 99th percentile of |north + south|, rounded up to 2 significant digits
lim(f) = round(quantile(abs.(filter(!isnan, vec(reduce(hcat, [vec(f(s)) for s in snaps])))), 0.99), RoundUp; sigdigits = 2)
columns = [("raw ssh − spatial mean", "raw", nothing, nothing),
           ("all frequencies", "", "ssh_north", "ssh_south"),
           ("M2 (9–15 h, λ < $(round(Int, λcut / 1e3)) km)", "_M2", "ssh_north_M2", "ssh_south_M2"),
           ("bc (all freq, λ < $(round(Int, λcut / 1e3)) km)", "_bc", "ssh_north_bc", "ssh_south_bc")]
clims = [c[3] === nothing ? lim(s -> s["raw_anom"]) : lim(s -> s[c[3]] .+ s[c[4]]) for c in columns]
println("colour ranges ± [m]: ", join(["$(c[1]) $l" for (c, l) in zip(columns, clims)], ", "))

stride = 2   # every 2nd grid point: ~300 px per panel, nothing visible is lost and drawing is ~4x faster
sub(a) = a[1:stride:end, 1:stride:end]
lon_s, lat_s, h_s = sub(lon_rho), sub(lat_rho), sub(h)

for (d, s) in zip(hours, snaps)
    fname = joinpath(figure_path, "ssh_check_$(Dates.format(d, "yyyymmddHH")).png")
    !overwrite && isfile(fname) && continue
    fig = Figure(size = (1500, 1350))
    for (j, (ctitle, _, nname, sname)) in enumerate(columns)
        rows = nname === nothing ? [("", s["raw_anom"])] :
               [("north + south", s[nname] .+ s[sname]), ("north", s[nname]), ("south", s[sname])]
        hm = nothing
        for (i, (rtitle, data)) in enumerate(rows)
            ax = topdown_axis(fig[i, j]; title = isempty(rtitle) ? ctitle : (i == 1 ? "$ctitle\n$rtitle" : rtitle),
                              xlabel = i == length(rows) ? "Longitude" : "", ylabel = j == 1 ? "Latitude" : "")
            hm = plot_curvilinear!(ax, lon_s, lat_s, sub(data); colormap = :balance, colorrange = (-clims[j], clims[j]))
            contour!(ax, lon_s, lat_s, h_s; levels = [200], color = :gray40, linewidth = 0.8)
            add_mooring!(ax)
        end
        Colorbar(fig[4, j], hm; vertical = false, label = "ssh [m]", flipaxis = false)
    end
    Legend(fig[2, 1], [[MarkerElement(marker = moor_style[g].marker, color = moor_style[g].color, markersize = 12,
                                      strokecolor = :black, strokewidth = 0.75)] for g in ["french", "M", "CPIES"]] ∪
                      [[LineElement(color = :black, linestyle = :dash)], [LineElement(color = :gray40)]],
           ["french", "M1–M4", "CPIES1–9", "mooring line (xi = $ix_tr)", "200 m isobath"];
           framevisible = false, tellwidth = false)   # tellwidth = false: the legend must not narrow the raw-ssh column
    Label(fig[0, :], "DFF ssh check   $(Dates.format(d, "yyyy-mm-dd HH:MM"))   (north/south = grid-north/south going)", fontsize = 16)
    save(fname, fig)
    println("saved $fname")
    # free the figure before the next one (as free_memory() in energyflux_tile_working_hpc.jl): CairoMakie keeps
    # each figure's buffers until collected, and the login node caps a process at 8 GB
    empty!(fig)
    GC.gc()
    Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)
end
close(ds_raw)

# ============================================================================
# (4) movie of the check day's hourly frames (same ffmpeg call as figure2movie/figure_make_movie.jl)
# ============================================================================
using FFMPEG_jll
movie_fps = 4   # 24 hourly frames -> a 6 s movie; lower it to slow the movie down

let prefix = "ssh_check_$(Dates.format(check_day, "yyyymmdd"))"   # this day's frames only, in time order
    pngs = sort(filter(f -> startswith(basename(f), prefix) && endswith(f, ".png"), readdir(figure_path, join = true)))
    length(pngs) == 24 || @warn "$(length(pngs)) of 24 hourly frames found for $check_day — movie will have gaps"
    listfile = joinpath(figure_path, "$(prefix)_filelist.txt")
    open(io -> foreach(p -> println(io, "file '$p'"), pngs), listfile, "w")
    movie_fname = joinpath(figure_path, "$(prefix)_movie.mp4")
    # pad to even width/height (libx264 needs it); yuv420p so the mp4 plays in browsers/PowerPoint
    FFMPEG_jll.ffmpeg() do ffmpeg_path
        run(`$ffmpeg_path -y -loglevel error -r $movie_fps -f concat -safe 0 -i $listfile -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p $movie_fname`)
    end
    rm(listfile)
    println("saved movie $movie_fname ($(length(pngs)) frames at $movie_fps fps)")
end


# ============================================================================
# (5) amplitude maps for each direction, and the south/north energy ratio — total, M2 and bc
# ============================================================================
# amplitude = √2 × std over time (variation about the time mean) — exact for one sine wave.
#   M2: the 9–15 h band also holds S2, N2, K2, so this is an effective amplitude averaged over whatever part of the
#       spring–neap cycle (~14.8 days) the time range covers: fine for testing on one day, use the full record for real.
#   total / bc: many frequencies, so not a tidal amplitude — read it as √2 × the ssh std. The time mean is taken out
#       first: a steady ssh pattern is not wave energy (and has no clear travel direction).
# Energy ∝ amplitude² with the same factor for both directions at one place (it depends on depth and stratification,
# not on direction), so (A_south / A_north)² is the south/north energy ratio there.
# The maps are built one snapshot at a time (running sums), so memory stays small even for the full record.
amp_tidx = findall(d -> Date(d) == check_day, date)   # test: the 24 hours of check_day
# amp_tidx = 1:nt                                      # full record: ~2353 map reads per field, run as a batch job

function amp_map(name, tidx)
    s1, s2 = zeros(nx_f, ny_f), zeros(nx_f, ny_f)
    for t in tidx
        a = Float64.(read_map(name, t))
        s1 .+= a
        s2 .+= a .^ 2
    end
    n = length(tidx)
    sqrt.(2 .* max.(s2 ./ n .- (s1 ./ n) .^ 2, 0))   # √2 × std [m]; max(…, 0) guards round-off below zero
end

amp_label = "$(Dates.format(date[amp_tidx[1]], "yyyymmddHH"))-$(Dates.format(date[amp_tidx[end]], "yyyymmddHH"))"
M2_band_str = "$(round(1 / fband_M2[2], digits = 1))–$(round(1 / fband_M2[1], digits = 1)) h"
amp_groups = [("total", "", "all frequencies, all wavelengths"),
              ("M2", "_M2", "M2 band ($M2_band_str), λ < $(round(Int, λcut / 1e3)) km"),
              ("bc", "_bc", "all frequencies, λ < $(round(Int, λcut / 1e3)) km (approximate baroclinic)")]
amp = Dict{String, NamedTuple}()   # amp["M2"].north, amp["bc"].ratio, ...  [xi, eta]

for (group, suffix, desc) in amp_groups
    A_north = amp_map("ssh_north$suffix", amp_tidx)
    A_south = amp_map("ssh_south$suffix", amp_tidx)
    ratio = (A_south ./ A_north) .^ 2   # energy ratio south / north
    amp[group] = (north = A_north, south = A_south, ratio = ratio)
    println("$group amplitude over $(length(amp_tidx)) hours ($amp_label): median north $(round(median(filter(!isnan, A_north)) * 100, digits = 2)) cm, ",
            "south $(round(median(filter(!isnan, A_south)) * 100, digits = 2)) cm; median energy ratio south/north ",
            "$(round(median(filter(isfinite, ratio)), digits = 2))")

    fig = Figure(size = (1500, 600))
    stride = 2
    sub(a) = a[1:stride:end, 1:stride:end]
    lon_s, lat_s, h_s = sub(lon_rho), sub(lat_rho), sub(h)
    # colour range from the interior only: the FFT edge leakage makes bands along the domain edges far stronger than
    # the interior and would wash it out; skip 30 grid points (~27 km) along each edge (the edges then saturate)
    edge = 30
    inner(a) = vec(a[edge+1:end-edge, edge+1:end-edge])
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
    Label(fig[0, :], "$group ssh ($desc), " *
                     "$(Dates.format(date[amp_tidx[1]], "yyyy-mm-dd HH:MM")) – $(Dates.format(date[amp_tidx[end]], "yyyy-mm-dd HH:MM"))", fontsize = 15)
    fname = joinpath(figure_path, "amp_north_south_$(group)_$(amp_label).png")
    save(fname, fig)
    println("saved $fname")
    empty!(fig)
    GC.gc()
end

## holvmolar for thransect

# ============================================================================
# (6) Hovmöller along the mooring line (xi = ix_tr, grid south -> north): extract the six fields, save, plot
# ============================================================================
# y = distance along the line from its grid-south (shelf) end, x = time. "north" waves travel up the plot, so their
# phase lines tilt up to the right; "south" waves tilt down. Slope = phase speed along the line.
hov_tidx = amp_tidx[1]:amp_tidx[end]   # same hours as section 5 (check_day); 1:nt for the full record
hov_label = "$(Dates.format(date[hov_tidx[1]], "yyyymmddHH"))-$(Dates.format(date[hov_tidx[end]], "yyyymmddHH"))"

dist_km = [0; cumsum(0.5 .* (1 ./ pn[ix_tr, 1:end-1] .+ 1 ./ pn[ix_tr, 2:end]))] ./ 1e3   # [km] along the line
dist_at(j) = dist_km[floor(Int, j)] + (j - floor(j)) * (dist_km[floor(Int, j)+1] - dist_km[floor(Int, j)])   # fractional eta
moor_dist = [dist_at(moor_ij[k][2]) for k in on_tr]
h_tr = h[ix_tr, :]
d200 = dist_km[findfirst(>=(200), h_tr)]   # shelf break: first point deeper than 200 m going offshore

# extract [eta, time] along the line: one read per field (reads only this column of each snapshot)
hov = Dict(name => nomissing(ds_ddf[name][ix_tr, :, hov_tidx], NaN32) for name in ssh_names)
println("Hovmöller: xi = $ix_tr, $(length(dist_km)) points over $(round(dist_km[end], digits = 1)) km, $(length(hov_tidx)) hours ($hov_label)")

# save the transect for other analysis
tr_fname = joinpath(dirname(ddf_fname), "ssh_ddf_transect_xi$(ix_tr)_$(hov_label).nc")
isfile(tr_fname) && rm(tr_fname)
NCDataset(tr_fname, "c") do ds
    defDim(ds, "eta_rho", ny_f)
    defDim(ds, "time", length(hov_tidx))
    defVar(ds, "ocean_time", date[hov_tidx], ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
    defVar(ds, "dist", dist_km, ("eta_rho",); attrib = ["long_name" => "distance along the mooring line from its grid-south end", "units" => "km"])
    defVar(ds, "lon_rho", lon_rho[ix_tr, :], ("eta_rho",); attrib = ["units" => "degree_east"])
    defVar(ds, "lat_rho", lat_rho[ix_tr, :], ("eta_rho",); attrib = ["units" => "degree_north"])
    defVar(ds, "h", h_tr, ("eta_rho",); attrib = ["units" => "meter"])
    for name in ssh_names
        defVar(ds, name, hov[name], ("eta_rho", "time"); fillvalue = NaN32,
               attrib = ["long_name" => ds_ddf[name].attrib["long_name"], "units" => "meter"])
    end
    ds.attrib["title"] = "DFF ssh along the mooring line (grid column xi = $ix_tr, Julia 1-based index), from $ddf_fname"
    ds.attrib["moorings_on_line"] = join(["$(moor_name[k]) at $(round(d, digits = 1)) km" for (k, d) in zip(on_tr, moor_dist)], "; ")
end
println("wrote $tr_fname")

let fig = Figure(size = (1500, 1000))
    span_h = length(hov_tidx) - 1
    tunit, tdiv = span_h <= 72 ? ("hours", 1) : ("days", 24)   # hours for short ranges, days for long ones
    t_ax = (0:span_h) ./ tdiv
    edge = 30   # colour range from the interior: skip the edge-leakage bands at both ends of the line
    groups = [("total", "", "all frequencies"), ("M2", "_M2", "M2 band, λ < $(round(Int, λcut / 1e3)) km"),
              ("bc", "_bc", "all freq, λ < $(round(Int, λcut / 1e3)) km")]
    for (j, (group, suffix, desc)) in enumerate(groups)
        both = vcat(vec(hov["ssh_north$suffix"][edge+1:end-edge, :]), vec(hov["ssh_south$suffix"][edge+1:end-edge, :]))
        cmax = round(quantile(abs.(filter(!isnan, both)), 0.99) * 100, RoundUp; sigdigits = 2)   # cm
        hm = nothing
        for (i, dir) in enumerate(["north", "south"])
            ax = Axis(fig[i, j]; title = i == 1 ? "$group ($desc)\n$dir going" : "$dir going",
                      xlabel = i == 2 ? "$tunit since $(Dates.format(date[hov_tidx[1]], "yyyy-mm-dd HH:MM"))" : "",
                      ylabel = j == 1 ? "distance along mooring line [km]\n(grid south → north)" : "")
            hm = heatmap!(ax, t_ax, dist_km, permutedims(hov["ssh_$(dir)$suffix"] .* 100);
                          colormap = :balance, colorrange = (-cmax, cmax))
            hlines!(ax, moor_dist; color = :black, linestyle = :dash, linewidth = 0.6)
            hlines!(ax, [d200]; color = :gray40, linewidth = 1)
            if j == 3   # mooring names once, on the right-hand column
                text!(ax, fill(t_ax[end], length(on_tr)), moor_dist; text = moor_name[on_tr], align = (:right, :bottom),
                      fontsize = 10, offset = (-3, 1))
                text!(ax, t_ax[end], d200; text = "200 m", align = (:right, :top), fontsize = 10, color = :gray30, offset = (-3, -1))
            end
        end
        Colorbar(fig[3, j], hm; vertical = false, label = "ssh [cm]", flipaxis = false)
    end
    Label(fig[0, :], "DFF ssh along the mooring line (xi = $ix_tr), $(Dates.format(date[hov_tidx[1]], "yyyy-mm-dd HH:MM")) – " *
                     "$(Dates.format(date[hov_tidx[end]], "yyyy-mm-dd HH:MM"))", fontsize = 16)
    fname = joinpath(figure_path, "hovmoller_xi$(ix_tr)_$(hov_label).png")
    save(fname, fig)
    println("saved $fname")
end
