using NCDatasets, Statistics, LinearAlgebra, Random, FFTW, Dates, CairoMakie

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")   # dff_run, run_dir, run_grid_fname, fig_dir (set by DFF_RUN)
include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings

# Vertical mode structures phi_n(z) and the stratification N2(z) behind them, from bcef_modes_puv.jl,
# at a few typical places: shelf, deep ocean near the shelf break, the moorings, and the deep-ocean average.
NMODE = parse(Int, get(ENV, "NMODE", "3"))
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, "ddf_bcef"))
struct_fname = joinpath(bcef_dir, NMODE == 1 ? "bcef_mode1_structure.nc" : "bcef_modes$(NMODE)_structure.nc")
isfile(struct_fname) || error("no mode structure file $struct_fname: run bcef_modes_puv.jl with NMODE = $NMODE first")
figure_path = joinpath(dirname(fig_dir), "mode_analysis")
mkpath(figure_path)
# mooring file per run (default: the 900 m one)
run_mooring = Dict("amazon_2700m_cmems_0801" => "roms_cmems_grd_2700m_mor_edata.nc")
mooring_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/$(get(run_mooring, dff_run, "roms_grd_900m_mor_edata.nc"))"
puv_fname = joinpath(bcef_dir, NMODE == 1 ? "bcef_mode1_puv.nc" : "bcef_modes$(NMODE)_puv.nc")   # steps 4 and 5
h_mask = 200.0   # [m] shelf not drawn in steps 4 and 5
# steps to run after (1) and (2), which always run: MODE_STEPS=5 or MODE_STEPS=4,5 (default all: 3,4,5,6,7,8).
# Step 3 reads ~200 MB of the structure file and step 5 the whole pn record once (then a small cache file).
mode_steps = parse.(Int, split(get(ENV, "MODE_STEPS", "3,4,5,6,7,8"), ","))
println("steps: 1, 2, ", join(mode_steps, ", "))

# ============================================================================
# (1) read the data
# ============================================================================
# 2D fields and the grid are read whole; the 3D/4D fields (z_r_mean, rho_mean, N2, phi) stay on disk
# and are read one column at a time (phi alone is ~0.9 GB on the 900 m grid).
ds = NCDataset(struct_fname)
nx, ny, nz = ds.dim["xi_rho"], ds.dim["eta_rho"], ds.dim["s_rho"]
nmode = ds.dim["mode"]
h = Float64.(ds["h"][:, :])
zeta_mean = Float64.(ds["zeta_mean"][:, :])
c = Float64.(ds["c"][:, :, :])            # (xi, eta, mode) [m/s]
lon_rho, lat_rho, pn = NCDataset(run_grid_fname) do g
    g["lon_rho"][1:nx, 1:ny], g["lat_rho"][1:nx, 1:ny], Float64.(g["pn"][1:nx, 1:ny])
end

# one water column at (i, j): mean depths of the rho levels, N2 on the interior interfaces, and the modes
function read_column(i, j)
    z_r = Float64.(ds["z_r_mean"][i, j, :])            # (nz) [m], bottom -> surface
    N2 = Float64.(ds["N2"][i, j, :])                   # (nz - 1) [s^-2]
    z_w = (z_r[1:end-1] .+ z_r[2:end]) ./ 2            # interior interfaces, where N2 sits (midpoints: the file has no z_w_mean)
    rho = Float64.(ds["rho_mean"][i, j, :])            # (nz) [kg m^-3]
    phi = Float64.(ds["phi"][i, j, :, :])              # (nz, mode)
    (; i, j, H = h[i, j] + zeta_mean[i, j], z_r, z_w, N2, rho, phi, c = c[i, j, :])
end

# moorings, as fractional (xi, eta) grid indices -> nearest rho point
moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
moor_lon = mod.(moor_lon, 360)   # the grid lon is 0..360, the mooring file -180..180
moor_ij = NCDataset(mooring_file) do g
    [(round(Int, g["$(n)_mor"][1] + 1), round(Int, g["$(n)_mor"][2] + 1)) for n in moor_name]
end

println("$dff_run: $struct_fname")
println("grid $nx x $ny x $nz, $nmode modes; h $(round.(extrema(h), digits = 1)) m")
println("$(length(moor_name)) moorings: ", join(["$n (h = $(round(h[ij...], digits = 0)) m)" for (n, ij) in zip(moor_name, moor_ij)], ", "))
println("figures -> $figure_path")

# ============================================================================
# (2) locations
# ============================================================================
# Along the mooring line (the middle xi line, as in the DFF transect plots), from the coast seaward:
# the first point deeper than each target depth, then points at a set distance along the line
# (distance from the grid-south end, as on the Hovmöller plots: the "node" seen ~200 km up the line in the DFF ssh / flux).
# The moorings are all in deep water (> 3000 m): M1 (nearest the slope), CPIES3 and french (mid line), CPIES7 (far offshore).
# The deep-ocean average uses the region h >= h_avg away from the edges (the same edge mask as the DFF plots).
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # topdown_axis, plot_curvilinear!
ix_line = round(Int, 0.5 + nx / 2)
line_targets = [("shelf", 100.0), ("upper slope", 1000.0), ("lower slope", 2000.0)]   # [m]
line_dists = [("200 km node", 200.0)]   # [km] along the line
line_moorings = ["M1", "CPIES3", "french", "CPIES7"]
edge = 60
h_avg = 3000.0   # [m]

first_deeper(H) = findfirst(>=(H), h[ix_line, :])
points = [(; name, i = ix_line, j = first_deeper(H)) for (name, H) in line_targets]
line_km = [0; cumsum(0.5 .* (1 ./ pn[ix_line, 1:end-1] .+ 1 ./ pn[ix_line, 2:end]))] ./ 1e3   # [km] along the line
for (name, d) in line_dists
    push!(points, (; name, i = ix_line, j = argmin(abs.(line_km .- d))))
end
for n in line_moorings
    k = findfirst(==(n), moor_name)
    k === nothing && error("no mooring $n in $mooring_file")
    push!(points, (; name = n, i = moor_ij[k][1], j = moor_ij[k][2]))
end
deep_avg = (h .>= h_avg) .& (edge .< (1:nx) .<= nx - edge) .& (edge .< (1:ny)' .<= ny - edge)
point_color = Makie.to_colormap(:tab10)   # 10 distinct colours

println("\nlocations (xi = $ix_line for the line points):")
for (k, p) in enumerate(points)
    println("  $(rpad(p.name, 12)) (i, j) = ($(p.i), $(p.j)), h = $(round(h[p.i, p.j])) m, ",
            "$(round(line_km[p.j], digits = 1)) km up the line, lon/lat = $(round(lon_rho[p.i, p.j], digits = 3))/$(round(lat_rho[p.i, p.j], digits = 3)), c = $(round.(c[p.i, p.j, :], digits = 2)) m/s")
end
println("  deep average: $(count(deep_avg)) columns with h >= $(Int(h_avg)) m, median c = $(round.([median(c[:, :, n][deep_avg]) for n in 1:nmode], digits = 2)) m/s")

let fig = Figure(size = (700, 750))
    ax = topdown_axis(fig[1, 1]; title = "$dff_run: mode-analysis locations")
    hm = plot_curvilinear!(ax, lon_rho, lat_rho, h; colormap = Reverse(:deep), colorrange = (0, 5000))
    contour!(ax, lon_rho, lat_rho, h; levels = [200, 1000, 2000, 3000], color = :gray40, linewidth = 0.6)
    contour!(ax, lon_rho, lat_rho, Float64.(deep_avg); levels = [0.5], color = :black, linewidth = 1.5)
    lines!(ax, lon_rho[ix_line, :], lat_rho[ix_line, :]; color = :black, linestyle = :dash, linewidth = 0.8)
    scatter!(ax, moor_lon, moor_lat; marker = :circle, markersize = 5, color = :yellow, strokewidth = 0.5)
    for (k, p) in enumerate(points)
        scatter!(ax, [lon_rho[p.i, p.j]], [lat_rho[p.i, p.j]]; marker = :star5, markersize = 16,
                 color = point_color[k], strokewidth = 0.8, label = "$(p.name) ($(round(Int, h[p.i, p.j])) m)")
    end
    lines!(ax, [NaN], [NaN]; color = :black, linewidth = 1.5, label = "deep average, h ≥ $(Int(h_avg)) m")
    axislegend(ax; position = :rb, framevisible = true, labelsize = 11)
    Colorbar(fig[1, 2], hm; label = "h [m]")
    save(joinpath(figure_path, "mode_locations.png"), fig; px_per_unit = 2)
end
println("saved mode_locations.png")

# bathymetry section along the mooring line with the points on it, next to a small location map
let fig = Figure(size = (1500, 520))
    on_line = [p.i == ix_line || abs(p.i - ix_line) <= 1 for p in points]   # the moorings sit on xi = ix_line - 1
    # map
    ax1 = topdown_axis(fig[1, 1]; title = "locations")
    hm = plot_curvilinear!(ax1, lon_rho, lat_rho, h; colormap = Reverse(:deep), colorrange = (0, 5000))
    contour!(ax1, lon_rho, lat_rho, Float64.(deep_avg); levels = [0.5], color = :black, linewidth = 1)
    lines!(ax1, lon_rho[ix_line, :], lat_rho[ix_line, :]; color = :red, linewidth = 1.5)
    scatter!(ax1, moor_lon, moor_lat; marker = :circle, markersize = 5, color = :yellow, strokewidth = 0.5)
    for (k, p) in enumerate(points)
        scatter!(ax1, [lon_rho[p.i, p.j]], [lat_rho[p.i, p.j]]; marker = :star5, markersize = 13, color = point_color[k], strokewidth = 0.8)
    end
    Colorbar(fig[1, 2], hm; label = "h [m]")
    # section
    ax2 = Axis(fig[1, 3]; title = "bathymetry along the mooring line (xi = $ix_line)",
               xlabel = "distance along line [km] (grid south → north)", ylabel = "depth [m]")
    band!(ax2, line_km, fill(-5000.0, ny), -h[ix_line, :]; color = :gray80)
    lines!(ax2, line_km, -h[ix_line, :]; color = :black, linewidth = 1.5)
    hlines!(ax2, [-h_avg]; color = :black, linestyle = :dot, linewidth = 0.8)
    text!(ax2, line_km[end], -h_avg; text = "deep average: h ≥ $(Int(h_avg)) m", align = (:right, :bottom), fontsize = 10)
    # every mooring on the line as a thin vertical line, labelled at the top
    for (n, (i, j)) in zip(moor_name, moor_ij)
        abs(i - ix_line) <= 1 || continue
        vlines!(ax2, [line_km[j]]; color = startswith(n, "M") ? :royalblue : :indianred, linewidth = 0.8)
        text!(ax2, line_km[j], -50; text = n, rotation = π / 2, align = (:right, :center), fontsize = 9)
    end
    # the analysis points: filled on the line, open where the point is off the line (at its own depth, at the distance of its j)
    for (k, p) in enumerate(points)
        scatter!(ax2, [line_km[p.j]], [-h[p.i, p.j]]; marker = :star5, markersize = 18, strokewidth = 1.2,
                 color = on_line[k] ? point_color[k] : :transparent, strokecolor = on_line[k] ? :black : point_color[k],
                 label = "$(p.name) ($(round(Int, h[p.i, p.j])) m)" * (on_line[k] ? "" : ", off line"))
    end
    xlims!(ax2, 0, line_km[end]); ylims!(ax2, -5000, 100)
    Legend(fig[1, 4], ax2; framevisible = false, labelsize = 11)
    colsize!(fig.layout, 3, Relative(0.55))
    save(joinpath(figure_path, "mode_locations_section.png"), fig; px_per_unit = 2)
end
println("saved mode_locations_section.png")

# ============================================================================
# figure helper for the sections along the mooring line (steps 5 and 6)
# ============================================================================
# text sized for print: the figure is fig_w units wide and meant to be paper_w_cm wide on paper,
# so 1 pt on paper = fig_w / (paper_w_cm / 2.54 * 72) units
fig_w, paper_w_cm = 1300, 18.0
pt = fig_w / (paper_w_cm / 2.54 * 72)   # [units per pt], ~2.55
mode_color = [:black, :crimson, :royalblue, :darkgreen, :orange][1:nmode]

# panels along the line: top = curves on a log axis, middle = shares [%] (left out when share = nothing), bottom =
# bathymetry with the moorings and analysis points; shelf (h < h_mask) not drawn, edge bands shaded.
# top / share: vectors of (y along the line, label, colour, linewidth); top_hline: a reference value drawn on the top axis
function line_section(fname, title; top, top_ylabel, share = nothing, share_ylabel = "", top_ymin = nothing,
                      top_ylims = nothing, top_hline = nothing)
    fig = Figure(size = (fig_w, share === nothing ? 950 : 1150), fontsize = 10pt)
    xmax = line_km[end]
    ax1 = Axis(fig[1, 1]; ylabel = top_ylabel, yscale = log10, title = title)
    ax2 = share === nothing ? nothing : Axis(fig[2, 1]; ylabel = share_ylabel)
    ax3 = Axis(fig[share === nothing ? 2 : 3, 1]; ylabel = "depth [m]", xlabel = "distance along line [km] (grid south → north)")
    axs = filter(!isnothing, [ax1, ax2, ax3])
    deepline = h[ix_line, :] .>= h_mask   # shelf: modes not meaningful
    nanshelf(a) = ifelse.(deepline, a, NaN)
    top_hline === nothing || hlines!(ax1, [top_hline]; color = :gray30, linestyle = :dot, linewidth = 1.5)
    for (y, lab, col, lw) in top
        lines!(ax1, line_km, nanshelf(y); color = col, linewidth = lw, label = lab)
    end
    if share !== nothing
        for (y, lab, col, lw) in share
            lines!(ax2, line_km, nanshelf(y); color = col, linewidth = lw, label = lab)
        end
    end
    band!(ax3, line_km, fill(-5000.0, ny), -h[ix_line, :]; color = :gray80)
    lines!(ax3, line_km, -h[ix_line, :]; color = :black, linewidth = 1.2)
    # moorings on the line (thin) and the analysis points on the line (coloured, dashed)
    for (nm, (i, j)) in zip(moor_name, moor_ij)
        abs(i - ix_line) <= 1 || continue
        for ax in axs
            vlines!(ax, [line_km[j]]; color = :gray60, linewidth = 0.6)
        end
        text!(ax3, line_km[j], -1200; text = nm, rotation = π / 2, align = (:right, :center), fontsize = 8pt)   # in the water, from -1200 m down
    end
    for (k, p) in enumerate(points)
        abs(p.i - ix_line) <= 1 && p.name in ("200 km node", "M1", "CPIES3") || continue
        for ax in axs
            vlines!(ax, [line_km[p.j]]; color = point_color[k], linestyle = :dash, linewidth = 1.5)
        end
        left = p.name == "200 km node"   # left of its line, clear of the M1 label
        text!(ax3, line_km[p.j], -150; text = left ? p.name * " " : " " * p.name, color = point_color[k],
              align = (left ? :right : :left, :top), fontsize = 10pt)   # in the water above the slope
    end
    # the edge bands of the DFF plots (edge points at each end of the line): sponge / boundary, not to be trusted
    for ax in axs
        vspan!(ax, [0, line_km[end - edge]], [line_km[edge], line_km[end]]; color = (:gray, 0.15))
    end
    if top_ylims === nothing
        vals = filter(v -> isfinite(v) && v > 0, reduce(vcat, [nanshelf(y) for (y, _, _, _) in top]))
        ylims!(ax1, something(top_ymin, 10^floor(log10(minimum(vals)))), 10^ceil(log10(maximum(vals))))
    else
        ylims!(ax1, top_ylims...)
    end
    share === nothing || ylims!(ax2, 0, 100)
    ylims!(ax3, -5000, 100)
    for ax in axs
        xlims!(ax, 0, xmax)
        ax.xticks = 0:50:xmax   # a tick every 50 km
    end
    linkxaxes!(axs...)
    foreach(ax -> hidexdecorations!(ax; grid = false), axs[1:end-1])
    axislegend(ax1; position = :lb, nbanks = 2, framevisible = true, backgroundcolor = (:white, 0.9))   # lower left: no data there
    share !== nothing && length(share) > 1 && any(!isempty(l) for (_, l, _, _) in share) &&
        axislegend(ax2; position = :rt, nbanks = 2, framevisible = true, backgroundcolor = (:white, 0.9))
    nrow = length(axs)
    Label(fig[nrow + 1, 1], "shelf (h < $(Int(h_mask)) m) not drawn; grey shading: $edge-point edge bands;\ngrey vertical lines: moorings on the line; dashed: analysis points"; fontsize = 8pt, tellwidth = false)
    rowsize!(fig.layout, 1, Relative(share === nothing ? 0.6 : 0.45)); rowsize!(fig.layout, nrow, Relative(share === nothing ? 0.27 : 0.2))
    for r in 1:nrow-1
        rowgap!(fig.layout, r, 2pt * 10)   # room between the panels' tick labels
    end
    save(joinpath(figure_path, fname), fig; px_per_unit = 2)
    println("saved $fname")
end

# ρ0 of the run (the his files' rho0, as bcef_modes_puv.jl used for p')
function run_rho0()
    his0 = first(filter(f -> startswith(f, "roms_his") && endswith(f, ".nc"), readdir(joinpath(run_dir, "after_joint"))))
    NCDataset(d -> Float64(d.attrib["rho0"]), joinpath(run_dir, "after_joint", his0))
end

# pn, un or vn of every record along the mooring line: one xi column across all records is millions of scattered reads,
# whole records are contiguous, so blocks of records (all xi, eta) are read and the line kept. Read once from the puv
# file, then kept in a small file per tag (p: ~25 MB, uv: ~50 MB). Returns Dict(var => (eta, time, mode)), dt [h].
function line_series(tag, vars)
    fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_$(tag)_line_xi$(ix_line).nc")
    if isfile(fname)
        println("\n$(join(vars, ", ")) along the line: using $fname (delete it to re-read the puv file)")
        return NCDataset(fname) do d
            Dict(v => Float64.(d[v][:, :, :]) for v in vars), Float64(d.attrib["dt_hours"])
        end
    end
    out, dt = NCDataset(puv_fname) do d
        dt = Dates.value(d["ocean_time"][2] - d["ocean_time"][1]) / 3.6e6   # [h]
        nt_l = d.dim["time"]
        o = Dict(v => zeros(ny, nt_l, nmode) for v in vars)
        t0 = time()
        for t1 in 1:100:nt_l
            tb = t1:min(t1 + 99, nt_l)
            for v in vars, n in 1:nmode
                o[v][:, tb, n] .= nomissing(d["$v$n"][:, :, tb], NaN32)[ix_line, :, :]
            end
            println("  $tag: read records $(last(tb)) / $nt_l ($(round(Int, time() - t0)) s)"); flush(stdout)
        end
        o, dt
    end
    NCDataset(fname * ".tmp", "c") do d
        defDim(d, "eta_rho", ny); defDim(d, "time", size(out[vars[1]], 2)); defDim(d, "mode", nmode)
        for v in vars
            defVar(d, v, Float32.(out[v]), ("eta_rho", "time", "mode");
                   attrib = ["long_name" => "mode-n amplitude $(v)n along the line, every record (from $(basename(puv_fname)))"])
        end
        defVar(d, "distance", Float32.(line_km), ("eta_rho",); attrib = ["long_name" => "distance along the line from the grid-south end", "units" => "km"])
        d.attrib["line"] = "xi = $ix_line (Julia 1-based)"
        d.attrib["dt_hours"] = dt
    end
    mv(fname * ".tmp", fname; force = true)
    println("wrote $fname")
    out, dt
end

# time mean of x^2 in a frequency band, per point and mode: rfft in time, zero the other frequencies, back, square,
# average. x (eta, time, mode); keepf(f in cph) -> true in the band. Returns (eta, mode) and the number of frequencies.
function band_msq(x, keepf, dt)
    nt_x = size(x, 2)
    keep = keepf.((0:div(nt_x, 2)) ./ (nt_x * dt))
    out = zeros(size(x, 1), size(x, 3))
    for n in axes(x, 3)
        X = rfft(x[:, :, n], 2)
        X[:, .!keep] .= 0
        out[:, n] .= vec(mean(irfft(X, nt_x, 2) .^ 2, dims = 2))
    end
    out, count(keep)
end

Hline = h[ix_line, :] .+ zeta_mean[ix_line, :]   # [m] H = h + mean zeta along the line

# frequency bands of steps 5 and 7, as in the DFF: M2 = periods 9-15 h, HH = periods shorter than 9 h (hp9; up to the
# 2 h Nyquist period of hourly output). To add a band, add a line: (name, keep(f in cph), description).
var_bands = [("M2", f -> 1 / 15 <= f <= 1 / 9, "M2-band (9–15 h)"),
             ("HH", f -> f > 1 / 9, "high-pass (periods < 9 h)")]

# ============================================================================
# (3) vertical profiles: N2(z) (top row) and phi_1..phi_n(z) (bottom row), one column per location + the deep average
# ============================================================================
if 3 in mode_steps
# The deep average: the N2 of the deep columns (a strided sample of deep_avg) is interpolated to one depth grid in
# metres (the pycnocline sits at a fixed depth, not at a fixed z/H) and averaged where the columns reach, and the
# modes are solved once from that mean N2 over the median depth, with the same eigenproblem as bcef_modes_puv.jl.
# Behind it, a thin grey cloud of the individual deep columns' N2 and phi, stretched by z/H onto the median depth.

# modes from N2 on the interior interfaces: the solver of bcef_modes_puv.jl (modes()), with N2 given instead of rho
function modes_N2(zr, zw, N2; nmode = nmode, N2min = 1e-7)
    nz = length(zr)
    Hz = diff(zw)
    dzc = diff(zr)
    B = max.(N2, N2min) .* dzc
    d = (1 ./ Hz[2:nz] .+ 1 ./ Hz[1:nz-1]) ./ B
    e = (-1 ./ Hz[2:nz-1]) ./ sqrt.(B[1:end-1] .* B[2:end])
    F = eigen(SymTridiagonal(d, e), 1:nmode)
    cn = 1 ./ sqrt.(F.values)
    φ = zeros(nz, nmode)
    for n in 1:nmode
        W = [0; F.vectors[:, n] ./ sqrt.(B); 0]
        φn = diff(W) ./ Hz
        φn ./= sqrt(sum(φn .^ 2 .* Hz) / sum(Hz))
        φn[end] < 0 && (φn .*= -1)
        φ[:, n] .= φn
    end
    cn, φ
end

# linear interpolation of (z, f) (z increasing) onto zq; NaN outside
function interp_z(z, f, zq)
    out = fill(NaN, length(zq))
    for (q, x) in enumerate(zq)
        (x < z[1] || x > z[end]) && continue
        k = clamp(searchsortedlast(z, x), 1, length(z) - 1)
        w = (x - z[k]) / (z[k+1] - z[k])
        out[q] = (1 - w) * f[k] + w * f[k+1]
    end
    out
end

avg_stride = 6     # every 6th point in xi and eta for the deep average (~8,600 columns on the 900 m grid)
n_cloud = 150      # deep columns drawn in the grey cloud
deep_cols = [(i, j) for j in 1:avg_stride:ny, i in 1:avg_stride:nx if deep_avg[i, j]]
H_med = median(h[i, j] + zeta_mean[i, j] for (i, j) in deep_cols)
z_avg_w = collect(range(-H_med, 0; length = 401))            # uniform interfaces, 400 layers
z_avg_r = (z_avg_w[1:end-1] .+ z_avg_w[2:end]) ./ 2
z_avg_N = z_avg_r[1:end-1] .+ diff(z_avg_r) ./ 2              # interior interfaces where N2 sits (= z_avg_w[2:end-1])

# read the sample in strided blocks (one read per variable, not one per column)
i_s, j_s = 1:avg_stride:nx, 1:avg_stride:ny
@time zr_blk, N2_blk = Float64.(ds["z_r_mean"][i_s, j_s, :]), Float64.(ds["N2"][i_s, j_s, :])
N2_sum, N2_cnt = zeros(length(z_avg_N)), zeros(Int, length(z_avg_N))
for (a, i) in enumerate(i_s), (b, j) in enumerate(j_s)
    deep_avg[i, j] || continue
    zr = zr_blk[a, b, :]
    v = interp_z((zr[1:end-1] .+ zr[2:end]) ./ 2, N2_blk[a, b, :], z_avg_N)
    ok = .!isnan.(v)
    N2_sum[ok] .+= v[ok]; N2_cnt[ok] .+= 1
end
N2_avg = N2_sum ./ max.(N2_cnt, 1)
# below the shallowest sampled column the mean has fewer columns: keep it, but the deepest levels (few columns) are noisy
c_avg, φ_avg = modes_N2(z_avg_r, z_avg_w, N2_avg)
println("\ndeep average: $(length(deep_cols)) columns (every $(avg_stride)th point), median H = $(round(H_med)) m, ",
        "c = $(round.(c_avg, digits = 2)) m/s (vs median of the columns' own c: $(round.([median(c[i, j, n] for (i, j) in deep_cols) for n in 1:nmode], digits = 2)))")

cloud = [read_column(ij...) for ij in shuffle(MersenneTwister(1), deep_cols)[1:min(n_cloud, length(deep_cols))]]

let ncol = length(points) + 1, fig = Figure(size = (230 * ncol + 60, 900))
    mode_color = [:black, :crimson, :royalblue, :darkgreen, :orange][1:nmode]
    cols = [(read_column(p.i, p.j), p.name) for p in points]
    for k in 1:ncol
        deep = k == ncol
        H = deep ? H_med : cols[k][1].H
        ttl = deep ? "deep average\nh ≥ $(Int(h_avg)) m (median $(round(Int, H_med)) m)" : "$(cols[k][2])\nh = $(round(Int, H)) m"
        ax1 = Axis(fig[1, k]; title = ttl, titlesize = 13, titlecolor = deep ? :black : point_color[k],
                   xscale = log10, xlabel = "N² [s⁻²]", ylabel = k == 1 ? "z [m]" : "")
        cn = deep ? c_avg : cols[k][1].c
        ax2 = Axis(fig[2, k]; xlabel = "φₙ", ylabel = k == 1 ? "z [m]" : "", titlesize = 12, titlefont = :regular,
                   title = "c = " * join(string.(round.(cn, digits = 2)), " / ") * " m/s")
        if deep
            for col in cloud
                s = H_med / col.H
                lines!(ax1, max.(col.N2, 1e-7), col.z_w .* s; color = (:gray, 0.15), linewidth = 0.5)
                for n in 1:nmode
                    lines!(ax2, col.phi[:, n], col.z_r .* s; color = (mode_color[n], 0.06), linewidth = 0.5)
                end
            end
            lines!(ax1, max.(N2_avg, 1e-7), z_avg_N; color = :black, linewidth = 2)
            for n in 1:nmode
                lines!(ax2, φ_avg[:, n], z_avg_r; color = mode_color[n], linewidth = 2)
            end
            xlims!(ax2, (extrema(φ_avg) .+ (-0.5, 0.5))...)   # the stretched cloud reaches further; keep the mean in view
        else
            col = cols[k][1]
            lines!(ax1, max.(col.N2, 1e-7), col.z_w; color = :black, linewidth = 1.5)
            for n in 1:nmode
                scatterlines!(ax2, col.phi[:, n], col.z_r; color = mode_color[n], linewidth = 1.5, markersize = 2)
            end
        end
        vlines!(ax2, [0]; color = :gray50, linestyle = :dash, linewidth = 0.8)
        xlims!(ax1, 1e-7, 1e-3)
        for ax in (ax1, ax2)
            ylims!(ax, -H, 0)
        end
        linkyaxes!(ax1, ax2)
    end
    Legend(fig[3, :], [LineElement(color = mode_color[n], linewidth = 2) for n in 1:nmode], ["mode $n" for n in 1:nmode];
           orientation = :horizontal, framevisible = false)
    Label(fig[0, :], "$dff_run: mean stratification and vertical modes (φₙ of p and u, (1/H)∫φₘφₙdz = δₘₙ)"; fontsize = 16, font = :bold)
    rowsize!(fig.layout, 1, Relative(0.4))
    save(joinpath(figure_path, "mode_profiles.png"), fig; px_per_unit = 2)
end
println("saved mode_profiles.png")

end   # (3)

# ============================================================================
# (4) share of the baroclinic pressure variance in each mode, as a map
# ============================================================================
if 4 in mode_steps
# share_n(x, y) = pvar_mode / pvar_total from bcef_modes_puv.jl: time mean of H pn^2 over time mean of ∫p'^2 dz
# (all frequencies). Masked as the DFF plots: the edge bands and the shelf (h < h_mask).
pvar_mode, pvar_total = NCDataset(puv_fname) do d
    Float64.(d["pvar_mode"][:, :, :]), Float64.(d["pvar_total"][:, :])
end
share_isobaths = [1000, 2000, 3000]  # [m]
show_mask = (h .>= h_mask) .& (edge .< (1:nx) .<= nx - edge) .& (edge .< (1:ny)' .<= ny - edge) .& (pvar_total .> 0)
share = [ifelse.(show_mask, 100 .* pvar_mode[:, :, n] ./ pvar_total, NaN) for n in 1:nmode]
share_all = ifelse.(show_mask, 100 .* dropdims(sum(pvar_mode, dims = 3), dims = 3) ./ pvar_total, NaN)
lims = (extrema(lon_rho[show_mask]) .+ (-0.05, 0.05), extrema(lat_rho[show_mask]) .+ (-0.05, 0.05))

println("\nshare of ∫p'² dz per mode (unmasked area, variance weighted): ",
        [round(100 * sum(pvar_mode[:, :, n][show_mask]) / sum(pvar_total[show_mask]), digits = 1) for n in 1:nmode], " %")
for p in points
    println("  $(rpad(p.name, 12)) ", [round(100 * pvar_mode[p.i, p.j, n] / pvar_total[p.i, p.j], digits = 1) for n in 1:nmode], " %")
end

let fig = Figure(size = (1500, 820))
    # colour range per mode: 2nd to 98th percentile of the unmasked share, rounded to 5 %
    map_w = 460   # [px] width of each map
    crange(a) = (v = filter(isfinite, a); (5floor(quantile(v, 0.02) / 5), 5ceil(quantile(v, 0.98) / 5)))
    for n in 1:nmode
        ax = topdown_axis(fig[1, n]; title = "mode $n: $(round(100 * sum(pvar_mode[:, :, n][show_mask]) / sum(pvar_total[show_mask]), digits = 1)) % of ∫p'² dz over the area",
                          ylabel = n == 1 ? "Latitude" : "", backgroundcolor = :gray80, aspect = nothing,
                          width = map_w, height = map_w * -(reverse(lims[2])...) / -(reverse(lims[1])...))   # fixed size at the data aspect
        hm = plot_curvilinear!(ax, lon_rho, lat_rho, share[n]; colormap = :viridis, colorrange = crange(share[n]), nan_color = :gray80)
        for (lev, ls) in zip(share_isobaths, [:dot, :dash, :solid])
            contour!(ax, lon_rho, lat_rho, h; levels = [lev], color = :white, linewidth = 0.9, linestyle = ls)
        end
        scatter!(ax, moor_lon, moor_lat; marker = :circle, markersize = 6, color = :yellow, strokewidth = 0.6)
        for (k, p) in enumerate(points)
            scatter!(ax, [lon_rho[p.i, p.j]], [lat_rho[p.i, p.j]]; marker = :star5, markersize = 15, color = point_color[k],
                     strokewidth = 0.8, strokecolor = :black, label = "$(p.name): $(round(100 * pvar_mode[p.i, p.j, n] / pvar_total[p.i, p.j], digits = 1)) %")
        end
        xlims!(ax, lims[1]...); ylims!(ax, lims[2]...)
        Colorbar(fig[2, n], hm; vertical = false, label = "share of mode $n [%]", flipaxis = false)
        Legend(fig[3, n], ax; nbanks = 2, framevisible = false, labelsize = 11, patchsize = (12, 12), rowgap = 0, tellheight = true)
    end
    Label(fig[0, :], "$dff_run: share of the baroclinic pressure variance in each mode (pvar_mode / pvar_total, all frequencies)\n" *
                     "masked: h < $(Int(h_mask)) m and $edge edge points;  white dotted / dashed / solid: $(join(share_isobaths, " / ")) m;  " *
                     "yellow: moorings;  legend: share at each point";
          fontsize = 14)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    save(joinpath(figure_path, "mode_share_map.png"), fig; px_per_unit = 2)
end
println("saved mode_share_map.png")

end   # (4)

# ============================================================================
# (5) modal pressure variance along the mooring line, per frequency band
# ============================================================================
if 5 in mode_steps
# H <pn^2> (time mean over the record) of each mode in a frequency band, along the middle xi line from the grid-south
# end, and the sum of modes 1..nmode (the modes are orthogonal, so the sum is the band's ∫p'^2 dz of these modes).
# pn along the line (all records) is read from the puv file once and kept in a small file (~25 MB); every band is then
# a band-pass of that file. Bands as in the DFF: M2 = periods 9-15 h, HH = periods shorter than 9 h (hp9; up to the
# 2 h Nyquist period of hourly output). To add a band, add a line: (name, keep(f in cph), description).
pline, dt_h = line_series("p", ["p"])
pline = pline["p"]   # (eta, time, mode) [Pa]

for (bname, bkeep, bdesc) in var_bands
    pv, nkeep = band_msq(pline, bkeep, dt_h)
    pv .*= Hline   # H <pn^2>
    pv_sum = vec(sum(pv, dims = 2))
    println("$bname band ($bdesc): $nkeep frequencies; max H<pn²> = ",
            [round(maximum(filter(isfinite, pv[:, n])), sigdigits = 3) for n in 1:nmode], " Pa² m")
    line_section("mode_$(bname)var_line.png", "$bdesc modal pressure variance along the mooring line\n$dff_run, xi = $ix_line";
                 top = [(pv_sum, "modes 1-$nmode", :gray50, 3); [(pv[:, n], "mode $n", mode_color[n], 1.8) for n in 1:nmode]],
                 top_ylabel = "H ⟨pₙ²⟩  [Pa² m]",
                 share = [(100 .* pv[:, n] ./ pv_sum, "", mode_color[n], 1.8) for n in 1:nmode],
                 share_ylabel = "share of modes 1-$nmode [%]",
                 top_ymin = 10^floor(log10(minimum(filter(v -> isfinite(v) && v > 0, pv[h[ix_line, :] .>= h_mask, nmode])))))
end
end   # (5)

# ============================================================================
# (6) residual of the modal fit along the mooring line: total - modes 1..nmode (all frequencies)
# ============================================================================
if 6 in mode_steps
# pvar_total = time mean of ∫p'^2 dz, pvar_mode = time mean of H pn^2 (bcef_modes_puv.jl, all frequencies; a band-passed
# total would need p' in 3-D, not saved). The modes are orthogonal, so the residual total - Σ_1^nmode H pn^2 is the
# variance in the higher modes (nmode+1 ...). Shares here are of the total (as the share map), not of modes 1..nmode.
pm_line, pt_line = NCDataset(puv_fname) do d
    Float64.(d["pvar_mode"][ix_line, :, :]), Float64.(d["pvar_total"][ix_line, :])   # (eta, mode), (eta)
end
pm_sum = vec(sum(pm_line, dims = 2))
resid = pt_line .- pm_sum
deep_l = (h[ix_line, :] .>= h_mask) .& (edge .< (1:ny) .<= ny - edge)
println("\nresidual total - modes 1-$nmode along the line (deep, inside the edges): ",
        "$(round(100 * sum(resid[deep_l]) / sum(pt_line[deep_l]), digits = 1)) % of the total; ",
        "range $(round.(100 .* extrema(resid[deep_l] ./ pt_line[deep_l]), digits = 1)) %; ",
        "min residual $(round(minimum(resid[deep_l]), sigdigits = 3)) Pa² m (≥ 0 expected)")
line_section("mode_residual_line.png", "modal-fit residual along the mooring line (all frequencies)\n$dff_run, xi = $ix_line";
             top = [(pt_line, "total ∫p′²dz", :gray50, 3), (pm_sum, "modes 1-$nmode", :black, 1.8),
                    (max.(resid, 0), "residual (modes > $nmode)", :darkorange, 2.2)],
             top_ylabel = "time mean  [Pa² m]",
             share = [[(100 .* pm_line[:, n] ./ pt_line, "mode $n", mode_color[n], 1.8) for n in 1:nmode];
                      (100 .* resid ./ pt_line, "residual", :darkorange, 2.2)],
             share_ylabel = "share of the total [%]")
end   # (6)

# ============================================================================
# (7) modal kinetic energy along the mooring line, per frequency band
# ============================================================================
if 7 in mode_steps
# KEn = ½ ρ0 H <un² + vn²> [J m-2], the depth-integrated kinetic energy of mode n (the modes are orthogonal, (1/H) ∫ φm φn dz
# = δmn, so ∫ ½ρ0 (u'² + v'²) dz = Σn ½ρ0 H (un² + vn²)), band-passed as in step 5; ρ0 as in bcef_modes_puv.jl (his rho0).
rho0 = run_rho0()
uv, dt_uv = line_series("uv", ["u", "v"])
for (bname, bkeep, bdesc) in var_bands
    msq_u, nkeep = band_msq(uv["u"], bkeep, dt_uv)
    msq_v, _ = band_msq(uv["v"], bkeep, dt_uv)
    ke = 0.5 .* rho0 .* Hline .* (msq_u .+ msq_v)   # (eta, mode) [J m-2]
    ke_sum = vec(sum(ke, dims = 2))
    println("$bname band KE ($nkeep frequencies, ρ0 = $rho0): max ", [round(maximum(filter(isfinite, ke[:, n])), sigdigits = 3) for n in 1:nmode], " J/m²")
    line_section("mode_$(bname)KE_line.png", "$bdesc modal kinetic energy along the mooring line\n$dff_run, xi = $ix_line";
                 top = [(ke_sum, "modes 1-$nmode", :gray50, 3); [(ke[:, n], "mode $n", mode_color[n], 1.8) for n in 1:nmode]],
                 top_ylabel = "KEₙ = ½ρ₀H⟨uₙ² + vₙ²⟩  [J m⁻²]",
                 share = [(100 .* ke[:, n] ./ ke_sum, "", mode_color[n], 1.8) for n in 1:nmode],
                 share_ylabel = "share of modes 1-$nmode [%]",
                 top_ymin = 10^floor(log10(minimum(filter(v -> isfinite(v) && v > 0, ke[h[ix_line, :] .>= h_mask, nmode])))))
end
end   # (7)

# ============================================================================
# (8) KE / pressure ratio along the mooring line, per frequency band
# ============================================================================
if 8 in mode_steps
# Rn = KEn / [H <pn²> / (2 ρ0 cn²)]. For one progressive wave without rotation un = pn / (ρ0 cn), so Rn = 1; with rotation
# (cn without rotation, as here) a progressive wave gives (ω² + f²) / (ω² − f²), ≈ 1 at these latitudes (f²/ω² < 0.01 for M2).
# A standing (or partly standing) pattern moves pressure and velocity apart: Rn > 1 near pressure nodes (strong currents,
# weak p), Rn < 1 near pressure antinodes; its average over a wavelength stays ~1.
rho0 = run_rho0()
p8, dt8 = line_series("p", ["p"])
uv8, _ = line_series("uv", ["u", "v"])
c_line = c[ix_line, :, :]   # (eta, mode) [m/s]
for (bname, bkeep, bdesc) in var_bands
    pp, _ = band_msq(p8["p"], bkeep, dt8)
    uu, _ = band_msq(uv8["u"], bkeep, dt8)
    vv, _ = band_msq(uv8["v"], bkeep, dt8)
    ke = 0.5 .* rho0 .* Hline .* (uu .+ vv)              # KEn [J m-2]
    pe_c = Hline .* pp ./ (2 .* rho0 .* c_line .^ 2)     # H <pn²> / (2 ρ0 cn²) [J m-2]
    R = ke ./ pe_c
    deep_l = (h[ix_line, :] .>= h_mask) .& (edge .< (1:ny) .<= ny - edge)
    println("$bname band: median Rn (deep, inside the edges) = ", [round(median(R[deep_l, n]), digits = 2) for n in 1:nmode],
            ", range ", [round.(extrema(R[deep_l, n]), digits = 2) for n in 1:nmode])
    line_section("mode_$(bname)ratio_line.png", "$bdesc: KEₙ / [H⟨pₙ²⟩ / (2ρ₀cₙ²)] along the mooring line (1 = one progressive wave)\n$dff_run, xi = $ix_line";
                 top = [(R[:, n], "mode $n", mode_color[n], 1.8) for n in 1:nmode],
                 top_ylabel = "Rₙ = KEₙ / [H⟨pₙ²⟩ / (2ρ₀cₙ²)]", top_ylims = (0.1, 10), top_hline = 1.0)
end
end   # (8)
