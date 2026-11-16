#= mean_flow_month.jl
Hsin-Yi Chen, USM DMS
Does the mean flow change the internal-tide energy? Mean flow of the analysis window and of each month, averaged
across the same ±200 km strip around the mooring line as mode_dff_binplot.jl, to set beside the strip-mean KE / APE.

Periods (the window of the harmonic fit, mode_bcef_harmonic.jl: 3-day spin-up dropped):
    all  2022-08-27 -> 11-30,   Aug 08-27 -> 08-31 (5 days only),   Sep,   Oct,   Nov
Levels:  surf  the top s-level of u, v (the North Brazil Current is surface intensified)
         bar   ubar, vbar (depth mean: small over the 3-4 km deep slope / basin)
u, v along grid xi / eta (v: along the mooring line, the direction of the M2 flux), moved to the rho points
(u2rho, v2rho); speed = |mean vector| (the speed of the mean flow; the mean of the hourly speed would be mostly tide).

Steps (MF_STEPS, default 1,2; ":" also works in sbatch --export):
  1  his files -> sums per period -> <bcef_dir>/mean_flow_month.nc  (reads every his file of the window:
     ~3 s per file on the 2.7 km grid, the u / v chunks hold all levels; use sbatch_files/julia_meanflow.sh)
  2  figures: one per level and variable (u, v, speed): the 5 period maps in a row, the strip means of the
     5 periods on one line plot below -> MF_FIG (default ~/figure_check)
Usage:
  julia DFF/mean_flow_month.jl                                   # amazon_2700m_cmems_0801 (ddf_bcef_0824)
  DFF_RUN=amazon_900m_3mon_1008_2 MF_STEPS=2 julia DFF/mean_flow_month.jl
  MF_NFILES=8 julia DFF/mean_flow_month.jl                       # quick test on the first 8 his files
=#

using NCDatasets, Statistics, Dates, Printf, CairoMakie
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
haskey(ENV, "DFF_RUN") || (ENV["DFF_RUN"] = "amazon_2700m_cmems_0801")
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")                 # dff_run, run_dir, run_grid_fname, fig_dir
include("/home/hchen54/internalwave_julia/function/uv2rho.jl")             # u2rho, v2rho
include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")       # topdown_axis, plot_curvilinear!, quiver_curvilinear!

# ============================================================================
# (0) settings, paths
# ============================================================================
NMODE = parse(Int, get(ENV, "NMODE", "3"))
default_bcef = Dict("amazon_2700m_cmems_0801" => "ddf_bcef_0824")
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, get(default_bcef, dff_run, "ddf_bcef")))
dff_fname = joinpath(bcef_dir, "bcef_dff_flux_modes$(NMODE).nc")       # only for the DFF crop (same grid as the bin plots)
nfiles_max = parse(Int, get(ENV, "MF_NFILES", "0"))                     # 0: every his file of the window
out_fname = joinpath(bcef_dir, nfiles_max > 0 ? "mean_flow_month_test.nc" : "mean_flow_month.nc")
mf_steps = parse.(Int, split(get(ENV, "MF_STEPS", "1,2"), r"[,:; ]+"; keepempty = false))
t_win = (DateTime(get(ENV, "MF_START", "2022-08-27")), DateTime(get(ENV, "MF_END", "2022-11-30")))   # inclusive
per_names = ["all", "Aug", "Sep", "Oct", "Nov"]
per_month = [0, 8, 9, 10, 11]                                          # 0: the whole window
t_ref = DateTime(1994, 1, 1)
run_mooring = Dict("amazon_2700m_cmems_0801" => "roms_cmems_grd_2700m_mor_edata.nc")
mooring_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/$(get(run_mooring, dff_run, "roms_grd_900m_mor_edata.nc"))"
h_mask = 200.0                                                         # [m] shelf not drawn
edge_km = parse(Float64, get(ENV, "EDGE_KM", "54"))
side_km = parse(Float64, get(ENV, "SIDE_KM", "200"))
println("$dff_run: $bcef_dir; window $(t_win[1]) -> $(t_win[2]); steps $(join(mf_steps, ","))")

# ============================================================================
# (1) his files -> mean u, v (top level) and ubar, vbar per period -> mean_flow_month.nc
# ============================================================================
function read_uv(f; tries = 5)   # top-level u, v and ubar, vbar on the rho points, time [s since 1994]; retries for Lustre
    for t in 1:tries
        try
            return NCDataset(f) do d
                us = u2rho(Float64.(nomissing(d["u"][:, :, end, :], 0.0)))
                vs = v2rho(Float64.(nomissing(d["v"][:, :, end, :], 0.0)))
                ub = u2rho(Float64.(nomissing(d["ubar"][:, :, :], 0.0)))
                vb = v2rho(Float64.(nomissing(d["vbar"][:, :, :], 0.0)))
                tt = [x isa DateTime ? Dates.value(x - t_ref) / 1000 : Float64(x) for x in d["ocean_time"][:]]   # raw seconds or decoded
                us, vs, ub, vb, tt
            end
        catch e
            t == tries && rethrow()
            @warn "read $(basename(f)) failed ($(sprint(showerror, e)[1:min(end, 80)])), retry $t"
            sleep(10t)
        end
    end
end

if 1 in mf_steps
    datadir = joinpath(run_dir, "after_joint")
    file_time(f) = DateTime(match(r"(\d{14})\.nc$", f)[1], DateFormat("yyyymmddHHMMSS"))
    his = sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir)))
    # a file holds the records from its name time to the next file's: keep those that overlap the window
    keep = [file_time(his[k]) <= t_win[2] && (k == length(his) || file_time(his[k + 1]) > t_win[1]) for k in eachindex(his)]
    his = joinpath.(datadir, his[keep])
    nfiles_max > 0 && (his = his[1:min(nfiles_max, end)])
    isempty(his) && error("no his files in $datadir for the window")
    nx0, ny0 = NCDataset(g -> (g.dim["xi_rho"], g.dim["eta_rho"]), run_grid_fname)
    names = ("u_surf", "v_surf", "ubar", "vbar")
    S = Dict(k => zeros(nx0, ny0, length(per_names)) for k in names)   # sums
    nrec = zeros(Int, length(per_names))
    t_first, t_last = DateTime(9999), DateTime(0)
    println("\nstep 1: $(length(his)) his files, $(basename(first(his))) -> $(basename(last(his)))")
    tic = time()
    for (k, f) in enumerate(his)
        us, vs, ub, vb, tt = read_uv(f)
        for r in eachindex(tt)
            tr = t_ref + Millisecond(round(Int, 1000tt[r]))
            t_win[1] <= tr <= t_win[2] || continue
            global t_first, t_last = min(t_first, tr), max(t_last, tr)
            for (p, m) in enumerate(per_month)
                (m == 0 || month(tr) == m) || continue
                nrec[p] += 1
                for (nm, a) in zip(names, (us, vs, ub, vb))
                    S[nm][:, :, p] .+= view(a, :, :, r)
                end
            end
        end
        (k % 25 == 0 || k == length(his)) && @printf("  %d / %d files, %d records (%.0f s)\n", k, length(his), nrec[1], time() - tic)
        k % 25 == 0 && GC.gc()
    end
    any(nrec .== 0) && @warn "periods without records: $(per_names[nrec .== 0])"
    tmp = out_fname * ".tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do o
        defDim(o, "xi_rho", nx0); defDim(o, "eta_rho", ny0); defDim(o, "period", length(per_names))
        o.attrib["title"] = "Mean flow of the analysis window and of each month, $dff_run"
        o.attrib["window"] = "$t_first to $t_last (records inside $(t_win[1]) .. $(t_win[2]))"
        o.attrib["periods"] = join(per_names, ",")
        o.attrib["note"] = "u, v along grid xi / eta on the rho points (u2rho / v2rho); surf = top s-level, bar = ubar / vbar"
        o.attrib["source"] = "$(dirname(first(his))): $(basename(first(his))) to $(basename(last(his)))"
        o.attrib["script"] = "internalwave_julia/DFF/mean_flow_month.jl"
        defVar(o, "nrec", nrec, ("period",); attrib = ["long_name" => "hourly records in the period"])
        for (nm, ln) in zip(names, ("top-level u along grid xi", "top-level v along grid eta",
                                    "depth-mean u along grid xi", "depth-mean v along grid eta"))
            defVar(o, nm, Float32.(S[nm] ./ reshape(ifelse.(nrec .> 0, nrec, NaN), 1, 1, :)), ("xi_rho", "eta_rho", "period");
                   attrib = ["units" => "m s-1", "long_name" => "mean $ln"])
        end
    end
    mv(tmp, out_fname; force = true)
    println("wrote $out_fname; records per period: ", join(["$p $n" for (p, n) in zip(per_names, nrec)], ", "))
end

# ============================================================================
# (2) figures: per level and variable, the 5 period maps and the strip means of the 5 periods
# ============================================================================
if 2 in mf_steps
    isfile(out_fname) || error("no $out_fname: run step 1 first")
    fig_path = expanduser(get(ENV, "MF_FIG", "~/figure_check"))
    mkpath(fig_path)
    # --- the DFF crop, grid, strip, moorings, crop (as mode_dff_binplot.jl)
    nx, ny = NCDataset(d -> (d.dim["xi_rho"], d.dim["eta_rho"]), dff_fname)
    lon, lat, pm, pn, angle, h = NCDataset(run_grid_fname) do g
        [Float64.(g[v][1:nx, 1:ny]) for v in ("lon_rho", "lat_rho", "pm", "pn", "angle", "h")]
    end
    dx_km = mean(1 ./ pm) / 1e3
    edge = round(Int, edge_km / dx_km)
    inside = (edge .< (1:nx) .<= nx - edge) .& (edge .< (1:ny)' .<= ny - edge)
    show = (h .>= h_mask) .& inside
    masked(a) = ifelse.(show, a, NaN)
    lon180 = ifelse.(lon .> 180, lon .- 360, lon)
    moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
    moor_lon180 = ifelse.(moor_lon .> 180, moor_lon .- 360, moor_lon)
    moor_ij = NCDataset(mooring_file) do g
        [(round(Int, g["$(m)_mor"][1] + 1), round(Int, g["$(m)_mor"][2] + 1)) for m in moor_name]
    end
    ix_line = round(Int, 0.5 + (nx + 1) / 2)
    line_km = [0; cumsum(0.5 .* (1 ./ pn[ix_line, 1:end-1] .+ 1 ./ pn[ix_line, 2:end]))] ./ 1e3
    ix_side = round(Int, side_km / dx_km)
    ix_west, ix_east = ix_line - ix_side, ix_line + ix_side
    strip = ix_west:ix_east
    min_cover = 0.5
    j_node = argmin(abs.(line_km .- 200))
    crop_after = get(ENV, "CROP_AFTER", "CPIES7"); crop_km = parse(Float64, get(ENV, "CROP_KM", "250"))
    j_crop = let k = findfirst(==(crop_after), moor_name)
        min(argmin(abs.(line_km .- (line_km[moor_ij[k][2]] + crop_km))), ny - edge)
    end
    moor_in = [(nm, j, abs(i - ix_line) <= 1) for (nm, (i, j)) in zip(moor_name, moor_ij) if abs(i - ix_line) <= ix_side]
    function strip_stats(a)
        m, s = fill(NaN, ny), fill(NaN, ny)
        for j in 1:ny
            v = [a[i, j] for i in strip if show[i, j] && isfinite(a[i, j])]
            length(v) / length(strip) >= min_cover || continue
            m[j], s[j] = mean(v), std(v)
        end
        m, s
    end
    # --- the means
    D = NCDataset(out_fname)
    per = split(D.attrib["periods"], ",")
    nrec = D["nrec"][:]
    rdm(v, p) = Float64.(D[v][1:nx, 1:ny, p])
    println("\nstep 2: $(D.attrib["window"]); records ", join(["$p $n" for (p, n) in zip(per, nrec)], ", "))
    # --- layout
    pt = 1300 / (18.0 / 2.54 * 72)
    map_lims = (extrema(lon180[show]) .+ (-0.1, 0.1), extrema(lat[show]) .+ (-0.1, 0.1))
    map_w = 300
    map_h = round(Int, map_w * (map_lims[2][2] - map_lims[2][1]) / ((map_lims[1][2] - map_lims[1][1]) * cosd(mean(map_lims[2]))))
    per_col = [:black, Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3], Makie.wong_colors()[6]]
    per_lab(p) = per[p] == "all" ? "all ($(nrec[p] ÷ 24) d)" : "$(per[p]) ($(round(nrec[p] / 24, digits = 0) |> Int) d)"
    level_lab = Dict("surf" => "surface (top s-level)", "bar" => "depth mean (ubar, vbar)")
    skip = max(1, round(Int, 45 / dx_km))
    for lev in ("surf", "bar")
        un, vn = lev == "surf" ? ("u_surf", "v_surf") : ("ubar", "vbar")
        U = [rdm(un, p) for p in eachindex(per)]
        V = [rdm(vn, p) for p in eachindex(per)]
        Sp = [hypot.(U[p], V[p]) for p in eachindex(per)]
        for (var, F, cmap, sym, vlab) in (("u", U, :balance, true, "u (grid ξ, across the line)"),
                                         ("v", V, :balance, true, "v (grid η, along the line)"),
                                         ("speed", Sp, :viridis, false, "speed |ū|"))
            vals = filter(isfinite, vcat([vec(masked(F[p])) for p in eachindex(per)]...))
            vmax = round(quantile(abs.(vals), 0.99); sigdigits = 2)
            crange = sym ? (-vmax, vmax) : (0.0, vmax)
            fig = Figure(fontsize = 9pt)
            Label(fig[0, 1:6], "Mean flow $(level_lab[lev]): $vlab, strip mean across ±$(round(Int, side_km)) km of the mooring line, $dff_run";
                  font = :bold, fontsize = 11pt)
            hm = nothing
            for p in eachindex(per)
                ax = topdown_axis(fig[1, p]; title = per_lab(p), backgroundcolor = :gray80, width = map_w, height = map_h,
                                  aspect = nothing, xlabel = "", ylabel = p == 1 ? "Latitude" : "",
                                  yticklabelsvisible = p == 1, xticks = WilkinsonTicks(3), yticks = WilkinsonTicks(4))
                hm = plot_curvilinear!(ax, lon180, lat, masked(F[p]); colormap = cmap, colorrange = crange)
                contour!(ax, lon180, lat, h; levels = [1000, 2000, 3000], color = :gray40, linewidth = 0.5)
                for (ix, col) in ((ix_west, :green), (ix_line, :black), (ix_east, :green))
                    lines!(ax, lon180[ix, :], lat[ix, :]; color = col, linestyle = :dash, linewidth = 1)
                end
                lines!(ax, lon180[strip, j_crop], lat[strip, j_crop]; color = :cyan, linestyle = :dash, linewidth = 1.5)
                if var == "speed" && nrec[p] > 0   # arrows of the mean flow, grid -> east / north
                    Ue = U[p] .* cos.(angle) .- V[p] .* sin.(angle); Vn = U[p] .* sin.(angle) .+ V[p] .* cos.(angle)
                    quiver_curvilinear!(ax, lon180, lat, masked(Ue), masked(Vn); skip = skip, lengthscale = 0.8 / vmax,
                                        color = (:white, 0.8), shaftwidth = 0.6, tipwidth = 3, tiplength = 3)
                end
                scatter!(ax, moor_lon180, moor_lat; marker = :circle, markersize = 4, color = :yellow, strokewidth = 0.4)
                scatter!(ax, [lon180[ix_line, j_node]], [lat[ix_line, j_node]]; marker = :star5, markersize = 11,
                         color = :red, strokecolor = :black, strokewidth = 0.5)
                limits!(ax, map_lims...)
            end
            Colorbar(fig[1, 6], hm; height = map_h, label = "[m s⁻¹]")
            # strip means of the 5 periods
            al = Axis(fig[2, 1:5]; title = "$vlab averaged across the strip (±$(round(Int, side_km)) km, ξ = $(ix_west)–$(ix_east))",
                      xlabel = "distance along the mooring line from the south boundary [km]", ylabel = "[m s⁻¹]",
                      xticks = 0:100:ceil(Int, line_km[end]), xminorticks = IntervalsBetween(2), xminorticksvisible = true,
                      xminorgridvisible = true, height = 320, width = 5map_w + 4 * 12)
            ms = [strip_stats(F[p]) for p in eachindex(per)]
            local ok = isfinite.(ms[1][1]) .& ((1:ny) .<= j_crop)
            yv = filter(isfinite, vcat([ms[p][1][ok] for p in eachindex(per)]..., (ms[1][1] .+ ms[1][2])[ok], (ms[1][1] .- ms[1][2])[ok]))
            ylo, yhi = extrema(vcat(yv, 0.0)); ytop = yhi + 0.3 * (yhi - ylo)
            ylims!(al, ylo - 0.05 * (yhi - ylo), ytop)
            hlines!(al, [0]; color = :black, linewidth = 0.6)
            for (nm, j, on) in moor_in
                vlines!(al, [line_km[j]]; color = on ? :gray60 : :gray85, linewidth = on ? 0.8 : 0.5)
                on && text!(al, line_km[j], ytop; text = nm * " ", rotation = π / 2, align = (:right, :center), fontsize = 7pt, color = :gray35)
            end
            vlines!(al, [line_km[j_node]]; color = :red, linestyle = :dash, linewidth = 1.2)
            x = line_km[ok]
            band!(al, x, (ms[1][1] .- ms[1][2])[ok], (ms[1][1] .+ ms[1][2])[ok]; color = (:gray, 0.25), label = "all ± 1 std (W–E)")
            for p in eachindex(per)
                lines!(al, x, ms[p][1][ok]; color = per_col[p], linewidth = p == 1 ? 2.5 : 1.5, label = per_lab(p))
            end
            xlims!(al, line_km[findfirst(ok)], line_km[j_crop])
            Legend(fig[2, 6], al; framevisible = false, labelsize = 8pt)
            Label(fig[3, 1:6], "u, v along grid ξ / η (v along the mooring line, + toward the north end); speed = |mean vector|; strip: shelf (h < $(Int(h_mask)) m), " *
                  "$(round(Int, edge_km)) km edge bands and rows with < $(round(Int, 100min_cover))% deep cover left out;\n" *
                  "green: strip sides; cyan: north end of the line plot ($(round(Int, crop_km)) km past $crop_after); red star / dashed: 200 km node; " *
                  "grey: moorings (labelled: on the line); Aug only from $(Date(t_win[1])) (after the spin-up)";
                  fontsize = 8pt, tellwidth = false, justification = :left)
            colgap!(fig.layout, 12)
            resize_to_layout!(fig)
            fname = joinpath(fig_path, "mean_flow_$(lev)_$(var).png")
            save(fname, fig; px_per_unit = 2)
            println("saved $fname")
            # numbers for the reply: strip mean at the node and over the strip up to the crop
            println("  $lev $var: ", join([@sprintf("%s node %.3f, mean %.3f", per[p], ms[p][1][j_node], mean(filter(isfinite, ms[p][1][ok])))
                                           for p in eachindex(per)], "; "))
            GC.gc()
        end
    end
    close(D)
end
