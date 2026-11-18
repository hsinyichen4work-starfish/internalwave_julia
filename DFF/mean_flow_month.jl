#= mean_flow_month.jl
Hsin-Yi Chen, USM DMS
Does the mean flow change the internal-tide energy? Mean flow of the analysis window and of each month, averaged
across the same ±200 km strip around the mooring line as mode_dff_binplot.jl, to set beside the strip-mean KE / APE.

Periods (the window of the harmonic fit, mode_bcef_harmonic.jl: 3-day spin-up dropped):
    all  2022-08-27 -> 11-30,   Aug 08-27 -> 08-31 (5 days only),   Sep,   Oct,   Nov
Levels:  surf  the top s-level of u, v (the North Brazil Current is surface intensified)
         bar   ubar, vbar (depth mean: small over the 3-4 km deep slope / basin)
         m1, m2  U_n = (1/H) ∫ U φ_n² dz: the mean flow weighted by mode n's energy profile (the flow mode n is
               advected / Doppler shifted by; φ_n of bcef_modes<N>_structure.nc, (1/H) ∫ φ_n² dz = 1)
u, v along grid xi / eta (v: along the mooring line, the direction of the M2 flux), moved to the rho points
(u2rho, v2rho); speed = |mean vector| (the speed of the mean flow; the mean of the hourly speed would be mostly tide).

Steps (MF_STEPS, default 1,2; ":" also works in sbatch --export):
  1  his files -> sums per period -> <bcef_dir>/mean_flow_month.nc  (reads every his file of the window:
     ~3 s per file on the 2.7 km grid, the u / v chunks hold all levels; use sbatch_files/julia_meanflow.sh)
  2  figures: one per level and variable (u, v, speed): the 5 period maps in a row, the strip means of the
     5 periods on one line plot below -> MF_FIG (default figure/<run>/mean_flow)
  3  speed maps (cmocean speed, arrows): rows U_1, U_2, ubar / vbar; columns Aug, Sep, Oct, Nov, all
     -> MF_FIG/mean_flow_speed_Un_ubar.png
  4  mode speed c_n (map, strip mean) and |U_n| / c_n by period (map, strip means of the 5 periods, 2 x 5 maps)
     -> MF_FIG/mean_flow_c_n.png, mean_flow_U_over_c_months.png, mean_flow_U_over_c_heatmap_month.png
  5  (not in the default steps) monthly stratification: mean density and zeta of each period from the his files ->
     N² and the mode speeds c_n of each period (the eigen problem of bcef_modes_puv.jl) -> <bcef_dir>/mean_flow_N2_month.nc
     (reads rho at every level of every file: batch job, MF_STEPS=5:6 sbatch_files/julia_meanflow.sh)
  6  (not in the default steps) figures of step 5: N² sections across the strip by period (log10 N², and the ratio to
     the whole window), strip-mean c_n of each period and |U_n| / c_n with each period's own c_n
  7  2-D stratification of each period from step 5 (login node): mixed layer depth, pycnocline depth, N_max, WKB mode-1
     speed (1/π)∫N dz -> <bcef_dir>/mean_flow_strat_month.nc and maps (value, change against all) + strip-mean lines
Usage:
  julia DFF/mean_flow_month.jl                                   # amazon_2700m_cmems_0801 (ddf_bcef_0824)
  DFF_RUN=amazon_900m_3mon_1008_2 MF_STEPS=2 julia DFF/mean_flow_month.jl
  MF_NFILES=8 julia DFF/mean_flow_month.jl                       # quick test on the first 8 his files
=#

using NCDatasets, Statistics, Dates, Printf, CairoMakie, LinearAlgebra
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
mf_steps = parse.(Int, split(get(ENV, "MF_STEPS", "1,2,3,4"), r"[,:; ]+"; keepempty = false))
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
function read_uv(f; tries = 5)   # u, v (all levels), ubar, vbar on the rho points, time [s since 1994]; retries for Lustre
    for t in 1:tries
        try
            return NCDataset(f) do d
                us = u2rho(Float32.(nomissing(d["u"][:, :, :, :], 0f0)))      # (xi, eta, s, time); the chunks hold all levels
                vs = v2rho(Float32.(nomissing(d["v"][:, :, :, :], 0f0)))
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
    names = ("u_surf", "v_surf", "ubar", "vbar", "u_m1", "v_m1", "u_m2", "v_m2")
    S = Dict(k => zeros(nx0, ny0, length(per_names)) for k in names)   # sums
    # weights of U_n = (1/H) Σ_k U_k φ_n,k² Hz_k: Hz from the mean rho-level depths (w levels half way between, -h at the
    # bottom, mean zeta at the top), as the modal decomposition
    struct_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc")
    Wn = NCDataset(struct_fname) do g      # Float32, one mode at a time (login node: 8 GB virtual memory)
        hh, zm, zr = Float32.(g["h"][:, :]), Float32.(g["zeta_mean"][:, :]), Float32.(g["z_r_mean"][:, :, :])
        zw = cat(reshape(-hh, size(hh)..., 1), 0.5f0 .* (zr[:, :, 1:end-1] .+ zr[:, :, 2:end]), reshape(zm, size(zm)..., 1); dims = 3)
        Hz = diff(zw; dims = 3); zr = zw = nothing
        [(φ = Float32.(g["phi"][:, :, :, n]); φ .^ 2 .* Hz ./ (hh .+ zm)) for n in 1:2]
    end
    GC.gc()
    h_s = NCDataset(g -> Float64.(g["h"][:, :]), struct_fname)
    ws = [vec(sum(Float64.(W); dims = 3))[vec(h_s .> 200)] for W in Wn]
    @printf("  U_n weights φ_n² Hz / H: sum over the column (h > 200 m) mode 1 %.4f-%.4f, mode 2 %.4f-%.4f (should be 1)\n",
            extrema(ws[1])..., extrema(ws[2])...)
    # Σ_k a[:, :, k, r] W[:, :, k] added to acc (typed kernel)
    function wsum!(acc::AbstractMatrix{Float64}, a::AbstractArray{<:Real, 4}, W::Array{Float32, 3}, r::Int)   # u2rho returns Float64
        @inbounds for k in axes(a, 3), j in axes(a, 2), i in axes(a, 1)
            acc[i, j] += Float64(a[i, j, k, r] * W[i, j, k])
        end
        acc
    end
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
                S["u_surf"][:, :, p] .+= view(us, :, :, size(us, 3), r)
                S["v_surf"][:, :, p] .+= view(vs, :, :, size(vs, 3), r)
                S["ubar"][:, :, p] .+= view(ub, :, :, r)
                S["vbar"][:, :, p] .+= view(vb, :, :, r)
                for n in 1:2
                    wsum!(view(S["u_m$n"], :, :, p), us, Wn[n], r)
                    wsum!(view(S["v_m$n"], :, :, p), vs, Wn[n], r)
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
        o.attrib["note"] = "u, v along grid xi / eta on the rho points (u2rho / v2rho); surf = top s-level, bar = ubar / vbar; " *
                           "m<n> = (1/H) int U phi_n^2 dz (phi_n, Hz: $(basename(struct_fname)))"
        o.attrib["source"] = "$(dirname(first(his))): $(basename(first(his))) to $(basename(last(his)))"
        o.attrib["script"] = "internalwave_julia/DFF/mean_flow_month.jl"
        defVar(o, "nrec", nrec, ("period",); attrib = ["long_name" => "hourly records in the period"])
        for (nm, ln) in zip(names, ("top-level u along grid xi", "top-level v along grid eta",
                                    "depth-mean u along grid xi", "depth-mean v along grid eta",
                                    "mode-1 weighted u (1/H) int u phi_1^2 dz", "mode-1 weighted v", "mode-2 weighted u", "mode-2 weighted v"))
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
    fig_path = expanduser(get(ENV, "MF_FIG", joinpath(dirname(fig_dir), "mean_flow")))
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
    level_lab = Dict("surf" => "surface (top s-level)", "bar" => "depth mean (ubar, vbar)",
                     "m1" => "mode-1 weighted U₁ = (1/H)∫U φ₁² dz", "m2" => "mode-2 weighted U₂ = (1/H)∫U φ₂² dz")
    level_vars = Dict("surf" => ("u_surf", "v_surf"), "bar" => ("ubar", "vbar"), "m1" => ("u_m1", "v_m1"), "m2" => ("u_m2", "v_m2"))
    # levels of step 2: MF_LEVELS (default surf:bar:m1:m2; m1 / m2 need the step-1 file with U_n)
    levels2 = String.(split(get(ENV, "MF_LEVELS", "surf:bar:m1:m2"), r"[,:; ]+"; keepempty = false))
    skip = max(1, round(Int, 45 / dx_km))
    for lev in levels2
        un, vn = level_vars[lev]
        haskey(D, un) || (println("  $lev: no $un in $out_fname (rerun step 1)"); continue)
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

# ============================================================================
# (5) monthly stratification: N² and c_n of each period (not in the default steps; batch job)
# ============================================================================
# Mean density rho (his "rho", every level) and zeta of each period (all, Aug, Sep, Oct, Nov; the window of step 1), then
# for every column the problem of bcef_modes_puv.jl (W'' + N²/c² W = 0, rigid lid; N² = -g/rho0 ∂rho/∂z ≥ 1e-7 between
# the mean level depths of the period): N² on the interior interfaces and c_n. -> <bcef_dir>/mean_flow_N2_month.nc:
# N2 (xi, eta, s_w_interior, period), c (xi, eta, mode, period), zeta_mean (xi, eta, period). Memory ~4 GB (sums of rho in
# Float64): batch job. The c_n of "all" should match bcef_modes<N>_structure.nc (printed check).
n2_fname = joinpath(bcef_dir, nfiles_max > 0 ? "mean_flow_N2_month_test.nc" : "mean_flow_N2_month.nc")
# the column problem of bcef_modes_puv.jl (modes): N² [s-2] on the interior interfaces and the nmode speeds
function modes_c(zr, zw, rho, rho0; nmode = 2, N2min = 1e-7)
    nz = length(zr)
    Hz = diff(zw); dzc = diff(zr)
    N2 = max.(-9.81 / rho0 .* diff(rho) ./ dzc, N2min)
    B = N2 .* dzc
    d = (1 ./ Hz[2:nz] .+ 1 ./ Hz[1:nz-1]) ./ B
    e = (-1 ./ Hz[2:nz-1]) ./ sqrt.(B[1:end-1] .* B[2:end])
    F = eigen(SymTridiagonal(d, e), 1:nmode)
    1 ./ sqrt.(F.values), N2
end
function acc_rho!(S::Array{Float64, 4}, rho::Array{Float32, 4}, r::Int, p::Int)
    @inbounds for k in axes(rho, 3), j in axes(rho, 2), i in axes(rho, 1)
        S[i, j, k, p] += rho[i, j, k, r]
    end
    S
end
if 5 in mf_steps
    datadir5 = joinpath(run_dir, "after_joint")
    ftime5(f) = DateTime(match(r"(\d{14})\.nc$", f)[1], DateFormat("yyyymmddHHMMSS"))
    his5 = sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir5)))
    keep5 = [ftime5(his5[k]) <= t_win[2] && (k == length(his5) || ftime5(his5[k + 1]) > t_win[1]) for k in eachindex(his5)]
    his5 = joinpath.(datadir5, his5[keep5]); nfiles_max > 0 && (his5 = his5[1:min(nfiles_max, end)])
    h5 = NCDataset(g -> Float64.(g["h"][:, :]), run_grid_fname); nx5, ny5 = size(h5)
    nz5, hc5, Cs_r5, Cs_w5, rho05 = NCDataset(his5[1]) do d
        d.dim["s_rho"], d.attrib["hc"], Float64.(d.attrib["Cs_r"]), Float64.(d.attrib["Cs_w"]), Float64(d.attrib["rho0"])
    end
    sc_r5 = ((1:nz5) .- nz5 .- 0.5) ./ nz5; sc_w5 = ((0:nz5) .- nz5) ./ nz5
    z0r5 = (hc5 .* reshape(sc_r5, 1, 1, :) .+ reshape(Cs_r5, 1, 1, :) .* h5) ./ (h5 .+ hc5)
    z0w5 = (hc5 .* reshape(sc_w5, 1, 1, :) .+ reshape(Cs_w5, 1, 1, :) .* h5) ./ (h5 .+ hc5)
    zlev5(z0, zeta) = zeta .+ (zeta .+ h5) .* z0
    Srho = zeros(nx5, ny5, nz5, length(per_names)); Szeta = zeros(nx5, ny5, length(per_names)); n5 = zeros(Int, length(per_names))
    println("\nstep 5: mean rho and zeta of each period from $(length(his5)) his files ($nz5 levels, rho0 = $rho05)")
    tic = time()
    for (k, f) in enumerate(his5)
        rho, zt, tt = NCDataset(f) do d
            Float32.(nomissing(d["rho"].var[:, :, :, :], 0f0)), Float64.(d["zeta"].var[:, :, :]),
            [x isa DateTime ? Dates.value(x - t_ref) / 1000 : Float64(x) for x in d["ocean_time"][:]]
        end
        for r in eachindex(tt)
            tr = t_ref + Millisecond(round(Int, 1000tt[r]))
            t_win[1] <= tr <= t_win[2] || continue
            for (p, m) in enumerate(per_month)
                (m == 0 || month(tr) == m) || continue
                n5[p] += 1
                acc_rho!(Srho, rho, r, p)
                Szeta[:, :, p] .+= view(zt, :, :, r)
            end
        end
        (k % 25 == 0 || k == length(his5)) && @printf("  %d / %d files, %d records (%.0f s)\n", k, length(his5), n5[1], time() - tic)
        k % 10 == 0 && GC.gc()
    end
    NP = length(per_names)
    N2o = zeros(Float32, nx5, ny5, nz5 - 1, NP); co = zeros(Float32, nx5, ny5, 2, NP); zo = zeros(Float32, nx5, ny5, NP)
    function all_cols!(N2o, co, zr, zw, rm, p, rho0)
        for j in axes(zr, 2), i in axes(zr, 1)
            c, n2 = modes_c(zr[i, j, :], zw[i, j, :], rm[i, j, :], rho0)
            co[i, j, :, p] .= c; N2o[i, j, :, p] .= n2
        end
    end
    for p in 1:NP
        n5[p] > 0 || continue
        t0 = time()
        rm = Srho[:, :, :, p] ./ n5[p]; zm = Szeta[:, :, p] ./ n5[p]
        all_cols!(N2o, co, zlev5(z0r5, zm), zlev5(z0w5, zm), rm, p, rho05)
        zo[:, :, p] .= zm
        @printf("  %-3s (%d records): modes of every column %.0f s\n", per_names[p], n5[p], time() - t0)
    end
    # check: c_n of "all" against the structure file of the modal decomposition (same window and method)
    let s5 = joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc")
        if isfile(s5)
            cs = NCDataset(g -> Float64.(g["c"][:, :, 1:2]), s5); deep = h5 .> 1000
            for n in 1:2
                rel = abs.(co[:, :, n, 1] .- cs[:, :, n]) ./ cs[:, :, n]
                @printf("  check c_%d (all) vs %s: median rel. diff %.2e, 99th pct %.2e (deep points)\n", n, basename(s5), median(rel[deep]), quantile(rel[deep], 0.99))
            end
        end
    end
    tmp = n2_fname * ".tmp"; isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do o
        defDim(o, "xi_rho", nx5); defDim(o, "eta_rho", ny5); defDim(o, "s_w_interior", nz5 - 1); defDim(o, "mode", 2); defDim(o, "period", NP)
        o.attrib["title"] = "Stratification and mode speeds of each period, $dff_run"
        o.attrib["periods"] = join(per_names, ","); o.attrib["rho0"] = rho05
        o.attrib["method"] = "mean rho, zeta of the period; N2 = -g/rho0 d(rho)/dz (>= 1e-7) between the mean level depths; c_n: W'' + N2/c2 W = 0, rigid lid (bcef_modes_puv.jl)"
        o.attrib["script"] = "internalwave_julia/DFF/mean_flow_month.jl (step 5)"
        defVar(o, "nrec", n5, ("period",))
        defVar(o, "zeta_mean", zo, ("xi_rho", "eta_rho", "period"); attrib = ["units" => "m"])
        defVar(o, "N2", N2o, ("xi_rho", "eta_rho", "s_w_interior", "period"); attrib = ["units" => "s-2", "long_name" => "N² on the interior interfaces 2..nz (k = 1 bottom)"])
        defVar(o, "c", co, ("xi_rho", "eta_rho", "mode", "period"); attrib = ["units" => "m s-1", "long_name" => "mode eigenspeeds"])
    end
    mv(tmp, n2_fname; force = true)
    println("wrote $n2_fname; records per period: ", join(["$p $n" for (p, n) in zip(per_names, n5)], ", "))
    Srho = nothing; GC.gc()
end

# ============================================================================
# (3) speed of the mean flow, U_1, U_2 and ubar / vbar, by period: maps with arrows
# ============================================================================
# Rows: U_1, U_2 (mean flow weighted by the mode's energy profile φ_n²) and ubar / vbar (depth mean); columns: Aug (5 days),
# Sep, Oct, Nov, all. Colour: speed |U| (cmocean speed), one range per row; arrows: the mean flow east / north (one length
# scale per row). Moorings yellow, ±200 km strip green, 200 km node red. -> MF_FIG/mean_flow_speed_Un_ubar.png
if any(in(mf_steps), (3, 4, 6, 7))                                   # map / strip setup shared by steps 3, 4, 6 and 7
    isfile(out_fname) || error("no $out_fname: run step 1 first")
    fig_path = expanduser(get(ENV, "MF_FIG", joinpath(dirname(fig_dir), "mean_flow"))); mkpath(fig_path)
    nx3, ny3 = NCDataset(d -> (d.dim["xi_rho"], d.dim["eta_rho"]), dff_fname)
    lon3, lat3, pm3, pn3, ang3, h3, msk3 = NCDataset(run_grid_fname) do g
        [Float64.(g[v][1:nx3, 1:ny3]) for v in ("lon_rho", "lat_rho", "pm", "pn", "angle", "h", "mask_rho")]
    end
    dx3 = mean(1 ./ pm3) / 1e3
    e3 = round(Int, edge_km / dx3)
    show3 = (msk3 .> 0) .& (h3 .>= h_mask) .& (e3 .< (1:nx3) .<= nx3 - e3) .& (e3 .< (1:ny3)' .<= ny3 - e3)
    mk3(a) = ifelse.(show3, a, NaN)
    lon3 = ifelse.(lon3 .> 180, lon3 .- 360, lon3)
    mname3, mlon3, mlat3 = read_moorings(mooring_file); mlon3 = ifelse.(mlon3 .> 180, mlon3 .- 360, mlon3)
    ixl = round(Int, 0.5 + (nx3 + 1) / 2); ixs = round(Int, side_km / dx3)
    lkm = [0; cumsum(0.5 .* (1 ./ pn3[ixl, 1:end-1] .+ 1 ./ pn3[ixl, 2:end]))] ./ 1e3
    jn = argmin(abs.(lkm .- 200))
    lims3 = (extrema(lon3[show3]) .+ (-0.1, 0.1), extrema(lat3[show3]) .+ (-0.1, 0.1))
    w3 = 260; h3p = round(Int, w3 * (lims3[2][2] - lims3[2][1]) / ((lims3[1][2] - lims3[1][1]) * cosd(mean(lims3[2]))))
    D3 = NCDataset(out_fname)
    haskey(D3, "u_m1") || error("$out_fname has no u_m1: rerun step 1 (U_n added)")
    per3 = split(D3.attrib["periods"], ","); nrec3 = D3["nrec"][:]
    pord = [findfirst(==(p), per3) for p in ("Aug", "Sep", "Oct", "Nov", "all")]
    ptit(p) = per3[p] == "all" ? "all ($(nrec3[p] ÷ 24) d)" : per3[p] == "Aug" ? "Aug ($(nrec3[p] ÷ 24) d)" : per3[p]
    rows_all = [("m1", "U₁ = (1/H)∫U φ₁² dz"), ("m2", "U₂ = (1/H)∫U φ₂² dz"), ("bar", "ubar, vbar (depth mean)")]
    # rows of the step-3 figure: MF_SPEED_ROWS (default m1:m2:bar -> mean_flow_speed_Un_ubar.png; m1:m2 -> mean_flow_speed_Un.png)
    row_sel = String.(split(get(ENV, "MF_SPEED_ROWS", "m1:m2:bar"), r"[,:; ]+"; keepempty = false))
    rows = [r for r in rows_all if r[1] in row_sel]
    rdu(lv, c, p) = Float64.(D3[lv == "bar" ? "$(c)bar" : "$(c)_$lv"][1:nx3, 1:ny3, p])
    qsk = max(1, round(Int, 45 / dx3))
    pt = 1300 / (18.0 / 2.54 * 72)                                  # pixels per point (as step 2), for steps 3, 4, 6 run alone
    mij3 = NCDataset(g -> [(round(Int, g["$(m)_mor"][1] + 1), round(Int, g["$(m)_mor"][2] + 1)) for m in mname3], mooring_file)
    jcrop3 = min(argmin(abs.(lkm .- (lkm[mij3[findfirst(==("CPIES7"), mname3)][2]] + 250))), ny3 - e3)
    strip3 = (ixl - ixs):(ixl + ixs)
    moor_in3 = [(nm, j) for (nm, (i, j)) in zip(mname3, mij3) if abs(i - ixl) <= 1]
    function strip_stats3(a)                                         # per eta row over the strip (>= 50 % deep cover)
        m, sd = fill(NaN, ny3), fill(NaN, ny3)
        for j in 1:ny3
            v = [a[i, j] for i in strip3 if show3[i, j] && isfinite(a[i, j])]
            length(v) / length(strip3) >= 0.5 || continue
            m[j], sd[j] = mean(v), std(v)
        end
        m, sd
    end
    rows_ok3(m) = isfinite.(m) .& ((1:ny3) .<= jcrop3)
    function deco_map3!(ax)
        contour!(ax, lon3, lat3, h3; levels = [1000, 2000, 3000], color = :gray45, linewidth = 0.5)
        for ix in (first(strip3), last(strip3))
            lines!(ax, lon3[ix, :], lat3[ix, :]; color = :green, linestyle = :dash, linewidth = 0.9)
        end
        lines!(ax, lon3[ixl, :], lat3[ixl, :]; color = :black, linestyle = :dash, linewidth = 0.8)
        lines!(ax, lon3[strip3, jcrop3], lat3[strip3, jcrop3]; color = :cyan, linestyle = :dash, linewidth = 1.5)
        scatter!(ax, mlon3, mlat3; marker = :circle, markersize = 4, color = :yellow, strokecolor = :black, strokewidth = 0.4)
        scatter!(ax, [lon3[ixl, jn]], [lat3[ixl, jn]]; marker = :star5, markersize = 10, color = :red, strokecolor = :black, strokewidth = 0.5)
        limits!(ax, lims3...)
    end
    function deco_line3!(al, ytop)
        hlines!(al, [0]; color = :black, linewidth = 0.6)
        for (nm, j) in moor_in3
            vlines!(al, [lkm[j]]; color = :gray60, linewidth = 0.8)
            text!(al, lkm[j], ytop; text = nm * " ", rotation = π / 2, align = (:right, :center), fontsize = 7pt, color = :gray35)
        end
        vlines!(al, [lkm[jn]]; color = :red, linestyle = :dash, linewidth = 1.2)
    end
  if 3 in mf_steps
    fig = Figure(fontsize = 10pt)
    println("\nstep 3: speed of U_1, U_2, ubar by period -> $fig_path")
    for (r, (lv, lab)) in enumerate(rows)
        U = [rdu(lv, "u", p) for p in eachindex(per3)]; V = [rdu(lv, "v", p) for p in eachindex(per3)]
        Sp = [hypot.(U[p], V[p]) for p in eachindex(per3)]
        lim = round(quantile(filter(isfinite, vcat([vec(mk3(x)) for x in Sp]...)), 0.99); sigdigits = 2)
        hm = nothing
        for (c, p) in enumerate(pord)
            ax = topdown_axis(fig[r, c]; title = r == 1 ? ptit(p) : "", backgroundcolor = :gray80, width = w3, height = h3p,
                              aspect = nothing, xlabel = r == length(rows) ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "",
                              yticklabelsvisible = c == 1, xticklabelsvisible = r == length(rows), xticks = WilkinsonTicks(3))
            hm = plot_curvilinear!(ax, lon3, lat3, mk3(Sp[p]); colormap = :speed, colorrange = (0, lim))
            contour!(ax, lon3, lat3, h3; levels = [1000, 2000, 3000], color = :gray45, linewidth = 0.5)
            Ue = U[p] .* cos.(ang3) .- V[p] .* sin.(ang3); Vn = U[p] .* sin.(ang3) .+ V[p] .* cos.(ang3)
            quiver_curvilinear!(ax, lon3, lat3, mk3(Ue), mk3(Vn); skip = qsk, lengthscale = 0.7 / lim, color = (:black, 0.7),
                                shaftwidth = 0.5, tipwidth = 3, tiplength = 3)
            for ix in (ixl - ixs, ixl + ixs)
                lines!(ax, lon3[ix, :], lat3[ix, :]; color = :green, linestyle = :dash, linewidth = 0.9)
            end
            lines!(ax, lon3[ixl, :], lat3[ixl, :]; color = :black, linestyle = :dash, linewidth = 0.8)
            scatter!(ax, mlon3, mlat3; marker = :circle, markersize = 4, color = :yellow, strokecolor = :black, strokewidth = 0.4)
            scatter!(ax, [lon3[ixl, jn]], [lat3[ixl, jn]]; marker = :star5, markersize = 10, color = :red, strokecolor = :black, strokewidth = 0.5)
            limits!(ax, lims3...)
            sm = mean(filter(isfinite, mk3(Sp[p])[ixl-ixs:ixl+ixs, :]))
            @printf("  %-4s %-4s speed: strip mean %.3f m/s, at the node %.3f m/s\n", lv, per3[p], sm, Sp[p][ixl, jn])
        end
        Colorbar(fig[r, 6], hm; height = h3p, label = "speed [m s⁻¹]")
        Label(fig[r, 0], lab; rotation = π / 2, font = :bold, fontsize = 10pt, tellheight = false)   # row label
    end
    Label(fig[0, :], "Mean-flow speed by period: mode-weighted U₁, U₂" * ("bar" in row_sel ? " and the depth mean" : "") * "\n$dff_run"; fontsize = 11pt, font = :bold)
    Label(fig[length(rows) + 1, :], "U_n = (1/H)∫U φ_n² dz (φ_n: $(basename(joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc"))), (1/H)∫φ_n²dz = 1): the flow mode n feels; " *
          "colour range per row; arrows: mean flow east / north;\nAug: $(nrec3[findfirst(==("Aug"), per3)] ÷ 24) days only (from 08-27); " *
          "green: ±$(round(Int, side_km)) km strip; red star: 200 km node; shelf (h < $(Int(h_mask)) m) and $(round(Int, edge_km)) km edge bands grey"; fontsize = 8pt)
    colgap!(fig.layout, 8); rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    fname = joinpath(fig_path, "bar" in row_sel ? "mean_flow_speed_Un_ubar.png" : "mean_flow_speed_Un.png"); save(fname, fig; px_per_unit = 2)
    println("saved $fname")
  end

  # ==========================================================================
  # (4) mode speed c_n and the ratio |U_n| / c_n by period
  # ==========================================================================
  # c_n: eigenspeed of mode n (bcef_modes<N>_structure.nc, from the time-mean stratification of the whole record: the
  # same for every period; monthly c_n would need monthly N²). U_n = (1/H)∫U φ_n² dz of step 1 (the flow mode n feels),
  # |U_n| / c_n: modal Froude number (≪ 1: weak Doppler shift / refraction; → 1: blocking). Dashed on the line plot:
  # the projection of the mean flow on mode n (the harmonic fit's mean term u<n>_mean, v<n>_mean; full window), as
  # mode_dff_binplot.jl step 13. Figures -> MF_FIG:
  #   mean_flow_c_n.png                   per mode: map of c_n and its strip mean ± 1 std
  #   mean_flow_U_over_c_months.png       per mode: map of |U_n|/c_n (all) with U_n arrows, strip means of the 5 periods
  #   mean_flow_U_over_c_heatmap_month.png  maps of |U_n|/c_n, rows modes, columns Aug / Sep / Oct / Nov / all
  if 4 in mf_steps
    struct4 = joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc")
    harm4 = joinpath(bcef_dir, "bcef_modes$(NMODE)_harmonic.nc")
    cn4 = NCDataset(g -> Float64.(g["c"][1:nx3, 1:ny3, 1:2]), struct4)
    println("\nstep 4: c_n ($struct4) and |U_n|/c_n by period -> $fig_path")
    w4 = 400; h4 = round(Int, w4 * h3p / w3)
    pcol = [:black, Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3], Makie.wong_colors()[6]]
    pline = [findfirst(==(p), per3) for p in ("all", "Aug", "Sep", "Oct", "Nov")]
    lineax(fig, pos, ttl, ylab, last) = Axis(fig[pos...]; title = ttl, ylabel = ylab,
        xlabel = last ? "distance along the mooring line from the south boundary [km]" : "",
        xticks = 0:100:ceil(Int, lkm[end]), xminorticks = IntervalsBetween(2), xminorticksvisible = true,
        xminorgridvisible = true, width = 2w4 + 60, height = h4)
    # (a) c_n
    fig = Figure(fontsize = 10pt)
    for n in 1:2
        c = cn4[:, :, n]
        ax = topdown_axis(fig[n, 1]; title = "c_$n, mode $n", backgroundcolor = :gray80, width = w4, height = h4, aspect = nothing,
                          xlabel = n == 2 ? "Longitude" : "")
        hm = plot_curvilinear!(ax, lon3, lat3, mk3(c); colormap = :haline, colorrange = extrema(filter(isfinite, mk3(c))))
        deco_map3!(ax)
        Colorbar(fig[n, 2], hm; height = h4, label = "c_$n [m s⁻¹]")
        cm, cs = strip_stats3(c); ok = rows_ok3(cm)
        al = lineax(fig, (n, 3), "c_$n averaged across the strip (±$(round(Int, side_km)) km)", "c_$n [m s⁻¹]", n == 2)
        ytop = 1.25 * maximum((cm .+ cs)[ok]); ylims!(al, 0, ytop); deco_line3!(al, ytop)
        band!(al, lkm, ifelse.(ok, cm .- cs, NaN), ifelse.(ok, cm .+ cs, NaN); color = (:teal, 0.18))
        lines!(al, lkm, ifelse.(ok, cm, NaN); color = :teal, linewidth = 2.5)
        xlims!(al, lkm[findfirst(ok)], lkm[jcrop3])
        @printf("  c_%d: strip mean %.2f m/s (%.2f-%.2f), at the node %.2f m/s\n", n, mean(cm[ok]), extrema(cm[ok])..., cm[jn])
    end
    Label(fig[0, :], "Mode speed c_n (eigenspeed of the time-mean stratification, the same for every period)\n$dff_run"; fontsize = 11pt, font = :bold)
    Label(fig[3, :], "strip mean ± 1 std across the strip; green: ±$(round(Int, side_km)) km strip; cyan: north end of the line plots; red star / dashed: 200 km node; grey: moorings on the line"; fontsize = 8pt)
    resize_to_layout!(fig)
    fname = joinpath(fig_path, "mean_flow_c_n.png"); save(fname, fig; px_per_unit = 2); println("saved $fname")
    # |U_n| / c_n per period
    R = Dict((n, p) => hypot.(rdu("m$n", "u", p), rdu("m$n", "v", p)) ./ cn4[:, :, n] for n in 1:2, p in eachindex(per3))
    Rproj = NCDataset(harm4) do g
        Dict(n => hypot.(Float64.(nomissing(g["u$(n)_mean"][1:nx3, 1:ny3], NaN)), Float64.(nomissing(g["v$(n)_mean"][1:nx3, 1:ny3], NaN))) ./ cn4[:, :, n] for n in 1:2)
    end
    # (b) map (all) + strip means of the periods
    fig = Figure(fontsize = 10pt)
    local al
    for n in 1:2
        pa = findfirst(==("all"), per3)
        lim = round(quantile(filter(isfinite, mk3(R[(n, pa)])), 0.99); sigdigits = 2)
        ax = topdown_axis(fig[n, 1]; title = "|U_$n| / c_$n, mode $n, $(ptit(pa))", backgroundcolor = :gray80, width = w4, height = h4,
                          aspect = nothing, xlabel = n == 2 ? "Longitude" : "")
        hm = plot_curvilinear!(ax, lon3, lat3, mk3(R[(n, pa)]); colormap = :viridis, colorrange = (0, lim))
        u, v = rdu("m$n", "u", pa), rdu("m$n", "v", pa)
        quiver_curvilinear!(ax, lon3, lat3, mk3(u .* cos.(ang3) .- v .* sin.(ang3)), mk3(u .* sin.(ang3) .+ v .* cos.(ang3));
                            skip = qsk, lengthscale = 0.6 / quantile(filter(isfinite, mk3(hypot.(u, v))), 0.99), color = (:white, 0.8),
                            shaftwidth = 0.5, tipwidth = 3, tiplength = 3)
        deco_map3!(ax)
        Colorbar(fig[n, 2], hm; height = h4)
        ms = [strip_stats3(R[(n, p)])[1] for p in pline]; ok = rows_ok3(ms[1])
        mp = strip_stats3(Rproj[n])[1]
        al = lineax(fig, (n, 3), "|U_$n| / c_$n averaged across the strip, by period", "|U_$n| / c_$n", n == 2)
        v_ = filter(isfinite, vcat([m[ok] for m in ms]..., mp[ok])); ytop = 1.3 * maximum(v_); ylims!(al, 0, ytop); deco_line3!(al, ytop)
        lines!(al, lkm, ifelse.(ok, mp, NaN); color = :gray40, linestyle = :dash, linewidth = 1.6, label = "projection on mode $n (fit mean term, all)")
        for (k, p) in enumerate(pline)
            lines!(al, lkm, ifelse.(ok, ms[k], NaN); color = pcol[k], linewidth = k == 1 ? 2.8 : 1.6, label = "φ²-weighted, $(ptit(p))")
        end
        xlims!(al, lkm[findfirst(ok)], lkm[jcrop3])
        @printf("  |U_%d|/c_%d strip mean / at the node: %s; projection (all) %.3f / %.3f\n", n, n,
                join(["$(per3[p]) $(round(mean(ms[k][ok]), digits = 3)) / $(round(ms[k][jn], digits = 3))" for (k, p) in enumerate(pline)], ", "),
                mean(mp[ok]), mp[jn])
    end
    Legend(fig[1:2, 4], al; framevisible = false, labelsize = 8pt)
    Label(fig[0, :], "Mean flow / mode speed |U_n| / c_n by period, U_n = (1/H)∫U φ_n² dz\n$dff_run"; fontsize = 11pt, font = :bold)
    Label(fig[3, :], "map: all ($(nrec3[findfirst(==("all"), per3)] ÷ 24) d), arrows: U_n east / north; c_n of the whole record for every period; " *
          "Aug: $(nrec3[findfirst(==("Aug"), per3)] ÷ 24) days only;\ngreen: ±$(round(Int, side_km)) km strip; cyan: north end of the line plots; " *
          "red star / dashed: 200 km node; grey: moorings on the line"; fontsize = 8pt)
    resize_to_layout!(fig)
    fname = joinpath(fig_path, "mean_flow_U_over_c_months.png"); save(fname, fig; px_per_unit = 2); println("saved $fname")
    # (c) 2 x 5 maps
    fig = Figure(fontsize = 10pt)
    for n in 1:2
        lim = round(quantile(filter(isfinite, vcat([vec(mk3(R[(n, p)])) for p in eachindex(per3)]...)), 0.99); sigdigits = 2)
        hm = nothing
        for (c, p) in enumerate(pord)
            ax = topdown_axis(fig[n, c]; title = "|U_$n|/c_$n, $(ptit(p))", backgroundcolor = :gray80, width = w3, height = h3p, aspect = nothing,
                              xlabel = n == 2 ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "", yticklabelsvisible = c == 1,
                              xticklabelsvisible = n == 2, xticks = WilkinsonTicks(3))
            hm = plot_curvilinear!(ax, lon3, lat3, mk3(R[(n, p)]); colormap = :viridis, colorrange = (0, lim))
            deco_map3!(ax)
        end
        Colorbar(fig[n, 6], hm; height = h3p, label = "|U_$n| / c_$n")
    end
    Label(fig[0, :], "|U_n| / c_n by period (U_n = (1/H)∫U φ_n² dz, c_n of the whole record)\n$dff_run"; fontsize = 11pt, font = :bold)
    Label(fig[3, :], "one colour range per mode (all periods); Aug: $(nrec3[findfirst(==("Aug"), per3)] ÷ 24) days only (from 08-27);\n" *
          "green: ±$(round(Int, side_km)) km strip; cyan: north end of the line plots; red star: 200 km node; shelf and edge bands grey"; fontsize = 8pt)
    colgap!(fig.layout, 8); rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    fname = joinpath(fig_path, "mean_flow_U_over_c_heatmap_month.png"); save(fname, fig; px_per_unit = 2); println("saved $fname")
  end

  # ==========================================================================
  # (6) monthly stratification (step 5): N² sections across the strip, c_n and |U_n| / c_n with each period's c_n
  # ==========================================================================
  # Sections: N² of each column of the strip interpolated to fixed depths (down to MF_N2_DEPTH, default 1500 m), averaged
  # across the strip per grid row: row 1 log10 N², row 2 log10(N² / N²_all) (the change against the whole window);
  # columns Aug, Sep, Oct, Nov, all. -> MF_FIG/mean_flow_N2_section_month.png
  # Lines: strip-mean c_n of each period, and |U_n| / c_n with each period's own c_n -> MF_FIG/mean_flow_c_n_month.png
  if 6 in mf_steps
    isfile(n2_fname) || error("no $n2_fname: run step 5 first (batch job)")
    D6 = NCDataset(n2_fname)
    per6 = split(D6.attrib["periods"], ","); nz6 = D6.dim["s_w_interior"] + 1
    h6 = NCDataset(g -> Float64.(g["h"][1:nx3, 1:ny3]), run_grid_fname)
    hcs = NCDataset(g -> (g.attrib["hc"], Float64.(g.attrib["Cs_w"])), first(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", basename(f)), readdir(joinpath(run_dir, "after_joint"); join = true))))
    sc_w6 = ((0:nz6) .- nz6) ./ nz6
    zmax6 = parse(Float64, get(ENV, "MF_N2_DEPTH", "1500"))
    dz6 = zmax6 <= 500 ? 2.0 : 10.0
    zg = collect(-dz6 / 2:-dz6:-zmax6)                            # fixed depths of the sections [m] (2 m steps for a shallow zoom)
    # strip-mean N² section of period p: (eta, zg); a depth counts where at least half of the strip's deep columns reach it
    function n2_section(p)
        N2p = Float64.(D6["N2"][1:nx3, 1:ny3, :, p]); zm = Float64.(D6["zeta_mean"][1:nx3, 1:ny3, p])
        sec = fill(NaN, ny3, length(zg))
        for j in 1:ny3
            acc = zeros(length(zg)); cnt = zeros(Int, length(zg)); ncol = 0
            for i in strip3
                show3[i, j] || continue
                ncol += 1
                zw = zm[i, j] .+ (zm[i, j] + h6[i, j]) .* ((hcs[1] .* sc_w6 .+ hcs[2] .* h6[i, j]) ./ (h6[i, j] + hcs[1]))
                zi = zw[2:nz6]                                     # interior interfaces (bottom -> top), increasing
                for (k, z) in enumerate(zg)
                    (zi[1] <= z <= zi[end]) || continue
                    m = searchsortedlast(zi, z); m = clamp(m, 1, length(zi) - 1)
                    w = (z - zi[m]) / (zi[m + 1] - zi[m])
                    acc[k] += log10(N2p[i, j, m]) * (1 - w) + log10(N2p[i, j, m + 1]) * w; cnt[k] += 1
                end
            end
            ncol >= 0.5 * length(strip3) || continue
            for k in eachindex(zg)
                cnt[k] >= 0.5 * ncol && (sec[j, k] = acc[k] / cnt[k])     # mean of log10 N²
            end
        end
        sec
    end
    println("\nstep 6: N² sections and c_n by period from $n2_fname -> $fig_path")
    secs = Dict(p => n2_section(p) for p in eachindex(per6))
    pa6 = findfirst(==("all"), per6)
    ok6 = [isfinite(lkm[j]) && j <= jcrop3 && any(isfinite, secs[pa6][j, :]) for j in 1:ny3]
    jr = findall(ok6)
    fig = Figure(fontsize = 10pt)
    lo, hi = quantile(filter(isfinite, vcat([vec(secs[p][jr, :]) for p in eachindex(per6)]...)), (0.02, 0.98))
    pord6 = [findfirst(==(p), per6) for p in ("Aug", "Sep", "Oct", "Nov", "all")]
    hms = Dict{Int, Any}()                                          # heatmap of each row, for its colour bar
    for (c, p) in enumerate(pord6)
        nrc = D6["nrec"][p]
        ttl = per6[p] == "all" ? "all ($(nrc ÷ 24) d)" : per6[p] == "Aug" ? "Aug ($(nrc ÷ 24) d)" : per6[p]
        for (r, (X, cm, cr, lab)) in enumerate(((secs[p], :thermal, (lo, hi), "log10 N² [s⁻²]"),
                                                 (secs[p] .- secs[pa6], :balance, (-0.3, 0.3), "log10(N² / N²_all)")))
            ax = Axis(fig[r, c]; title = r == 1 ? ttl : "$ttl vs all", width = 240, height = 220,
                      xlabel = "", ylabel = c == 1 ? "depth [m]" : "",
                      yticklabelsvisible = c == 1, xticklabelsvisible = r == 2)
            hm = heatmap!(ax, lkm[jr], zg, X[jr, :]; colormap = cm, colorrange = cr)
            vlines!(ax, [lkm[jn]]; color = :red, linestyle = :dash, linewidth = 1)
            hms[r] = hm
        end
    end
    Colorbar(fig[1, 6], hms[1]; label = "log10 N² [s⁻²]", height = 220); Colorbar(fig[2, 6], hms[2]; label = "log10(N² / N²_all)", height = 220)
    Label(fig[3, 1:5], "distance along the mooring line from the south boundary [km]"; tellwidth = false)
    Label(fig[0, :], "Stratification across the ±$(round(Int, side_km)) km strip by period (mean of log10 N² over the strip's deep columns)\n$dff_run"; fontsize = 11pt, font = :bold)
    Label(fig[4, :], "N² = -g/ρ₀ ∂ρ/∂z of each period's mean density (step 5); row 2: change against the whole window (+0.3 = 2x); red dashed: 200 km node; " *
          "Aug: $(D6["nrec"][findfirst(==("Aug"), per6)] ÷ 24) days only"; fontsize = 8pt)
    resize_to_layout!(fig)
    fname = joinpath(fig_path, zmax6 == 1500 ? "mean_flow_N2_section_month.png" : "mean_flow_N2_section_month_0-$(round(Int, zmax6))m.png")
    save(fname, fig; px_per_unit = 2); println("saved $fname")
    # c_n by period and |U_n| / c_n with each period's c_n
    pcol6 = [:black, Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3], Makie.wong_colors()[6]]
    pl6 = [findfirst(==(p), per6) for p in ("all", "Aug", "Sep", "Oct", "Nov")]
    fig = Figure(fontsize = 10pt)
    local al6
    for n in 1:2
        cs = [Float64.(D6["c"][1:nx3, 1:ny3, n, p]) for p in eachindex(per6)]
        for (col, (ttl, f)) in enumerate((("c_$n by period", p -> cs[p]),
                                          ("|U_$n| / c_$n, each period's c_$n", p -> hypot.(rdu("m$n", "u", findfirst(==(per6[p]), per3)), rdu("m$n", "v", findfirst(==(per6[p]), per3))) ./ cs[p])))
            ms = [strip_stats3(f(p))[1] for p in pl6]; ok = rows_ok3(ms[1])
            al6 = Axis(fig[n, col]; title = ttl, ylabel = col == 1 ? "c_$n [m s⁻¹]" : "|U_$n| / c_$n",
                       xlabel = n == 2 ? "distance along the mooring line [km]" : "",
                       xticks = 0:100:ceil(Int, lkm[end]), width = 560, height = 300)
            ytop = 1.3 * maximum(filter(isfinite, vcat([m[ok] for m in ms]...))); ylims!(al6, 0, ytop); deco_line3!(al6, ytop)
            for (k, p) in enumerate(pl6)
                lines!(al6, lkm, ifelse.(ok, ms[k], NaN); color = pcol6[k], linewidth = k == 1 ? 2.8 : 1.6,
                       label = per6[p] == "all" ? "all" : per6[p] == "Aug" ? "Aug (5 d)" : per6[p])
            end
            xlims!(al6, lkm[findfirst(ok)], lkm[jcrop3])
            @printf("  mode %d %-36s strip mean / node: %s\n", n, ttl, join(["$(per6[p]) $(round(mean(ms[k][ok]), sigdigits = 3)) / $(round(ms[k][jn], sigdigits = 3))" for (k, p) in enumerate(pl6)], ", "))
        end
    end
    Legend(fig[1:2, 3], al6; framevisible = false, labelsize = 9pt)
    Label(fig[0, :], "Mode speed of each period's stratification and |U_n| / c_n, mean across ±$(round(Int, side_km)) km of the mooring line\n$dff_run"; fontsize = 11pt, font = :bold)
    Label(fig[3, :], "c_n from each period's mean density (step 5); U_n = (1/H)∫U φ_n² dz of the period (step 1); red dashed: 200 km node; grey: moorings on the line"; fontsize = 8pt)
    resize_to_layout!(fig)
    fname = joinpath(fig_path, "mean_flow_c_n_month.png"); save(fname, fig; px_per_unit = 2); println("saved $fname")
    close(D6)
  end

  # ==========================================================================
  # (7) 2-D stratification fields of each period (from the step-5 N²)
  # ==========================================================================
  # Per column and period, from N² of mean_flow_N2_month.nc on the period's mean level depths (zeta_mean of the period):
  #   MLD       density rebuilt from N² as it was made (rho_k = rho_k+1 + rho0/g N²_k Δz, k = 1 bottom), the depth where
  #             rho exceeds its 10 m value by MF_MLD_DRHO (default 0.03 kg m-3, de Boyer Montégut et al. 2004); h if never
  #   z_pyc     depth of the largest N² below 10 m (the reference depth of the MLD: the thin plume layer at the top left out)
  #   N_max     sqrt of that N² [cycles per hour]
  #   c_WKB     (1/π) Σ N Δz over the column [m s-1], the WKB estimate of c_1 (compare with c_1 of step 5)
  # -> <bcef_dir>/mean_flow_strat_month.nc (xi, eta, period); figures -> MF_FIG:
  #   mean_flow_strat_<MLD|zpyc|Nmax|cWKB>.png  maps, row 1 the value, row 2 the change against all (columns Aug .. all)
  #   mean_flow_strat_lines.png                 strip means of the four fields by period
  if 7 in mf_steps
    isfile(n2_fname) || error("no $n2_fname: run step 5 first (batch job)")
    D7 = NCDataset(n2_fname)
    per7 = split(D7.attrib["periods"], ","); rho07 = Float64(D7.attrib["rho0"]); nz7 = D7.dim["s_w_interior"] + 1
    drho = parse(Float64, get(ENV, "MF_MLD_DRHO", "0.03"))
    h7 = NCDataset(g -> Float64.(g["h"][1:nx3, 1:ny3]), run_grid_fname)
    hc7, Cs_r7 = NCDataset(g -> (g.attrib["hc"], Float64.(g.attrib["Cs_r"])),
                           first(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", basename(f)), readdir(joinpath(run_dir, "after_joint"); join = true))))
    sc_r7 = ((1:nz7) .- nz7 .- 0.5) ./ nz7
    function strat_cols(N2::Array{Float64, 3}, zm::Matrix{Float64}, h::Matrix{Float64}, hc, Cs_r, sc_r, rho0, drho)
        nx_, ny_, nz_ = size(N2, 1), size(N2, 2), size(N2, 3) + 1
        mld, zp, nmax, cw = fill(NaN, nx_, ny_), fill(NaN, nx_, ny_), fill(NaN, nx_, ny_), fill(NaN, nx_, ny_)
        zr = zeros(nz_); rho = zeros(nz_)
        @inbounds for j in 1:ny_, i in 1:nx_
            hh = h[i, j]
            for k in 1:nz_
                zr[k] = zm[i, j] + (zm[i, j] + hh) * (hc * sc_r[k] + Cs_r[k] * hh) / (hh + hc)
            end
            rho[nz_] = 0.0                                          # relative to the top level
            for k in nz_-1:-1:1
                rho[k] = rho[k + 1] + rho0 / 9.81 * N2[i, j, k] * (zr[k + 1] - zr[k])
            end
            c = 0.0; nm = -1.0; zpk = NaN
            for k in 1:nz_-1
                c += sqrt(N2[i, j, k]) * (zr[k + 1] - zr[k])
                zk = 0.5 * (zr[k] + zr[k + 1])
                if zk < -10 && N2[i, j, k] > nm
                    nm = N2[i, j, k]; zpk = -zk
                end
            end
            cw[i, j] = c / π; zp[i, j] = zpk; nmax[i, j] = nm > 0 ? sqrt(nm) / (2π) * 3600 : NaN
            # density at 10 m (linear in z between levels), then the first depth below with rho >= rho10 + drho
            hh < 20 && continue
            k10 = findlast(k -> zr[k] <= -10, 1:nz_)
            (k10 === nothing || k10 == nz_) && continue
            w = (-10 - zr[k10]) / (zr[k10 + 1] - zr[k10]); r10 = rho[k10] * (1 - w) + rho[k10 + 1] * w
            m = hh
            for k in k10:-1:1
                if rho[k] >= r10 + drho
                    zu = k == k10 ? -10.0 : zr[k + 1]; ru = k == k10 ? r10 : rho[k + 1]
                    m = -(zu + (r10 + drho - ru) / (rho[k] - ru) * (zr[k] - zu)); break
                end
            end
            mld[i, j] = m
        end
        mld, zp, nmax, cw
    end
    println("\nstep 7: MLD (Δρ = $drho kg/m³ from 10 m), pycnocline depth, N_max, c_WKB from $n2_fname")
    F7 = Dict{String, Array{Float64, 3}}(k => fill(NaN, nx3, ny3, length(per7)) for k in ("MLD", "zpyc", "Nmax", "cWKB"))
    for p in eachindex(per7)
        t0 = time()
        r = strat_cols(Float64.(D7["N2"][1:nx3, 1:ny3, :, p]), Float64.(D7["zeta_mean"][1:nx3, 1:ny3, p]), h7, hc7, Cs_r7, sc_r7, rho07, drho)
        for (k, a) in zip(("MLD", "zpyc", "Nmax", "cWKB"), r); F7[k][:, :, p] .= a; end
        @printf("  %-3s: %.0f s\n", per7[p], time() - t0)
    end
    c1_7 = Float64.(D7["c"][1:nx3, 1:ny3, 1, :])
    let ok = show3 .& (h7 .> 1000), pa = findfirst(==("all"), per7)
        @printf("  check c_WKB vs c_1 (all, deep): median ratio %.3f\n", median((F7["cWKB"][:, :, pa] ./ c1_7[:, :, pa])[ok]))
    end
    st_fname = joinpath(bcef_dir, "mean_flow_strat_month.nc")
    tmp = st_fname * ".tmp"; isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do o
        defDim(o, "xi_rho", nx3); defDim(o, "eta_rho", ny3); defDim(o, "period", length(per7))
        o.attrib["title"] = "Stratification fields of each period, $dff_run"; o.attrib["periods"] = join(per7, ",")
        o.attrib["source"] = n2_fname; o.attrib["script"] = "internalwave_julia/DFF/mean_flow_month.jl (step 7)"
        for (k, u, ln) in (("MLD", "m", "mixed layer depth, rho - rho(10 m) >= $drho kg m-3"), ("zpyc", "m", "depth of the largest N² below 10 m"),
                           ("Nmax", "cph", "sqrt of the largest N² below 10 m"), ("cWKB", "m s-1", "(1/pi) int N dz, WKB mode-1 speed"))
            defVar(o, k, Float32.(F7[k]), ("xi_rho", "eta_rho", "period"); attrib = ["units" => u, "long_name" => ln])
        end
    end
    mv(tmp, st_fname; force = true); println("  wrote $st_fname")
    # maps: row 1 value, row 2 change against all
    pord7 = [findfirst(==(p), per7) for p in ("Aug", "Sep", "Oct", "Nov", "all")]; pa7 = findfirst(==("all"), per7)
    nrec7 = D7["nrec"][:]
    pt7(p) = per7[p] == "all" ? "all ($(nrec7[p] ÷ 24) d)" : per7[p] == "Aug" ? "Aug ($(nrec7[p] ÷ 24) d)" : per7[p]
    fields7 = [("MLD", "mixed layer depth (Δρ = $drho kg m⁻³ from 10 m)", "MLD [m]", :deep, false, "m"),
               ("zpyc", "pycnocline depth (largest N² below 10 m)", "z_pyc [m]", :deep, false, "m"),
               ("Nmax", "pycnocline strength N_max", "N_max [cph]", :thermal, true, "%"),
               ("cWKB", "WKB mode-1 speed (1/π)∫N dz", "c_WKB [m s⁻¹]", :haline, true, "%")]
    # MF_STRAT_FIGS: which figures (default MLD:zpyc:Nmax:cWKB:lines; the login node manages about two per process)
    figs7 = String.(split(get(ENV, "MF_STRAT_FIGS", "MLD:zpyc:Nmax:cWKB:lines"), r"[,:; ]+"; keepempty = false))
    for (k, ttl, lab, cm, rel, dunit) in fields7
        k in figs7 || continue
        A = F7[k]
        vals = filter(isfinite, vcat([vec(mk3(A[:, :, p])) for p in eachindex(per7)]...))
        lo, hi = quantile(vals, (0.01, 0.99))
        dif(p) = rel ? 100 .* (A[:, :, p] ./ A[:, :, pa7] .- 1) : A[:, :, p] .- A[:, :, pa7]
        dlim = round(quantile(abs.(filter(isfinite, vcat([vec(mk3(dif(p))) for p in eachindex(per7) if p != pa7]...))), 0.98); sigdigits = 2)
        fig = Figure(fontsize = 10pt)
        hms = Dict{Int, Any}()
        for (c, p) in enumerate(pord7), r in 1:2
            ax = topdown_axis(fig[r, c]; title = r == 1 ? pt7(p) : "$(per7[p]) − all", backgroundcolor = :gray80, width = w3, height = h3p, aspect = nothing,
                              xlabel = r == 2 ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "", yticklabelsvisible = c == 1,
                              xticklabelsvisible = r == 2, xticks = WilkinsonTicks(3))
            hms[r] = r == 1 ? plot_curvilinear!(ax, lon3, lat3, mk3(A[:, :, p]); colormap = cm, colorrange = (lo, hi)) :
                              plot_curvilinear!(ax, lon3, lat3, mk3(dif(p)); colormap = :balance, colorrange = (-dlim, dlim))
            deco_map3!(ax)
        end
        Colorbar(fig[1, 6], hms[1]; height = h3p, label = lab)
        Colorbar(fig[2, 6], hms[2]; height = h3p, label = rel ? "change [%]" : "change [$dunit]")
        Label(fig[0, :], "$ttl by period\n$dff_run"; fontsize = 11pt, font = :bold)
        Label(fig[3, :], "from each period's mean density (step 5); row 2: the period minus the whole window" * (rel ? " (in % of all)" : "") *
              "; Aug: $(nrec7[findfirst(==("Aug"), per7)] ÷ 24) days only;\ngreen: ±$(round(Int, side_km)) km strip; cyan: north end of the line plots; red star: 200 km node; shelf (h < $(Int(h_mask)) m) and edge bands grey"; fontsize = 8pt)
        colgap!(fig.layout, 8); rowgap!(fig.layout, 8); resize_to_layout!(fig)
        fname = joinpath(fig_path, "mean_flow_strat_$(k).png"); save(fname, fig; px_per_unit = 2); println("  saved $fname")
    end
    # strip means of the four fields by period
  if "lines" in figs7
    pcol7 = [:black, Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3], Makie.wong_colors()[6]]
    pl7 = [findfirst(==(p), per7) for p in ("all", "Aug", "Sep", "Oct", "Nov")]
    fig = Figure(fontsize = 10pt)
    local al7
    for (q, (k, ttl, lab, _, _, _)) in enumerate(fields7)
        r, c = fldmod1(q, 2)
        ms = [strip_stats3(F7[k][:, :, p])[1] for p in pl7]; ok = rows_ok3(ms[1])
        al7 = Axis(fig[r, c]; title = ttl, ylabel = lab, xlabel = r == 2 ? "distance along the mooring line [km]" : "",
                   xticks = 0:100:ceil(Int, lkm[end]), width = 560, height = 280, yreversed = k in ("MLD", "zpyc"))
        v = filter(isfinite, vcat([m[ok] for m in ms]...)); lo, hi = extrema(v); pad = 0.15 * (hi - lo)
        ylims!(al7, k in ("MLD", "zpyc") ? (hi + pad, max(lo - pad, 0)) : (max(lo - pad, 0), hi + pad))
        vlines!(al7, [lkm[jn]]; color = :red, linestyle = :dash, linewidth = 1.2)
        for (nm, j) in moor_in3; vlines!(al7, [lkm[j]]; color = :gray80, linewidth = 0.6); end
        k == "cWKB" && lines!(al7, lkm, ifelse.(ok, strip_stats3(c1_7[:, :, pa7])[1], NaN); color = :gray40, linestyle = :dash, linewidth = 1.6, label = "c_1 (eigen, all)")
        for (q2, p) in enumerate(pl7)
            lines!(al7, lkm, ifelse.(ok, ms[q2], NaN); color = pcol7[q2], linewidth = q2 == 1 ? 2.6 : 1.5, label = pt7(p))
        end
        xlims!(al7, lkm[findfirst(ok)], lkm[jcrop3])
        @printf("  %-5s strip mean / at the node: %s\n", k, join(["$(per7[p]) $(round(mean(ms[q2][ok]), sigdigits = 3)) / $(round(ms[q2][jn], sigdigits = 3))" for (q2, p) in enumerate(pl7)], ", "))
    end
    Legend(fig[1:2, 3], al7; framevisible = false, labelsize = 9pt)
    Label(fig[0, :], "Stratification by period, mean across ±$(round(Int, side_km)) km of the mooring line\n$dff_run"; fontsize = 11pt, font = :bold)
    Label(fig[3, :], "MLD: ρ − ρ(10 m) ≥ $drho kg m⁻³; z_pyc, N_max: largest N² below 10 m; c_WKB = (1/π)∫N dz (dashed: c_1 of the eigen problem); red dashed: 200 km node"; fontsize = 8pt)
    resize_to_layout!(fig)
    fname = joinpath(fig_path, "mean_flow_strat_lines.png"); save(fname, fig; px_per_unit = 2); println("  saved $fname")
  end
    close(D7)
  end
    close(D3)
end
