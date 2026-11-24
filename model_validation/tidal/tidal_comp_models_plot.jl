#= tidal_comp_models_plot.jl
Hsin-Yi Chen, USM DMS
Model intercomparison of the SSH tide vs FES2022b. Reads the <SRC>_FES_sshfit_comp[_s<n>].nc files of
ssh_tidalcomp_fes_models.jl (looked for in P.fitdirs: expanse also finds the ones made on aurelia) and
recomputes the statistics over the COMMON comparison area (intersection of every model's area), so every
model is scored on the same points. Plotting only.
Settings: tidal_models_setup.jl (TC_HOST, TC_STRIDE); TC_SRC is ignored here, every model found is used
(TC_PLOT_SRC = comma list to choose).

Output (P.out):
  models_FES_summary[_s<n>].txt     table per model and constituent, common area
  models_FES_rms[_s<n>].png         RMS (shelf / deep / all), amplitude ratio, phase difference per constituent
  models_FES_dA_maps[_s<n>].png     A_bt - A_FES maps, rows = models, columns = constituents
  models_IT_maps[_s<n>].png         coherent internal-tide SSH amplitude |Z_R - Z_B| per model, M2 and K1

Usage:
  julia --project=. -t 1 --heap-size-hint=3G model_validation/tidal/tidal_comp_models_plot.jl
=#

using NCDatasets, CairoMakie, Printf, Statistics
include(joinpath(@__DIR__, "tidal_models_setup.jl"))
include(P.func * "fig_setup.jl")                                         # figkw, figsave
include(P.func * "plotting_fun.jl")                                      # map_grid, grid_edge, lonfmt, latfmt

const PLOT_SRC = haskey(ENV, "TC_PLOT_SRC") ? split(ENV["TC_PLOT_SRC"], ",") : SRC_ORDER
files = [(s, findfile(compfile(s))) for s in PLOT_SRC]
for (s, f) in files; f === nothing && println("no $(compfile(s)) for $s, skipped"); end
files = filter(x -> x[2] !== nothing, files)
isempty(files) && error("no comparison files in $(join(P.fitdirs, ", ")) (run ssh_tidalcomp_fes_models.jl first)")

## --- read
rdZ(ds, n) = Float64.(ds["A_$n"][:, :, :]) .* cis.(-deg2rad.(Float64.(ds["G_$n"][:, :, :])))
D = map(files) do (s, f)
    NCDataset(f) do ds
        (src = s, label = ds.attrib["label"], cons = String.(ds["con"][:]), inferred = Bool.(ds["inferred"][:]),
         lon = Float64.(ds["lon_rho"][:, :]), lat = Float64.(ds["lat_rho"][:, :]), h = Float64.(ds["h"][:, :]),
         area = Float64.(ds["area"][:, :]), M = Bool.(ds["M"][:, :]), hshelf = Float64(ds.attrib["hshelf"]), hdeep = Float64(ds.attrib["hdeep"]),
         ZR = rdZ(ds, "R"), ZF = rdZ(ds, "F"), ZB = rdZ(ds, "B"), ZIT = rdZ(ds, "IT"), tc = ds.attrib["tc"])
    end
end
NM = length(D); cons = D[1].cons; K = length(cons)
for d in D; (d.cons == cons && size(d.lon) == size(D[1].lon)) || error("$(d.src): other constituents or grid than $(D[1].src)"); end
lonR, latR, hR = D[1].lon, D[1].lat, D[1].h
MC = reduce(.&, [d.M for d in D])                                          # common comparison area
W = MC .* D[1].area; shelf = hR .< D[1].hshelf
CB = [c * (i ? "*" : "") for (c, i) in zip(cons, D[1].inferred)]
println("models: ", join([d.src for d in D], ", "), "; common area $(count(MC)) points (", join([count(d.M) for d in D], ", "), " per model)")

## --- statistics over the common area (same definitions as ssh_tidalcomp_fes.jl)
wmean(X, w = W) = sum(w[w .> 0] .* X[w .> 0]) / sum(w)
rmsd(Z1, Z2, w = W) = sqrt(wmean(abs2.(Z1 .- Z2) ./ 2, w))
srms(Z, w = W) = sqrt(wmean(abs2.(Z) ./ 2, w))
stat = Dict{Tuple{String, String}, NamedTuple}()
for d in D, (k, c) in enumerate(cons)
    zb, zf, zr = d.ZB[:, :, k], d.ZF[:, :, k], d.ZR[:, :, k]
    stat[(d.src, c)] = (S = srms(zf), rms = rmsd(zb, zf), rms_sh = rmsd(zb, zf, W .* shelf), S_sh = srms(zf, W .* shelf),
                        rms_dp = rmsd(zb, zf, W .* .!shelf), S_dp = srms(zf, W .* .!shelf), rms_full = rmsd(zr, zf),
                        ratio = wmean(abs.(zb)) / wmean(abs.(zf)), dG = -rad2deg(angle(wmean(zb .* conj.(zf)))), it = srms(d.ZIT[:, :, k]))
end
io = IOBuffer()
@printf(io, "SSH tide vs FES2022b, common comparison area of %s: %d points on the ROMS 2700 m grid, tc %s\n", join([d.src for d in D], ", "), count(MC), D[1].tc)
@printf(io, "bt = FES + LP(model - FES); RMS = sqrt(<|dZ|^2/2>) [cm], rel = RMS / FES signal rms; ratio = <A_bt>/<A_FES>; dG = model - FES [deg], + = model lags\n")
@printf(io, "shelf h < %.0f m; RMSfull = model unfiltered; IT = rms of model - bt; * corrected with the FES ratio (K2, NU2, P1)\n", D[1].hshelf)
for (k, c) in enumerate(cons)
    @printf(io, "\n%-4s FES signal rms %.3f cm (shelf %.3f, deep %.3f)\n", CB[k], stat[(D[1].src, c)].S, stat[(D[1].src, c)].S_sh, stat[(D[1].src, c)].S_dp)
    @printf(io, "  %-30s %7s %6s | %7s %6s %7s %6s | %6s %7s | %7s %6s\n", "model", "RMS", "rel", "RMSshlf", "rel", "RMSdeep", "rel", "ratio", "dG", "RMSfull", "IT")
    for d in D
        r = stat[(d.src, c)]
        @printf(io, "  %-30s %7.3f %5.1f%% | %7.3f %5.1f%% %7.3f %5.1f%% | %6.3f %+7.2f | %7.3f %6.2f\n", d.label, r.rms, 100r.rms / r.S,
                r.rms_sh, 100r.rms_sh / r.S_sh, r.rms_dp, 100r.rms_dp / r.S_dp, r.ratio, r.dG, r.rms_full, r.it)
    end
end
@printf(io, "\nRSS over %s [cm]:\n", join(cons, " "))
for d in D
    rs(key) = sqrt(sum(getfield(stat[(d.src, c)], key)^2 for c in cons))
    @printf(io, "  %-30s bt %.3f (shelf %.3f, deep %.3f), unfiltered %.3f\n", d.label, rs(:rms), rs(:rms_sh), rs(:rms_dp), rs(:rms_full))
end
tab = String(take!(io)); print(tab); mkpath(P.out)
write(P.out * "models_FES_summary$(SFX).txt", tab); println("wrote ", P.out * "models_FES_summary$(SFX).txt")

## --- figure 1: RMS (all / shelf / deep), amplitude ratio, phase difference per constituent, one bar per model
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227"][1:NM]
fig1 = Figure(; figkw(18, 20)...)
bw = 0.8 / NM; xo(m) = (m - (NM + 1) / 2) * bw
panels = [("RMS difference bt − FES, comparison area", "cm", r -> r.rms, true),
          ("RMS, shelf (h < $(Int(D[1].hshelf)) m)", "cm", r -> r.rms_sh, true),
          ("RMS, deep (h ≥ $(Int(D[1].hshelf)) m)", "cm", r -> r.rms_dp, true),
          ("amplitude ratio <A_bt> / <A_FES>", "", r -> r.ratio, false),
          ("phase bt − FES (+ = model lags)", "deg", r -> r.dG, false)]
axs = Axis[]
for (q, (ttl, lab, f, lg)) in enumerate(panels)
    ax = Axis(fig1[q, 1]; title = ttl, titlesize = 9, titlefont = :regular, ylabel = lab, xticks = (1:K, CB), xticklabelsvisible = q == length(panels),
              yscale = lg ? log10 : identity)
    for (m, d) in enumerate(D)
        v = [f(stat[(d.src, c)]) for c in cons]
        barplot!(ax, (1:K) .+ xo(m), lg ? max.(v, 1e-3) : v; width = bw * 0.95, color = cols[m], strokewidth = 0,
                 fillto = lg ? 1e-3 : (q == 4 ? 1.0 : 0.0), label = d.label)
    end
    if lg
        S = [q == 1 ? stat[(D[1].src, c)].S : q == 2 ? stat[(D[1].src, c)].S_sh : stat[(D[1].src, c)].S_dp for c in cons]
        scatter!(ax, 1:K, S; marker = :hline, markersize = 18, color = :black, label = "FES signal rms")
        ylims!(ax, 0.1, 2 * maximum(S))
    else
        hlines!(ax, [q == 4 ? 1.0 : 0.0]; color = :gray40, linewidth = 0.7)
    end
    xlims!(ax, 0.4, K + 0.6); push!(axs, ax)
end
Legend(fig1[length(panels) + 1, 1], axs[1]; orientation = :horizontal, nbanks = 2, framevisible = false, labelsize = 9)
Label(fig1[0, 1], "SSH tide vs FES2022b, common comparison area ($(count(MC)) points of the ROMS 2700 m grid)\n* corrected for K2 / NU2 / P1 with the FES ratio",
      fontsize = 10, tellwidth = false)
figsave(P.out * "models_FES_rms$(SFX).png", fig1); println("wrote ", P.out * "models_FES_rms$(SFX).png")

## --- maps on a regular lon/lat grid (function/plotting_fun.jl)
mg = map_grid(lonR, latR; dll = 0.03 * STRIDE)
lonv, latv, rg = mg.lonv, mg.latv, mg.rg
hmap = rg(hR)
ib = findall(MC); box = grid_edge(lonR, latR; i = minimum(first.(Tuple.(ib))):maximum(first.(Tuple.(ib))),
                                         j = minimum(last.(Tuple.(ib))):maximum(last.(Tuple.(ib))))
mapax = (; aspect = DataAspect(), xtickformat = lonfmt, ytickformat = latfmt, xticks = 308:8:324, yticks = -4:4:12,
         xticklabelsize = 7, yticklabelsize = 7)
function decor!(a)
    contour!(a, lonv, latv, hmap; levels = [200, 1000, 3000], color = RGBf(0.3, 0.3, 0.3), linewidth = 0.5)
    lines!(a, grid_edge(lonR, latR)...; color = :gray30, linewidth = 0.5)
    lines!(a, box...; color = :gray20, linewidth = 0.7, linestyle = :dot)
    limits!(a, first(lonv), last(lonv), first(latv), last(latv))
end

## --- figure 2: A_bt - A_FES, rows = models, columns = constituents; one colour range per constituent
fig2 = Figure(; figkw(18, 3.7NM + 2.4)...)
for (k, c) in enumerate(cons)
    dAs = [ifelse.(MC, abs.(d.ZB[:, :, k]) .- abs.(d.ZF[:, :, k]), NaN) for d in D]
    dm = quantile(abs.(filter(isfinite, reduce(vcat, vec.(dAs)))), 0.99)
    local h
    for (m, d) in enumerate(D)
        dd = filter(isfinite, dAs[m])
        a = Axis(fig2[m, k]; mapax..., title = @sprintf("%s\nmean %+.1f, rms %.1f", m == 1 ? CB[k] : "", mean(dd), sqrt(mean(abs2, dd))),
                 titlesize = 8, yticklabelsvisible = k == 1, xticklabelsvisible = m == NM)
        h = heatmap!(a, lonv, latv, rg(dAs[m]); colormap = :balance, colorrange = (-dm, dm), nan_color = :white)
        decor!(a)
    end
    Colorbar(fig2[NM + 1, k], h; vertical = false, flipaxis = false, height = 7, ticklabelsize = 7, label = "cm", labelsize = 8)
end
rowlab(d) = replace(d.label, " (" => "\n(")
for (m, d) in enumerate(D); Label(fig2[m, 0], rowlab(d); rotation = π / 2, tellheight = false, fontsize = 8, font = :bold); end
Label(fig2[-1, 0:K], "A_bt − A_FES [cm] over the common comparison area (red: model too high); title: area mean and rms\n" *
      "grey: 200, 1000, 3000 m isobaths; * corrected for K2 / NU2 / P1 with the FES ratio", fontsize = 9, tellwidth = false)
colgap!(fig2.layout, 4); rowgap!(fig2.layout, 4)
figsave(P.out * "models_FES_dA_maps$(SFX).png", fig2); println("wrote ", P.out * "models_FES_dA_maps$(SFX).png")

## --- figure 3: coherent internal-tide SSH amplitude, M2 and K1
cit = filter(c -> c in cons, ["M2", "K1"])
fig3 = Figure(; figkw(18, 5NM ÷ 2 + 6)...)
for (q, c) in enumerate(cit)
    k = findfirst(==(c), cons)
    deepm = MC .& (hR .>= D[1].hdeep)
    top = quantile(filter(isfinite, reduce(vcat, [abs.(d.ZIT[:, :, k])[deepm] for d in D])), 0.995)
    local h
    for (m, d) in enumerate(D)
        a = Axis(fig3[q, m]; mapax..., title = @sprintf("%s\n%s, IT rms %.2f cm", d.label, c, stat[(d.src, c)].it), titlesize = 8,
                 yticklabelsvisible = m == 1, xticklabelsvisible = q == length(cit))
        h = heatmap!(a, lonv, latv, rg(ifelse.(MC, abs.(d.ZIT[:, :, k]), NaN)); colormap = :Blues, colorrange = (0, top), highclip = :black, nan_color = :white)
        decor!(a)
    end
    Colorbar(fig3[q, NM + 1], h; width = 7, ticklabelsize = 7, label = "cm", labelsize = 8)
end
Label(fig3[0, 1:NM], "Coherent internal-tide SSH amplitude |Z − Z_bt| [cm] (colour limit from h > $(round(Int, D[1].hdeep)) m; black above it)", fontsize = 9, tellwidth = false)
colgap!(fig3.layout, 4)
figsave(P.out * "models_IT_maps$(SFX).png", fig3); println("wrote ", P.out * "models_IT_maps$(SFX).png")
