#= tidal_comp_plotting.jl
Hsin-Yi Chen, USM DMS
Figures of the ROMS SSH tide vs FES2022b. Plotting only: everything comes from the file saved by
ssh_tidalcomp_fes.jl (ROMS_FES_sshfit_comp[_s<STRIDE>].nc), no ROMS or FES files are read here.

Section 1: read the comparison file. Each Z = A exp(-iG) [cm] is stored as amplitude and phase:
  R ROMS tide, F FES2022b, B ROMS barotropic, IT coherent internal-tide SSH (R - B),
  CP ROMS low-passed over deep water for co-phase lines (NaN outside deep water).

Usage:
  julia -t 2 --heap-size-hint=3G tidal_comp_plotting.jl   # full-grid file (the hint keeps it under the login node limit)
  STRIDE=20 julia tidal_comp_plotting.jl    # the every-20th-point test file (_s20)
=#

using NCDatasets, CairoMakie, Printf, Statistics, Dates
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")   # figkw, figsave, ...
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # map_grid, grid_edge, lonfmt, latfmt

## --- settings
const data_folder = "/expanse/lustre/projects/uso102/hchen54/figure/amazon_900m_3mon_1008/tidal_comp/"
const STRIDE      = parse(Int, get(ENV, "STRIDE", "1"))   # which file to read (1 = full grid)
const fname       = data_folder * "ROMS_FES_sshfit_comp$(STRIDE > 1 ? "_s$STRIDE" : "").nc"
const figdir      = "/home/hchen54/figure_check/"          # figures go here for now
mkpath(figdir)

## --- section 1: read the comparison file (Julia order: xi, eta, con)
isfile(fname) || error("no file $fname (run ssh_tidalcomp_fes.jl first)")
comp_data = NCDataset(fname) do ds
    rdZ(n) = Float64.(ds["A_$n"][:, :, :]) .* cis.(-deg2rad.(Float64.(ds["G_$n"][:, :, :])))   # A exp(-iG) [cm]
    (; cons     = String.(ds["con"][:]),
       inferred = Bool.(ds["inferred"][:]),                       # corrected with a FES ratio
       lon      = Float64.(ds["lon_rho"][:, :]), lat = Float64.(ds["lat_rho"][:, :]), h = Float64.(ds["h"][:, :]),
       M        = Bool.(ds["M"][:, :]),                           # comparison area
       ZR = rdZ("R"), ZF = rdZ("F"), ZB = rdZ("B"), ZIT = rdZ("IT"), ZCP = rdZ("CP"),
    # ROMS tide / FES atlas / ROMS barotropic (remove IT) / nternal tide only / ROMS low-passed over deep water
       resid    = Float64.(ds["resid_std"][:, :]), sshstd = Float64.(ds["ssh_std"][:, :]),   # [cm]
       A_mean   = ds["A_mean"][:], S = ds["S"][:], rms = ds["rms"][:], ratio = ds["ratio"][:], dG = ds["dG"][:],
       rms_shelf = ds["rms_shelf"][:], rms_deep = ds["rms_deep"][:], rms_full = ds["rms_full"][:], it_rms = ds["it_rms"][:],
       title    = ds.attrib["title"], tc = ds.attrib["tc"],
       hshelf   = Float64(ds.attrib["hshelf"]), hdeep = Float64(ds.attrib["hdeep"]))
end
NX, NY = size(comp_data.lon); K = length(comp_data.cons)
kc(c) = findfirst(==(c), comp_data.cons)                          # index of a constituent

println(comp_data.title)
@printf("grid %d x %d, %d constituents: %s; comparison area %d points\n", NX, NY, K, join(comp_data.cons, " "), count(comp_data.M))
for (k, c) in enumerate(comp_data.cons)
    @printf("  %-3s S %6.2f cm   RMS %5.2f cm   ratio %.3f   dG %+6.2f deg\n", c, comp_data.S[k], comp_data.rms[k], comp_data.ratio[k], comp_data.dG[k])
end


## --- figure 1: RMS difference per constituent (top), relative to the FES signal (bottom)
CB = [c * (i ? "*" : "") for (c, i) in zip(comp_data.cons, comp_data.inferred)]        # * corrected with a FES ratio
cols = ["#2a78d6", "#eb6834", "#1baf7a"]
fig1 = Figure(; figkw(17, 13)...)
ax = Axis(fig1[1, 1]; ylabel = "RMS difference [cm]", xticks = (1:K, CB), title = "ROMS Amazon 900 m vs FES2022b, comparison area",
          yscale = log10, yminorticksvisible = true, yminorgridvisible = true)
ser = [("ROMS barotropic vs FES, shelf (h < $(Int(comp_data.hshelf)) m)", comp_data.rms_shelf),
       ("ROMS barotropic vs FES, deep", comp_data.rms_deep), ("ROMS vs FES, unfiltered", comp_data.rms_full)]
for (s, (lab, v)) in enumerate(ser)
    barplot!(ax, (1:K) .+ (s - 2) * 0.27, max.(v, 1e-3); width = 0.25, color = cols[s], label = lab, fillto = 1e-3, strokewidth = 0)
end
scatter!(ax, 1:K, comp_data.S; marker = :hline, markersize = 16, color = :black, label = "FES signal rms")
ylims!(ax, 1e-1, 2 * maximum(comp_data.S))
Legend(fig1[2, 1], ax; orientation = :horizontal, nbanks = 2, framevisible = false, labelsize = 9)
ax2 = Axis(fig1[3, 1]; ylabel = "RMS / signal rms [%]", xticks = (1:K, CB))
rel = 100 .* comp_data.rms ./ comp_data.S                                # ROMS barotropic vs FES
barplot!(ax2, 1:K, rel; color = cols[1], strokewidth = 0)
text!(ax2, 1:K, rel; text = [@sprintf("%.1f", v) for v in rel], align = (:center, :bottom), offset = (0, 2), fontsize = 9, color = "#52514e")
ylims!(ax2, 0, 1.4 * maximum(rel))
xlims!(ax, 0.4, K + 0.6); xlims!(ax2, 0.4, K + 0.6)                       # same x-positions in both panels
Label(fig1[4, 1], "* corrected for K2 / NU2 / P1 with the FES ratio (record too short to separate)", fontsize = 9, tellwidth = false)
rowsize!(fig1.layout, 3, Relative(0.25))
figsave(figdir * "ROMS_FES_rms$(STRIDE > 1 ? "_s$STRIDE" : "").png", fig1)

## --- map helpers: fields are drawn on a regular lon/lat grid (function/plotting_fun.jl: map_grid, grid_edge, lonfmt, latfmt)
const DLL = 0.01                                                         # deg, map grid spacing (~1.1 km)
const BATHY = [500, 1000, 2000]                                          # m, grey isobaths
lonR, latR, hR = comp_data.lon, comp_data.lat, comp_data.h
mg = map_grid(lonR, latR; dll = DLL)
lonv, latv, rg = mg.lonv, mg.latv, mg.rg
hmap = rg(hR)
mapax = (; aspect = DataAspect(), xtickformat = lonfmt, ytickformat = latfmt, xticks = 312:4:320, yticks = -2:2:8)
deep(X) = ifelse.(hmap .>= comp_data.hdeep, X, NaN)                      # map-grid field, NaN where h < hdeep
const Md = comp_data.M .& (hR .>= comp_data.hdeep)                       # deep comparison area
## phase [deg] without the 0/360 jump: centred on the area-mean phase of zc
unwrap(z, zc) = (gm = rad2deg(angle(sum(zc[Md]))); mod.(-rad2deg.(angle.(z)) .+ gm .+ 180, 360) .- 180 .- gm)
function phase_lev(G...)                                                 # co-phase levels: every 1, 2 or 3 deg
    g = reduce(vcat, [X[Md] for X in G]); g = filter(isfinite, g)
    rng = quantile(g, 0.99) - quantile(g, 0.01)
    dl = rng <= 12 ? 1 : rng <= 24 ? 2 : 3
    return dl, (floor(minimum(g) / dl) * dl):dl:(ceil(maximum(g) / dl) * dl)
end

## number labels on contour lines (Makie's automatic labels skip some lines): for each level, the map point on the
## line at fraction f of the line's latitude range (f cycles through fs, so labels of close lines do not collide),
## phase written mod 360 with a white halo
function label_lines!(a, G, lev; fs = [0.5], color = :black)
    for (i, L) in enumerate(lev)
        f = fs[mod1(i, length(fs))]
        tol = max(0.02, 0.5 * step(lev) * 0.05); on = findall(p -> abs(G[p] - L) < tol, CartesianIndices(G))
        isempty(on) && continue
        la = [latv[p[2]] for p in on]; tgt = quantile(la, f)
        p = on[argmin(abs.(la .- tgt))]
        t = string(round(Int, mod(L, 360)))
        text!(a, lonv[p[1]], latv[p[2]]; text = t, fontsize = 9, align = (:center, :center), color = :white, strokecolor = :white, strokewidth = 2.5)
        text!(a, lonv[p[1]], latv[p[2]]; text = t, fontsize = 9, align = (:center, :center), color = color)
    end
end

## --- figure 2: deep ocean amplitude and phase , black line for roms phase and oragne dash line for fes phase
## amplitude and black lines: ZCP (ROMS alone, low-passed over deep water); orange dashed: FES; h > hdeep only
fig2 = Figure(; figkw(24, 20)...)
for (k, c) in enumerate(comp_data.cons)
    zc, zf = comp_data.ZCP[:, :, k], comp_data.ZF[:, :, k]
    Ar = abs.(zc); Gr = unwrap(zc, zc); Gf = unwrap(zf, zc)
    dl, lev = phase_lev(Gr, Gf)
    r = (k - 1) ÷ 3 + 1; q = 2 * ((k - 1) % 3) + 1                       # panel row, column
    a = Axis(fig2[r, q]; title = @sprintf("%s  phase every %d°", c * (comp_data.inferred[k] ? "*" : ""), dl), mapax...,
             yticklabelsvisible = q == 1, xticklabelsvisible = r == 2 || k == 3)
    h = heatmap!(a, lonv, latv, rg(Ar); colormap = Reverse(:Spectral), nan_color = :white,
                 colorrange = (quantile(filter(isfinite, Ar[Md]), 0.001), quantile(filter(isfinite, Ar[Md]), 0.999)))
    contour!(a, lonv, latv, hmap; levels = BATHY, color = RGBf(0.3, 0.3, 0.3), linewidth = 0.7)
    lines!(a, grid_edge(lonR, latR)...; color = :gray30, linewidth = 0.6)
    Gfm, Grm = deep(rg(Gf)), deep(rg(Gr))
    contour!(a, lonv, latv, Gfm; levels = lev, color = :darkorange, linewidth = 1, linestyle = :dash)
    contour!(a, lonv, latv, Grm; levels = lev, color = :black, linewidth = 1)
    label_lines!(a, Gfm, lev; fs = [0.9, 0.7, 0.5], color = RGBf(0.75, 0.35, 0))   # FES labels in the north part of each line
    label_lines!(a, Grm, lev; fs = [0.35, 0.15])                                   # ROMS labels lower on the line
    limits!(a, first(lonv), last(lonv), first(latv), last(latv))
    Colorbar(fig2[r, q + 1], h; height = Relative(0.75), width = 8)
end
Legend(fig2[2, 5:6], [LineElement(color = :black, linewidth = 1), LineElement(color = :darkorange, linewidth = 1, linestyle = :dash),
                      LineElement(color = RGBf(0.3, 0.3, 0.3), linewidth = 0.7)],
       ["ROMS phase [deg]", "FES2022b phase", "$(join(BATHY, ", ")) m isobaths"]; framevisible = false, labelsize = 10, tellwidth = false)
Label(fig2[0, 1:6], "ROMS Amazon 900 m vs FES2022b: deep-ocean amplitude [cm, colour] and Greenwich phase (h > $(round(Int, comp_data.hdeep)) m)\n* corrected for K2 / NU2 / P1 with the FES ratio", fontsize = 11, tellwidth = false)
colgap!(fig2.layout, 6)
figsave(figdir * "ROMS_FES_deep_cotidal$(STRIDE > 1 ? "_s$STRIDE" : "").png", fig2)

