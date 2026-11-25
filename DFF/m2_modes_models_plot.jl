#= m2_modes_models_plot.jl
Hsin-Yi Chen, USM DMS
M2 internal-tide strength in every model, from the files of m2_modes_models.jl (<SRC>_modes_harmonic[_s<n>].nc, looked
for in the expanse output and in the aurelia one over ~/aurelia). Every model is put on the ROMS 2700 m grid
(bilinear, function/grid_locate.jl) and scored over the COMMON area: 2700 m grid interior (M2P_EDGE km removed along the
edges), h >= M2P_HMIN m (2700 m bathymetry), and every model analysed there.

Quantities per model and mode (M2; another constituent with M2P_CON):
  E = HKE + APE [J m-2] -> area integral [PJ] and area mean; |F| = |(Fx, Fy)| [W m-1] -> area mean
  regions: slope (M2P_HMIN .. 2000 m) and deep (> 2000 m)
Output (expanse: figure/m2_modes_models/):
  m2_modes_summary[_s<n>].txt   table
  m2_modes_energy[_s<n>].png    area-integrated E per model (stacked modes), slope and deep; mean |F|
  m2_modes_flux_maps[_s<n>].png |F| maps with flux arrows, rows = models, columns = modes
  m2_modes_E_maps[_s<n>].png    E maps, rows = models, columns = modes

Usage:
  julia --project=. -t 4 --heap-size-hint=3G DFF/m2_modes_models_plot.jl
  M2P_SRC=ROMS_HYCOMparent,HYCOM M2P_STRIDE=10 julia --project=. DFF/m2_modes_models_plot.jl
=#

using NCDatasets, CairoMakie, Printf, Statistics

const HOSTX = isdir("/expanse/lustre") ? "expanse" : "aurelia"
const FUNC = HOSTX == "expanse" ? "/home/hchen54/internalwave_julia/function/" : "/home/hsinyi/Documents/Julia/function/"
const GRID = HOSTX == "expanse" ? "/expanse/lustre/projects/uso102/hchen54/new_design_fix/roms_grd_2700m_1104.nc" :
                                  "/home/hsinyi/roms_data/new_design_fix/roms_grd_2700m_1104.nc"
const DIRS = HOSTX == "expanse" ? ["/expanse/lustre/projects/uso102/hchen54/figure/m2_modes_models/", "/home/hchen54/aurelia/roms_data/m2_modes_models/"] :
                                  ["/home/hsinyi/roms_data/m2_modes_models/"]
const OUTP = DIRS[1]
include(FUNC * "grid_locate.jl"); include(FUNC * "fig_setup.jl"); include(FUNC * "plotting_fun.jl")

const SRCS   = split(get(ENV, "M2P_SRC", "ROMS_HYCOMparent,ROMS_GLORYSparent,HYCOM,NCOM2700,GLORYS"), ",")
const STRIDE = parse(Int, get(ENV, "M2P_STRIDE", "1"))     # which model files (_s<n>) and target thinning
const SFX    = STRIDE > 1 ? "_s$STRIDE" : ""
const CON    = get(ENV, "M2P_CON", "M2")
const EDGE   = parse(Float64, get(ENV, "M2P_EDGE", "100"))  # km
const HMIN   = parse(Float64, get(ENV, "M2P_HMIN", "200"))  # m
const HDEEP  = 2000.0
lon180(x) = x > 180 ? x - 360 : x

## --- target grid
tg = NCDataset(GRID) do g
    nx, ny = size(g["lon_rho"]); ii = 1:STRIDE:nx; jj = 1:STRIDE:ny
    (lon = Float64.(g["lon_rho"][ii, jj]), lat = Float64.(g["lat_rho"][ii, jj]), h = Float64.(g["h"][ii, jj]),
     mask = g["mask_rho"][ii, jj] .== 1, area = Float64.(STRIDE^2 ./ (g["pm"][ii, jj] .* g["pn"][ii, jj])),
     dx = 1e-3 * mean(1 ./ g["pm"][:, :]), dy = 1e-3 * mean(1 ./ g["pn"][:, :]))
end
NX, NY = size(tg.lon)

## --- read and interpolate each model
D = []
for s in SRCS
    f = nothing
    for d in DIRS; isfile(d * "$(s)_modes_harmonic$(SFX).nc") && (f = d * "$(s)_modes_harmonic$(SFX).nc"; break); end
    f === nothing && (println("no $(s)_modes_harmonic$(SFX).nc, skipped"); continue)
    m = NCDataset(f) do ds
        cons = String.(ds["con"][:]); k = findfirst(==(CON), cons)
        k === nothing && error("$f: no $CON")
        nm = Int(ds.attrib["nmode"])
        rd(v) = Float64.(coalesce.(ds[v][:, :, :, k], NaN))
        (lon = lon180.(Float64.(ds["lon"][:, :])), lat = Float64.(ds["lat"][:, :]), nm = nm, label = ds.attrib["label"],
         E = rd("HKE") .+ rd("APE"), HKE = rd("HKE"), Fx = rd("Fx"), Fy = rd("Fy"), dt = Float64(ds.attrib["dt_hours"]))
    end
    same = size(m.lon) == (NX, NY) && maximum(abs.(m.lon .- lon180.(tg.lon))) < 1e-6
    L = same ? nothing : locator(m.lon, m.lat, lon180.(tg.lon), tg.lat)
    ip(X) = same ? X : bilin(L, X)
    g3(A) = cat([ip(A[:, :, q]) for q in 1:m.nm]...; dims = 3)
    push!(D, (src = s, label = m.label, nm = m.nm, E = g3(m.E), HKE = g3(m.HKE), Fx = g3(m.Fx), Fy = g3(m.Fy), dt = m.dt))
    @printf("%-18s %s (%s, dt %.1f h)\n", s, f, same ? "same grid" : "bilinear", m.dt)
end
isempty(D) && error("no model files found in $(join(DIRS, ", "))")
NM = length(D); NMODE = minimum(d.nm for d in D)

## --- common area
nsx = round(Int, EDGE / (STRIDE * tg.dx)); nsy = round(Int, EDGE / (STRIDE * tg.dy))
inner = falses(NX, NY); inner[nsx+1:NX-nsx, nsy+1:NY-nsy] .= true
MC = inner .& tg.mask .& (tg.h .>= HMIN)
for d in D, q in 1:NMODE; MC .&= isfinite.(d.E[:, :, q]) .& isfinite.(d.Fx[:, :, q]); end
regions = [("slope ($(Int(HMIN))-$(Int(HDEEP)) m)", MC .& (tg.h .< HDEEP)), ("deep (> $(Int(HDEEP)) m)", MC .& (tg.h .>= HDEEP)), ("all (h > $(Int(HMIN)) m)", MC)]
println("common area: $(count(MC)) points, $(round(sum(tg.area[MC]) / 1e6 / 1e3, digits = 1)) x 10^3 km^2")

## --- statistics
io = IOBuffer()
@printf(io, "%s internal tide, vertical modes 1-%d, common area of %s on the ROMS 2700 m grid (%.0f km edge removed, h >= %.0f m): %d points, %.1f x 10^3 km^2\n",
        CON, NMODE, join([d.src for d in D], ", "), EDGE, HMIN, count(MC), sum(tg.area[MC]) / 1e9)
@printf(io, "E = HKE + APE (time mean, depth integrated); Eint = area integral [PJ]; <E> area mean [J/m2]; <|F|> area mean of the flux magnitude [W/m]\n")
stat = Dict()
for (rn, R) in regions
    @printf(io, "\n%s: %d points, %.1f x 10^3 km^2\n", rn, count(R), sum(tg.area[R]) / 1e9)
    @printf(io, "  %-30s", "model"); for q in 1:NMODE; @printf(io, " | mode %d: Eint   <E>  <|F|>", q); end; @printf(io, " | modes 1-%d: Eint  <|F|>  HKE/E\n", NMODE)
    for d in D
        @printf(io, "  %-30s", d.label); Et = 0.0; Kt = 0.0
        for q in 1:NMODE
            E = d.E[:, :, q][R]; a = tg.area[R]; F = hypot.(d.Fx[:, :, q][R], d.Fy[:, :, q][R])
            Ei = sum(E .* a) / 1e15; Em = sum(E .* a) / sum(a); Fm = sum(F .* a) / sum(a)
            stat[(rn, d.src, q)] = (Ei = Ei, Em = Em, Fm = Fm)
            @printf(io, " |     %6.3f %6.0f %6.0f", Ei, Em, Fm)
            Et += Ei; Kt += sum(d.HKE[:, :, q][R] .* a) / 1e15
        end
        Fx = sum(d.Fx[:, :, q] for q in 1:NMODE)[R]; Fy = sum(d.Fy[:, :, q] for q in 1:NMODE)[R]
        Fm = sum(hypot.(Fx, Fy) .* tg.area[R]) / sum(tg.area[R])
        stat[(rn, d.src, 0)] = (Ei = Et, Fm = Fm)
        @printf(io, " |          %6.3f %6.0f  %5.2f\n", Et, Fm, Kt / Et)
    end
end
tab = String(take!(io)); print(tab); mkpath(OUTP); write(OUTP * "m2_modes_summary$(SFX).txt", tab)
println("wrote ", OUTP * "m2_modes_summary$(SFX).txt")

## --- figure 1: energy and flux per model, stacked modes
mcol = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227"][1:NMODE]
fig = Figure(; figkw(18, 14)...)
labs = [replace(d.label, " (" => "\n(") for d in D]
for (c, (rn, R)) in enumerate(regions[1:2])
    ax = Axis(fig[1, c]; title = "$CON energy, $rn", ylabel = "area integral [PJ]", xticks = (1:NM, labs), xticklabelsize = 7, xticklabelrotation = π / 6, titlesize = 10)
    v = [stat[(rn, d.src, q)].Ei for d in D, q in 1:NMODE]
    barplot!(ax, repeat(1:NM, NMODE), vec(v); stack = repeat(1:NMODE, inner = NM), color = repeat(mcol, inner = NM), strokewidth = 0)
    tot = vec(sum(v; dims = 2))
    text!(ax, 1:NM, tot; text = [@sprintf("%.2f", x) for x in tot], align = (:center, :bottom), fontsize = 8, offset = (0, 2))
    ylims!(ax, 0, 1.15 * maximum(tot) + eps())
    ax2 = Axis(fig[2, c]; title = "$CON flux, $rn", ylabel = "area mean |F| [W/m]", xticks = (1:NM, labs), xticklabelsize = 7, xticklabelrotation = π / 6, titlesize = 10)
    w = [stat[(rn, d.src, q)].Fm for d in D, q in 1:NMODE]
    barplot!(ax2, repeat(1:NM, NMODE), vec(w); dodge = repeat(1:NMODE, inner = NM), color = repeat(mcol, inner = NM), strokewidth = 0)
end
Legend(fig[3, 1:2], [PolyElement(color = mcol[q]) for q in 1:NMODE], ["mode $q" for q in 1:NMODE]; orientation = :horizontal, framevisible = false)
Label(fig[0, 1:2], "$CON internal tide from the vertical-mode harmonic fit (E = HKE + APE)\ncommon area on the ROMS 2700 m grid", fontsize = 10, tellwidth = false)
figsave(OUTP * "m2_modes_energy$(SFX).png", fig); println("wrote ", OUTP * "m2_modes_energy$(SFX).png")

## --- maps
mg = map_grid(tg.lon, tg.lat; dll = 0.03 * STRIDE)
lonv, latv, rg = mg.lonv, mg.latv, mg.rg; hmap = rg(tg.h)
mapax = (; aspect = DataAspect(), xtickformat = lonfmt, ytickformat = latfmt, xticks = 308:8:324, yticks = -4:4:12, xticklabelsize = 7, yticklabelsize = 7)
function decor!(a)
    contour!(a, lonv, latv, hmap; levels = [200, 1000, 3000], color = RGBf(0.35, 0.35, 0.35), linewidth = 0.5)
    lines!(a, grid_edge(tg.lon, tg.lat)...; color = :gray30, linewidth = 0.5)
    limits!(a, first(lonv), last(lonv), first(latv), last(latv))
end
qs = 1:min(NMODE, 3)
for (kind, fname) in ((:F, "m2_modes_flux_maps"), (:E, "m2_modes_E_maps"))
    figm = Figure(; figkw(18, 4.4NM + 2.4)...)
    for (c, q) in enumerate(qs)
        X = [kind == :F ? hypot.(d.Fx[:, :, q], d.Fy[:, :, q]) : d.E[:, :, q] for d in D]
        top = quantile(filter(isfinite, reduce(vcat, [x[MC] for x in X])), 0.99)
        local hm
        for (r, d) in enumerate(D)
            a = Axis(figm[r, c]; mapax..., yticklabelsvisible = c == 1, xticklabelsvisible = r == NM,
                     title = r == 1 ? "mode $q" : "", titlesize = 9)
            hm = heatmap!(a, lonv, latv, rg(ifelse.(tg.mask, X[r], NaN)); colormap = :inferno, colorrange = (0, top), highclip = :white, nan_color = RGBf(0.85, 0.85, 0.85))
            decor!(a)
            if kind == :F                                     # arrows on a coarse subset of the target grid
                sk = max(1, round(Int, 40 / STRIDE)); ii = 1:sk:NX; jj = 1:sk:NY
                fx = d.Fx[ii, jj, q]; fy = d.Fy[ii, jj, q]; keep = vec(isfinite.(fx) .& (hypot.(fx, fy) .> 0.05top))
                xs = vec(tg.lon[ii, jj])[keep]; ys = vec(tg.lat[ii, jj])[keep]
                sc = 0.6 / top                                 # deg per W/m (arrow of length 0.6 deg = top)
                any(keep) && arrows2d!(a, xs, ys, sc .* vec(fx)[keep], sc .* vec(fy)[keep]; color = :cyan, shaftwidth = 0.6, tipwidth = 3, tiplength = 3)   # none for a model without tides (GLORYS)
            end
        end
        Colorbar(figm[NM + 1, c], hm; vertical = false, flipaxis = false, height = 7, ticklabelsize = 7,
                 label = kind == :F ? "|F| [W/m]" : "E [J/m²]", labelsize = 8)
    end
    for (r, d) in enumerate(D); Label(figm[r, 0], replace(d.label, " (" => "\n("); rotation = π / 2, tellheight = false, fontsize = 8, font = :bold); end
    Label(figm[-1, 0:length(qs)], kind == :F ? "$CON energy flux per mode (colour |F|, arrows direction; one colour scale per mode)" :
                                     "$CON energy E = HKE + APE per mode (one colour scale per mode)"; fontsize = 9, tellwidth = false)
    colgap!(figm.layout, 4); rowgap!(figm.layout, 4)
    figsave(OUTP * "$(fname)$(SFX).png", figm); println("wrote ", OUTP * "$(fname)$(SFX).png")
end
