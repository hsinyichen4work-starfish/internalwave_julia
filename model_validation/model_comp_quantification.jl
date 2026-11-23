# model_comp_quantification.jl
#
# Model vs model comparison of the time-mean surface fields made by meanflow/mean_flow_avg_loop.jl.
# Data: /home/hsinyi/roms_data/halfmonth/<model>_<yyyymmdd>_<yyyymmdd>/<model>_meanflow_avg.nc, each with
#   lon, lat (x, y)  [-180..180]; zeta, u_east, v_north, temp, salt (x, y, bin), NaN on land / outside the data;
#   bin_name ("2022-09", "09-01–15", "09-16–30", ..., "total"), bin_start, bin_end, nrec (records averaged)
# Every model is on its own grid (ROMS curvilinear, parents regular lon/lat, NCOM rotated).
# u_east / v_north are true east / north components in every file (mean_flow_avg_loop.jl rotated the ROMS
# u, v by the grid angle; the parents are east / north already), so they are interpolated as they are.
#
# Every bin in `bins` (Sep–Nov half-months and months) is done in turn by run_bin(bin); its figures and CSVs go to
# figure_path/<bin>/. Then section 6 collects all bins: similarity_all_bins.csv, vector_correlation_all_bins.csv
# and the time series of r, Ê, |ρ|, |g| over the bins (figure_path/timeseries_*.png).
#
# 1. read one bin (bin_sel) of every variable of every model
# 2. interpolate onto the ROMS 2700m grid (roms_grd_2700m_1104.nc); speed = |ū| = √(ū² + v̄²) of the mean
#    velocity, computed on that grid from the interpolated u_east, v_north
# 3. per variable: quick-look maps, and the similarity of every model to HYCOM and to GLORYS
#    (pattern correlation r, centred RMS difference E′, normalised Ê, mean difference B)
# 4. summary of r and Ê over all variables

using NCDatasets, Dates, Statistics, NearestNeighbors
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")   # CairoMakie, topdown_axis, plot_curvilinear!
include("/home/hsinyi/Documents/Julia/function/fig_setup.jl")      # figkw, figsave

datadir     = "/home/hsinyi/roms_data/halfmonth"
grid_file   = "/home/hsinyi/roms_data/new_design_fix/roms_grd_2700m_1104.nc"   # the common grid
figure_path = "/home/hsinyi/figure/20261120_model_comp_quant"

# key (folder / file prefix) => legend label
models = [
    "design_1113_hycom_2700" => "ROMS 2700m (HYCOM parent)",
    "design_1116_cmems_2700" => "ROMS 2700m (GLORYS parent)",
    # "amazon_900m_dbry_0927"  => "ROMS 900m (dbry_0927)",   # left out: its domain is too small for the comparison
    "HYCOM"                  => "HYCOM GOFS 3.5 1/25°",
    "NCOM"                   => "NCOM 2700m",
    "CMEMS"                  => "GLORYS 1/12°",
]
short_of = Dict("design_1113_hycom_2700" => "ROMS H", "design_1116_cmems_2700" => "ROMS G",
                "HYCOM" => "HYCOM", "NCOM" => "NCOM", "CMEMS" => "GLORYS")
refs = ["HYCOM", "CMEMS"]            # reference models for the similarity: HYCOM and GLORYS (CMEMS)

# bins to compare (the dash is an en dash, as in the files); half-months first, then the months
halfmonths = ["09-01–15", "09-16–30", "10-01–15", "10-16–31", "11-01–15", "11-16–30"]
months     = ["2022-09", "2022-10", "2022-11"]
bins       = vcat(halfmonths, months)
vars     = ["temp", "salt", "u_east", "v_north", "speed"]   # also possible: "zeta" (B is then left out)
file_vars = ["temp", "salt", "u_east", "v_north", "zeta"]   # variables stored in the files ("speed" is derived)

# per variable: long name, unit, colormap and range of the maps, range of the spread map
var_info = Dict(
    "temp"    => (name = "surface temperature",   unit = "°C",  cmap = :thermal, crange = (26.0, 30.0),  spread = (0.0, 1.0)),
    "salt"    => (name = "surface salinity",      unit = "psu", cmap = :haline,  crange = (32.5, 36.5),  spread = (0.0, 1.5)),
    "u_east"  => (name = "eastward velocity ū",   unit = "m/s", cmap = :balance, crange = (-1.2, 1.2),   spread = (0.0, 0.4)),
    "v_north" => (name = "northward velocity v̄",  unit = "m/s", cmap = :balance, crange = (-1.2, 1.2),   spread = (0.0, 0.4)),
    "speed"   => (name = "surface speed |ū| (of the mean velocity)", unit = "m/s", cmap = :viridis, crange = (0.0, 1.5), spread = (0.0, 0.4)),
    "zeta"    => (name = "sea-surface height",    unit = "m",   cmap = Reverse(:RdBu), crange = (-0.3, 0.3), spread = (0.0, 0.1)),
)

## ===== 1. read =====
# the model's folder: <key>_<yyyymmdd>_<yyyymmdd> (the time span the run / data cover)
function model_file(key)
    d = filter(n -> occursin(Regex("^\\Q$key\\E_\\d{8}_\\d{8}\$"), n), readdir(datadir))
    length(d) == 1 || error("$key: expected one folder <key>_<start>_<end> in $datadir, found $d")
    return joinpath(datadir, d[1], "$(key)_meanflow_avg.nc")
end

"""
    read_means(key, vars, bin) -> (; lon, lat, fields = Dict(var => (x, y)), nrec, t0, t1)

Mean fields `vars` of model `key` in bin `bin`, NaN where there is no data.
"""
function read_means(key, vars, bin)
    NCDataset(model_file(key)) do ds
        names = ds["bin_name"][:]
        b = findfirst(==(bin), names)
        b === nothing && error("$key: no bin '$bin' (has $(join(names, ", ")))")
        (; lon = Float64.(ds["lon"][:, :]), lat = Float64.(ds["lat"][:, :]),
           fields = Dict(v => Float64.(nomissing(ds[v][:, :, b], NaN)) for v in vars),
           nrec = ds["nrec"][b], t0 = ds["bin_start"][b], t1 = ds["bin_end"][b])
    end
end

# the grids only (no fields), for the interpolation weights; the same in every bin
grids = Dict(key => read_means(key, String[], bins[1]) for (key, _) in models)

## ===== 2. common grid: the ROMS 2700m (1104) grid =====
# The two ROMS 2700m runs are already on it. The parents are interpolated bilinearly:
#   HYCOM, GLORYS: regular lon/lat; NCOM: rotated (curvilinear) grid. One method for all of them:
#   1. the source point nearest to the target point (KD-tree on lon*cos(lat0), lat)
#   2. the fractional source index (fi, fj) of the target from the local grid spacing there
#      (exact on a regular grid, and on NCOM's smooth rotated grid to well below a cell)
#   3. bilinear weights of the 4 surrounding source points; a land (NaN) corner is dropped and the weights of
#      the wet corners rescaled (as in the mooring extraction); NaN if all 4 are land or the target is outside
# The weights depend only on the grids, so they are computed once per model and reused for every field.
tgt = NCDataset(grid_file) do g
    lo = Float64.(g["lon_rho"][:, :]); lo = ifelse.(lo .> 180, lo .- 360, lo)
    (lon = lo, lat = Float64.(g["lat_rho"][:, :]), mask = g["mask_rho"][:, :] .== 1,
     area = 1 ./ (g["pm"][:, :] .* g["pn"][:, :]))
end

"""
    interp_weights(slon, slat, tlon, tlat) -> W

Bilinear interpolation of fields on the source grid (slon, slat) to the points (tlon, tlat), all (x, y) matrices.
W = (; i, j, p, q): lower-left source corner and fractional position in the cell per target point,
i = 0 where the target is outside the source grid.
"""
function interp_weights(slon, slat, tlon, tlat)
    nx, ny = size(slon)
    c0 = cosd(mean(tlat))
    tree = KDTree(vcat(vec(slon)' .* c0, vec(slat)'))
    CI = CartesianIndices(slon)
    I0 = zeros(Int, size(tlon)); J0 = zeros(Int, size(tlon)); P = zeros(size(tlon)); Q = zeros(size(tlon))
    for n in eachindex(tlon)
        k, _ = nn(tree, [tlon[n] * c0, tlat[n]])
        i, j = Tuple(CI[k])
        ip, jp = min(i, nx - 1), min(j, ny - 1)
        # local Jacobian of (lon, lat) with respect to the source index (i, j)
        A = [slon[ip+1, j] - slon[ip, j]  slon[i, jp+1] - slon[i, jp];
             slat[ip+1, j] - slat[ip, j]  slat[i, jp+1] - slat[i, jp]]
        d = A \ [tlon[n] - slon[i, j], tlat[n] - slat[i, j]]
        fi, fj = i + d[1], j + d[2]
        (1 <= fi <= nx && 1 <= fj <= ny) || continue                # outside the source grid
        i0, j0 = min(floor(Int, fi), nx - 1), min(floor(Int, fj), ny - 1)
        I0[n], J0[n], P[n], Q[n] = i0, j0, fi - i0, fj - j0
    end
    return (; i = I0, j = J0, p = P, q = Q)
end

# field F on the source grid -> target grid with the weights W (NaN corners dropped, see above)
function apply_weights(W, F)
    out = fill(NaN, size(W.i))
    for n in eachindex(out)
        i, j = W.i[n], W.j[n]
        i == 0 && continue
        p, q = W.p[n], W.q[n]
        s = ws = 0.0
        for (v, w) in ((F[i, j], (1-p)*(1-q)), (F[i+1, j], p*(1-q)), (F[i, j+1], (1-p)*q), (F[i+1, j+1], p*q))
            isnan(v) && continue
            s += w * v; ws += w
        end
        ws > 0 && (out[n] = s / ws)
    end
    return out
end

# weights per model (the ROMS 2700m runs: nothing to do, checked to be on the target grid)
on_target(m) = size(m.lon) == size(tgt.lon) && maximum(abs.(m.lon .- tgt.lon)) < 1e-6 &&
               maximum(abs.(m.lat .- tgt.lat)) < 1e-6
W = Dict{String,Any}()
for (key, _) in models
    m = grids[key]
    W[key] = on_target(m) ? nothing : (t = time(); w = interp_weights(m.lon, m.lat, tgt.lon, tgt.lat);
                                       println("$key: interpolation weights, $(round(time() - t, digits = 1)) s"); w)
end
# field on the target grid, NaN on ROMS land (mask_rho = 0) and where the model has no data
function to_target(key, F)
    f = W[key] === nothing ? copy(F) : apply_weights(W[key], F)
    f[.!tgt.mask] .= NaN
    return f
end

# Per bin (set by run_bin, globals used by the functions below):
#   bin_sel, outdir (figure_path/<bin>), src[key] (the model's fields in the bin),
#   fld[var][key] = mean field on the ROMS 2700m grid, common[var] = mask of the points where every model has a value
function load_bin!(b)
    global bin_sel = b
    global outdir = joinpath(figure_path, b); mkpath(outdir)
    global src = Dict(key => read_means(key, file_vars, b) for (key, _) in models)
    println("\n========== bin $b ==========")
    for (key, label) in models
        m = src[key]
        println(rpad(label, 28), " nrec ", lpad(m.nrec, 4), "  ", m.t0, " .. ", m.t1)
    end
    global fld = Dict(v => Dict(key => to_target(key, src[key].fields[v]) for (key, _) in models) for v in file_vars)
    fld["speed"] = Dict(key => hypot.(fld["u_east"][key], fld["v_north"][key]) for (key, _) in models)
    # common mask per variable: ROMS wet points where every model has a value; the comparison uses only these
    global common = Dict(v => tgt.mask .& reduce(.&, [.!isnan.(fld[v][key]) for (key, _) in models]) for v in keys(fld))
    println("ROMS 2700m grid $(size(tgt.lon)): $(count(tgt.mask)) wet points")
    for v in vars
        println(rpad(v, 8), ": $(count(common[v])) with every model ($(round(100 * count(common[v]) / count(tgt.mask), digits = 1))%)")
    end
end

## ===== 3a. quick-look maps: every model on the common grid + the spread between them =====
# quiver = true: draw the mean surface velocity (u_east, v_north) as arrows on the model panels (not on the spread).
# The axes are DataAspect (1° lon = 1° lat on paper); at 0–14°N cos(lat) ≥ 0.97, so u, v are used as the
# arrow's lon/lat components directly (direction error < 2°). Every `qskip`-th grid point (2.7 km) is drawn,
# an arrow of 1 m/s is `qscale` degrees long; the 1 m/s key is in the lower-right corner of every panel.
const qskip = 30; const qscale = 1.0
function plot_maps(v; quiver = false)
    info = var_info[v]
    F, C = fld[v], common[v]
    spread = fill(NaN, size(tgt.lon))
    spread[C] = [std([F[k][n] for (k, _) in models]) for n in findall(C)]
    panels = vcat([(F[k], l, info.cmap, info.crange) for (k, l) in models],
                  [(spread, "spread between the models (std)", :amp, info.spread)])
    fig = Figure(; figkw(18, 15)..., figure_padding = (4, 8, 4, 4))
    for (n, (A, title, cmap, crange)) in enumerate(panels)
        r, c = fldmod1(n, 3)
        ax = topdown_axis(fig[r, c]; title = title, titlesize = 9, titlegap = 2,
                          xlabel = r == 2 ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "")
        h = plot_curvilinear!(ax, tgt.lon, tgt.lat, A; colormap = cmap, colorrange = crange, nan_color = :gray80)
        if quiver && n <= length(models)
            k = models[n][1]
            ix = (1:qskip:size(tgt.lon, 1)) .+ qskip ÷ 2 .- 1; iy = (1:qskip:size(tgt.lon, 2)) .+ qskip ÷ 2 .- 1
            ix, iy = ix[ix .<= size(tgt.lon, 1)], iy[iy .<= size(tgt.lon, 2)]
            sub(M) = vec(M[ix, iy])
            xs, ys, us, vs = sub(tgt.lon), sub(tgt.lat), sub(fld["u_east"][k]), sub(fld["v_north"][k])
            keep = .!isnan.(us) .& .!isnan.(vs)
            arrows2d!(ax, xs[keep], ys[keep], us[keep], vs[keep]; lengthscale = qscale, color = :white,
                      shaftwidth = 1.0, tipwidth = 6, tiplength = 6)
            x0, y0 = maximum(tgt.lon) - 4.2, minimum(tgt.lat) + 0.8         # 1 m/s key (over land, lower right)
            arrows2d!(ax, [x0], [y0], [1.0], [0.0]; lengthscale = qscale, color = :black,
                      shaftwidth = 1.0, tipwidth = 6, tiplength = 6)
            text!(ax, x0 + 0.5qscale, y0; text = "1 m/s", align = (:center, :bottom), offset = (0, 4), fontsize = 7)
        end
        c == 3 && Colorbar(fig[r, 4], h; label = n == length(panels) ? "std ($(info.unit))" : "$v ($(info.unit))", width = 8)
        r == 1 && hidexdecorations!(ax; ticks = false, grid = false)
        c > 1 && hideydecorations!(ax; ticks = false, grid = false)
    end
    Label(fig[0, :], "Mean $(info.name) $bin_sel on the ROMS 2700m grid (grey: land / no data)" *
          (quiver ? "\narrows: mean surface velocity (ū, v̄), every $(qskip)th point (≈ $(round(Int, 2.7qskip)) km)" : ""),
          fontsize = 11, tellwidth = false)
    rowgap!(fig.layout, 4); colgap!(fig.layout, 4)
    fname = joinpath(outdir, "$(v)_$(bin_sel)_on_roms2700$(quiver ? "_quiver" : "").png")
    figsave(fname, fig)
    println("saved ", fname)
end

## ===== 3b. similarity of each model to a reference model (HYCOM, GLORYS) =====
# On the common mask, area weighted (w = ROMS cell area 1/(pm pn), normalised to Σw = 1), x̄ = Σ w x, σ² = Σ w (x - x̄)²:
#   r   pattern correlation, spatial means removed: Σ w (x_m - x̄_m)(x_r - x̄_r) / (σ_m σ_r)
#   E′  centred RMS difference: √Σ w [(x_m - x̄_m) - (x_r - x̄_r)]²            (unit of the variable)
#   Ê   symmetric normalised E′: E′ / √[(σ_m² + σ_r²)/2]  (0 = identical patterns, √2 = uncorrelated,
#       does not depend on which of the two is the reference)
#   B   mean difference x̄_m - x̄_r (model - reference); left out for zeta (SSH reference level is arbitrary per model)
# m = the model, r = the reference model.
function similarity(xm, xr, w)
    mm, mr = sum(w .* xm), sum(w .* xr)
    am, ar = xm .- mm, xr .- mr
    σm, σr = sqrt(sum(w .* am .^ 2)), sqrt(sum(w .* ar .^ 2))
    E = sqrt(sum(w .* (am .- ar) .^ 2))
    return (; r = sum(w .* am .* ar) / (σm * σr), E, Ehat = E / sqrt((σm^2 + σr^2) / 2), B = mm - mr, σm, σr)
end

cmp_keys = first.(models)
all_sim_rows, all_vc_rows = String[], String[]     # every bin, for the *_all_bins.csv
all_sim, all_vc = Dict{String,Any}(), Dict{String,Any}()   # bin => sim, VC (for the time series)
# sim[var][metric] = (reference, model) matrix; NaN on the diagonal (a model against itself)
function compute_similarity()
global sim = Dict{String,Dict{String,Matrix{Float64}}}()
csv_rows = String[]
for v in vars
    metrics = v == "zeta" ? ["r", "E", "Ehat"] : ["r", "E", "Ehat", "B"]
    C = common[v]
    w = tgt.area[C] ./ sum(tgt.area[C])
    S = Dict(m => fill(NaN, length(refs), length(cmp_keys)) for m in metrics)
    println("\n$v ($(var_info[v].unit)):")
    for (a, ref) in enumerate(refs), (b, key) in enumerate(cmp_keys)
        key == ref && continue
        s = similarity(fld[v][key][C], fld[v][ref][C], w)
        for m in metrics; S[m][a, b] = getfield(s, Symbol(m)); end
        push!(csv_rows, join((v, bin_sel, ref, key, round(s.r, digits = 4), round(s.E, sigdigits = 4),
                              round(s.Ehat, digits = 4), v == "zeta" ? "" : round(s.B, sigdigits = 4),
                              round(s.σm, sigdigits = 4), round(s.σr, sigdigits = 4)), ","))
        println("  ", rpad("$(short_of[key]) vs $(short_of[ref])", 18), " r = ", rpad(round(s.r, digits = 3), 6),
                " E′ = ", rpad(round(s.E, sigdigits = 3), 7), " Ê = ", rpad(round(s.Ehat, digits = 3), 6),
                v == "zeta" ? "" : " B = $(round(s.B, sigdigits = 3))")
    end
    sim[v] = S
end
csvname = joinpath(outdir, "similarity_$(bin_sel).csv")
open(csvname, "w") do io
    println(io, sim_header)
    foreach(r -> println(io, r), csv_rows)
end
append!(all_sim_rows, csv_rows)
all_sim[bin_sel] = sim
println("\nsaved ", csvname)
end
const sim_header = "var,bin,reference,model,r,E_centred_rms,E_hat,B_model_minus_ref,sigma_model,sigma_ref"

# one figure per variable: one panel per metric, rows = reference, columns = model, value written in each cell
function plot_similarity(v)
    info, S = var_info[v], sim[v]
    metrics = v == "zeta" ? ["r", "E", "Ehat"] : ["r", "E", "Ehat", "B"]
    metric_title = Dict("r" => "pattern correlation r", "E" => "centred RMS diff. E′ ($(info.unit))",
                        "Ehat" => "normalised E′ (Ê)", "B" => "mean diff. B = model − ref ($(info.unit))")
    fig = Figure(; figkw(18, 4.2 * cld(length(metrics), 2) + 1.5)..., figure_padding = (4, 8, 4, 4))
    for (n, m) in enumerate(metrics)
        M = S[m]
        good = filter(!isnan, M)
        cmap, crange = m == "r"    ? (:viridis, (min(0.5, floor(minimum(good), digits = 1)), 1.0)) :
                       m == "Ehat" ? (Reverse(:viridis), (0.0, max(1.0, ceil(maximum(good), digits = 1)))) :
                       m == "E"    ? (Reverse(:viridis), (0.0, ceil(maximum(good), sigdigits = 2))) :
                                     (:balance, (-1, 1) .* ceil(maximum(abs.(good)), sigdigits = 2))
        r_, c_ = fldmod1(n, 2)
        ax = Axis(fig[r_, 2c_-1]; title = metric_title[m], titlesize = 9, titlegap = 2, yreversed = true,
                  xticks = (1:length(cmp_keys), [replace(short_of[k], " " => "\n") for k in cmp_keys]), xticklabelsize = 8,
                  yticks = (1:length(refs), ["ref " * short_of[k] for k in refs]), yticklabelsize = 8)
        hm = heatmap!(ax, 1:length(cmp_keys), 1:length(refs), M'; colormap = cmap, colorrange = crange,
                      nan_color = :grey90)
        for a in axes(M, 1), b in axes(M, 2)
            isnan(M[a, b]) && (text!(ax, b, a; text = "—", align = (:center, :center), fontsize = 8); continue)
            frac = (M[a, b] - crange[1]) / (crange[2] - crange[1])
            dark = m == "B" ? abs(frac - 0.5) > 0.3 : (m == "r" ? frac < 0.5 : frac > 0.5)
            text!(ax, b, a; text = string(round(M[a, b], sigdigits = m == "r" ? 3 : 2)), align = (:center, :center),
                  fontsize = 8, color = dark ? :white : :black)
        end
        Colorbar(fig[r_, 2c_], hm; width = 7)
    end
    Label(fig[0, :], "Similarity of the mean $(info.name) ($bin_sel) to HYCOM and to GLORYS\n" *
          "ROMS 2700m grid, area weighted; ROMS H / G: ROMS 2700m with HYCOM / GLORYS parent",
          fontsize = 10, tellwidth = false)
    # spatial std σ_o of each reference (area weighted, common mask): the scale for E′. A model with the same σ
    # gives E′ = 0 if identical, √2 σ_o if uncorrelated (r = 0), 2 σ_o if anti-correlated (r = -1)
    C = common[v]; w = tgt.area[C] ./ sum(tgt.area[C])
    fmt(x) = string(round(x, sigdigits = 3))
    σtxt = join(["ref $(short_of[ref]): σ_o = $(fmt(σo)), √2 σ_o = $(fmt(sqrt(2) * σo)), 2 σ_o = $(fmt(2σo)) $(info.unit)"
                 for ref in refs for σo in (sqrt(sum(w .* (fld[v][ref][C] .- sum(w .* fld[v][ref][C])) .^ 2)),)], "\n")
    Label(fig[cld(length(metrics), 2) + 1, :], σtxt * "\nσ_o: spatial std of the reference; E′ = √2 σ_o for an " *
          "uncorrelated field of the same σ, 2 σ_o for an anti-correlated one",
          fontsize = 8, halign = :left, justification = :left, tellwidth = false)
    rowgap!(fig.layout, 6); colgap!(fig.layout, 4)
    fname = joinpath(outdir, "similarity_$(v)_$(bin_sel).png")
    figsave(fname, fig)
    println("saved ", fname)
end

## ===== 4. summary over the variables: r and Ê of every comparison =====
# rows = comparison (model vs reference), columns = variable
function plot_summary()
    pairs = [(ref, key) for ref in refs for key in cmp_keys if key != ref]
    rowlab = ["$(short_of[k]) vs $(short_of[r])" for (r, k) in pairs]
    fig = Figure(; figkw(18, 0.75 * length(pairs) + 3.5)..., figure_padding = (4, 8, 4, 4))
    for (c, (m, title, cmap, crange)) in enumerate((("r", "pattern correlation r", :viridis, (0.5, 1.0)),
                                                    ("Ehat", "normalised centred RMS difference Ê", Reverse(:viridis), (0.0, 1.0))))
        M = [sim[v][m][findfirst(==(r), refs), findfirst(==(k), cmp_keys)] for (r, k) in pairs, v in vars]
        ax = Axis(fig[1, 2c-1]; title = title, titlesize = 9, titlegap = 2, yreversed = true,
                  xticks = (1:length(vars), vars), xticklabelsize = 8,
                  yticks = (1:length(pairs), c == 1 ? rowlab : fill("", length(pairs))), yticklabelsize = 8)
        hm = heatmap!(ax, 1:length(vars), 1:length(pairs), M'; colormap = cmap, colorrange = crange,
                      lowclip = :black, highclip = :black)
        for i in axes(M, 1), j in axes(M, 2)
            frac = clamp((M[i, j] - crange[1]) / (crange[2] - crange[1]), 0, 1)
            dark = m == "r" ? frac < 0.5 : frac > 0.5
            text!(ax, j, i; text = string(round(M[i, j], digits = 2)), align = (:center, :center), fontsize = 7,
                  color = dark ? :white : :black)
        end
        hlines!(ax, [length(cmp_keys) - 0.5]; color = :white, linewidth = 2)   # HYCOM | GLORYS reference
        Colorbar(fig[1, 2c], hm; width = 7)
    end
    Label(fig[0, :], "Similarity to HYCOM (top 4 rows) and to GLORYS (bottom 4 rows)\n" *
          "mean surface fields $bin_sel on the ROMS 2700m grid, area weighted\n" *
          "ROMS H / G: ROMS 2700m with HYCOM / GLORYS parent; black: outside the colour range",
          fontsize = 9, tellwidth = false)
    colgap!(fig.layout, 4)
    fname = joinpath(outdir, "similarity_summary_$(bin_sel).png")
    figsave(fname, fig)
    println("saved ", fname)
end

## ===== 5. complex (vector) correlation of the mean surface velocity (Kundu 1976) =====
# Each velocity field as a complex number w = u_east + i v_north; a = the model, b = the reference (HYCOM, GLORYS),
# ⟨·⟩ = area-weighted mean over the common velocity mask (as above). The means are NOT removed (as in the
# definition), so the large-scale mean flow counts, and strong currents weigh more than weak ones.
#   ρ = ⟨a b*⟩ / √(⟨|a|²⟩⟨|b|²⟩)
#   |ρ|      0..1, similarity of the two flow patterns allowing one rotation by the same angle everywhere
#   θ = arg ρ  average angle of the model's flow counterclockwise from the reference's (°)
#   Re ρ = |ρ| cos θ  similarity with no rotation allowed (the R of a Taylor diagram)
#   g = ⟨a b*⟩ / ⟨|b|²⟩  complex gain (regression slope for vectors): |g| = amplitude ratio, arg g = θ
function vector_corr(ua, va, ub, vb, w)
    a, b = complex.(ua, va), complex.(ub, vb)
    ab = sum(w .* a .* conj.(b))
    aa, bb = sum(w .* abs2.(a)), sum(w .* abs2.(b))
    ρ = ab / sqrt(aa * bb)
    g = ab / bb
    return (; absρ = abs(ρ), θ = rad2deg(angle(ρ)), reρ = real(ρ), absg = abs(g))
end

vc_metrics = ["absρ", "θ", "reρ", "absg"]
vc_title = Dict("absρ" => "|ρ|, rotation allowed", "θ" => "θ = arg ρ (°, model ccw from ref)",
                "reρ" => "Re ρ = |ρ| cos θ, no rotation", "absg" => "|g|, amplitude ratio model / ref")
const vc_header = "bin,reference,model,abs_rho,theta_deg_ccw,re_rho,abs_gain"
function compute_vc()
    global VC = Dict(m => fill(NaN, length(refs), length(cmp_keys)) for m in vc_metrics)
    C = common["u_east"]; w = tgt.area[C] ./ sum(tgt.area[C])
    vc_rows = String[]
    println("\ncomplex velocity correlation (Kundu 1976):")
    for (a, ref) in enumerate(refs), (b, key) in enumerate(cmp_keys)
        key == ref && continue
        s = vector_corr(fld["u_east"][key][C], fld["v_north"][key][C], fld["u_east"][ref][C], fld["v_north"][ref][C], w)
        for m in vc_metrics; VC[m][a, b] = getfield(s, Symbol(m)); end
        push!(vc_rows, join((bin_sel, ref, key, round(s.absρ, digits = 4), round(s.θ, digits = 2),
                             round(s.reρ, digits = 4), round(s.absg, digits = 4)), ","))
        println("  ", rpad("$(short_of[key]) vs $(short_of[ref])", 18), " |ρ| = ", rpad(round(s.absρ, digits = 3), 6),
                " θ = ", rpad(round(s.θ, digits = 1), 6), "°  Re ρ = ", rpad(round(s.reρ, digits = 3), 6),
                " |g| = ", round(s.absg, digits = 3))
    end
    csvname = joinpath(outdir, "vector_correlation_$(bin_sel).csv")
    open(csvname, "w") do io
        println(io, vc_header)
        foreach(r -> println(io, r), vc_rows)
    end
    append!(all_vc_rows, vc_rows)
    all_vc[bin_sel] = VC
    println("saved ", csvname)
end

function plot_vc()
    fig = Figure(; figkw(18, 4.2 * 2 + 2.5)..., figure_padding = (4, 8, 4, 4))
    for (n, m) in enumerate(vc_metrics)
        M = VC[m]
        good = filter(!isnan, M)
        cmap, crange = m == "θ"    ? (:balance, (-1, 1) .* max(5.0, ceil(maximum(abs.(good))))) :
                       m == "absg" ? (:balance, (1 - max(0.2, maximum(abs.(good .- 1))), 1 + max(0.2, maximum(abs.(good .- 1))))) :
                                     (:viridis, (min(0.5, floor(minimum(good), digits = 1)), 1.0))
        r_, c_ = fldmod1(n, 2)
        ax = Axis(fig[r_, 2c_-1]; title = vc_title[m], titlesize = 9, titlegap = 2, yreversed = true,
                  xticks = (1:length(cmp_keys), [replace(short_of[k], " " => "\n") for k in cmp_keys]), xticklabelsize = 8,
                  yticks = (1:length(refs), ["ref " * short_of[k] for k in refs]), yticklabelsize = 8)
        hm = heatmap!(ax, 1:length(cmp_keys), 1:length(refs), M'; colormap = cmap, colorrange = crange, nan_color = :grey90)
        for a in axes(M, 1), b in axes(M, 2)
            isnan(M[a, b]) && (text!(ax, b, a; text = "—", align = (:center, :center), fontsize = 8); continue)
            frac = (M[a, b] - crange[1]) / (crange[2] - crange[1])
            dark = m in ("θ", "absg") ? abs(frac - 0.5) > 0.3 : frac < 0.5
            text!(ax, b, a; text = m == "θ" ? "$(round(M[a, b], digits = 1))°" : string(round(M[a, b], digits = 3)),
                  align = (:center, :center), fontsize = 8, color = dark ? :white : :black)
        end
        Colorbar(fig[r_, 2c_], hm; width = 7)
    end
    Label(fig[0, :], "Complex (vector) correlation of the mean surface velocity ($bin_sel) with HYCOM and with GLORYS\n" *
          "ROMS 2700m grid, area weighted; ROMS H / G: ROMS 2700m with HYCOM / GLORYS parent", fontsize = 10, tellwidth = false)
    Label(fig[3, :], "w = u + iv, ρ = ⟨a b*⟩ / √(⟨|a|²⟩⟨|b|²⟩), a = model, b = reference, means not removed (Kundu 1976)\n" *
          "θ > 0: the model's flow is rotated counterclockwise from the reference's; g = ⟨a b*⟩ / ⟨|b|²⟩, arg g = θ",
          fontsize = 8, halign = :left, justification = :left, tellwidth = false)
    rowgap!(fig.layout, 6); colgap!(fig.layout, 4)
    fname = joinpath(outdir, "vector_correlation_$(bin_sel).png")
    figsave(fname, fig)
    println("saved ", fname)
end

## ===== run every bin =====
function run_bin(b)
    load_bin!(b)
    compute_similarity()
    for v in vars
        plot_maps(v)
        plot_similarity(v)
    end
    plot_maps("speed"; quiver = true)    # the speed maps again, with the mean velocity arrows
    plot_summary()
    compute_vc()
    plot_vc()
end
foreach(run_bin, bins)

## ===== 6. all bins together =====
for (name, header, rows) in (("similarity_all_bins.csv", sim_header, all_sim_rows),
                             ("vector_correlation_all_bins.csv", vc_header, all_vc_rows))
    fname = joinpath(figure_path, name)
    open(io -> (println(io, header); foreach(r -> println(io, r), rows)), fname, "w")
    println("saved ", fname)
end

# time series over the bins: x = the 6 half-months (dots + solid line), the monthly value as a dashed bar over its
# two half-months; one colour per model, columns = reference (HYCOM, GLORYS), rows = variable (or vector metric)
model_color = Dict(k => Makie.wong_colors()[n] for (n, k) in enumerate(cmp_keys))
month_of = Dict("2022-09" => 1:2, "2022-10" => 3:4, "2022-11" => 5:6)   # its half-months (x positions)
function plot_timeseries(getM, rowkeys, rowlabel, title, fname; ylims = nothing)
    fig = Figure(; figkw(18, 3.4 * length(rowkeys) + 2.5)..., figure_padding = (4, 8, 4, 4))
    axs = Matrix{Axis}(undef, length(rowkeys), length(refs))
    for (i, rk) in enumerate(rowkeys), (a, ref) in enumerate(refs)
        ax = axs[i, a] = Axis(fig[i, a]; title = i == 1 ? "reference $(short_of[ref])" : "", titlesize = 10,
                  ylabel = a == 1 ? rowlabel(rk) : "", ylabelsize = 9, yticklabelsize = 8, xticklabelsize = 8,
                  xticks = (1:length(halfmonths), i == length(rowkeys) ? replace.(halfmonths, "–" => "–\n") : fill("", length(halfmonths))))
        for (b, key) in enumerate(cmp_keys)
            key == ref && continue
            y = [getM(all_sim, all_vc, hb, rk)[a, b] for hb in halfmonths]
            scatterlines!(ax, 1:length(halfmonths), y; color = model_color[key], markersize = 6, linewidth = 1.5,
                          label = short_of[key])
            for mo in months
                ym = getM(all_sim, all_vc, mo, rk)[a, b]
                lines!(ax, [first(month_of[mo]) - 0.3, last(month_of[mo]) + 0.3], [ym, ym];
                       color = model_color[key], linestyle = :dash, linewidth = 1)
            end
        end
        ylims === nothing || ylims!(ax, ylims(rk)...)
        a == 2 && hideydecorations!(ax; ticks = false, ticklabels = false, grid = false)
    end
    Label(fig[0, :], title * "\ndots: half-month means; dashed: monthly mean over its two half-months\n" *
          "ROMS H / G: ROMS 2700m with HYCOM / GLORYS parent", fontsize = 10, tellwidth = false)
    Legend(fig[length(rowkeys) + 1, :],
           [[LineElement(color = model_color[k], linewidth = 1.5), MarkerElement(color = model_color[k], marker = :circle, markersize = 6)]
            for k in cmp_keys], [short_of[k] for k in cmp_keys];
           orientation = :horizontal, labelsize = 8, patchsize = (14, 6), framevisible = false, tellwidth = false)
    foreach(i -> linkyaxes!(axs[i, :]...), axes(axs, 1))   # the two references of a row share the y axis
    rowgap!(fig.layout, 6); colgap!(fig.layout, 6)
    fname = joinpath(figure_path, fname)
    figsave(fname, fig)
    println("saved ", fname)
end


unit_of(v) = var_info[v].unit
plot_timeseries((S, V, b, v) -> S[b][v]["r"], vars, v -> "r  ($v)", "Pattern correlation r with HYCOM and with GLORYS\nmean surface fields on the ROMS 2700m grid, Sep–Nov 2022",
                "timeseries_r.png")
plot_timeseries((S, V, b, v) -> S[b][v]["Ehat"], vars, v -> "Ê  ($v)", "Normalised centred RMS difference Ê with HYCOM and with GLORYS\nmean surface fields on the ROMS 2700m grid, Sep–Nov 2022",
                "timeseries_Ehat.png")
plot_timeseries((S, V, b, v) -> S[b][v]["E"], vars, v -> "E′ ($v, $(unit_of(v)))", "Centred RMS difference E′ with HYCOM and with GLORYS\nmean surface fields on the ROMS 2700m grid, Sep–Nov 2022",
                "timeseries_E.png")
plot_timeseries((S, V, b, v) -> S[b][v]["B"], vars, v -> "B ($v, $(unit_of(v)))", "Mean difference B = model − reference (HYCOM, GLORYS)\nmean surface fields on the ROMS 2700m grid, Sep–Nov 2022",
                "timeseries_B.png")
plot_timeseries((S, V, b, m) -> V[b][m], ["absρ", "θ", "absg"],
                m -> Dict("absρ" => "|ρ|", "θ" => "θ (°, ccw)", "absg" => "|g|")[m],
                "Complex correlation of the mean surface velocity (Kundu 1976) with HYCOM and with GLORYS\nmean surface fields on the ROMS 2700m grid, Sep–Nov 2022",
                "timeseries_vector_correlation.png")



###========mooring loc / transect comp=============
