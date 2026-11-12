#= DFF_on_harmonic.jl
Hsin-Yi Chen, USM DMS
Directional Fourier filter (DFF) of the harmonic fit of the modal pn, un, vn (mode_bcef_harmonic.jl): the fit has
done the time filter, so each constituent only needs the space filter. For A cos(ω(t - tc) - g), Z = A e^{-ig} is the
positive-frequency slice of the time-series DFF (DFF_on_bcef.jl), so the same wavenumber mask (waxis, wdir: θ = the
direction of travel from the grid xi axis) applied to fft2(Z) splits the constituent by direction (dff_sector_harm).

Two sector schemes, both in grid coordinates (θ from grid xi, counter-clockwise; grid eta = 90°):
    half:  north 0-180, south 180-360                         (as DFF_on_bcef.jl)
    quad:  north 45-135, west 135-225, south 225-315, east 315-45
Flux of each direction from its own p, u, v (one sinusoid: <p w> = ½ Re(P W*)):
    Fx_sec = ½ H Re(P_sec U_sec*),  Fy_sec = ½ H Re(P_sec V_sec*)   [W m-1], grid xi / eta
    cross  = all - Σ sectors (the terms between directions: crossing waves, standing patterns)
Checks: "all" (every wavenumber) returns Z itself; the sectors of a scheme add up to Z; a synthetic plane wave
travelling grid-north ends up in "north".
Output: <bcef_dir>/harm_dff_flux.nc   Fx_<con>_<scheme>_<sector>_mode<n>, Fy_... (xi_rho, eta_rho) on the DFF crop
Test figures (TEST_CONS, default M2): ~/figure_check/harm_dff_<scheme>_<con>_mode<n>.png  (|F| and arrows per sector)
Usage:
  julia DFF/DFF_on_harmonic.jl                       # amazon_2700m_cmems_0801 (ddf_bcef_0824), every constituent
  DFF_RUN=... BCEF_OUT=... julia DFF/DFF_on_harmonic.jl
=#

using NCDatasets, FFTW, Statistics, Printf
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
haskey(ENV, "DFF_RUN") || (ENV["DFF_RUN"] = "amazon_2700m_cmems_0801")
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")    # dff_run, run_dir, run_grid_fname, fig_dir
include("/home/hchen54/internalwave_julia/DFF/dff_fun.jl")    # waxis, wdir, dff_sector_harm

# ============================================================================
# (1) settings, the harmonic fit, grid
# ============================================================================
NMODE = parse(Int, get(ENV, "NMODE", "3"))
default_bcef = Dict("amazon_2700m_cmems_0801" => "ddf_bcef_0824")
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, get(default_bcef, dff_run, "ddf_bcef")))
harm_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_harmonic.nc")
out_fname = joinpath(bcef_dir, "harm_dff_flux.nc")
check_path = expanduser("~/figure_check"); mkpath(check_path)
dsh = NCDataset(harm_fname)
nx0, ny0 = dsh.dim["xi_rho"], dsh.dim["eta_rho"]
nx, ny = isodd(nx0) ? nx0 : nx0 - 1, isodd(ny0) ? ny0 : ny0 - 1     # odd sizes (as DFF_on_bcef.jl)
cons = String.(dsh["con"][:])
modes = parse.(Int, split(dsh.attrib["modes"], ","))
H = Float64.(dsh["H"][1:nx, 1:ny])
lon, lat, land_mask, angle_g, h = NCDataset(run_grid_fname) do g
    Float64.(g["lon_rho"][1:nx, 1:ny]), Float64.(g["lat_rho"][1:nx, 1:ny]), Float64.(g["mask_rho"][1:nx, 1:ny]),
    Float64.(g["angle"][1:nx, 1:ny]), Float64.(g["h"][1:nx, 1:ny])
end
println("$dff_run: $harm_fname, grid $nx x $ny (crop of $nx0 x $ny0), modes $(join(modes, ",")), cons $(join(cons, ","))")

# sector masks in the wavenumber plane, θ in 0..360 [deg] from grid xi, direction of travel
kx, ky = waxis(nx), waxis(ny)
θ = wdir(kx, ky)
schemes = [("half", [("north", 0, 180), ("south", 180, 360)]),
           ("quad", [("north", 45, 135), ("west", 135, 225), ("south", 225, 315), ("east", 315, 45)])]
sec_mask(a1, a2) = a1 < a2 ? (a1 .<= θ .< a2) : ((θ .>= a1) .| (θ .< a2))   # a1 > a2: wraps through 0
mask_all = trues(nx, ny)
for (sch, secs) in schemes   # every wavenumber exactly once (the k = 0 point has θ = 0: in north / east)
    cover = sum(Int.(sec_mask(a1, a2)) for (_, a1, a2) in secs)
    all(cover .== 1) || error("$sch sectors do not split the wavenumber plane once")
end

# complex amplitude Z = A e^{-ig} of field var (p1, u1, ...) at constituent k; 0 on land / NaN (the FFT needs numbers)
function zfield(var, k)
    A = Float64.(nomissing(dsh["$(var)_amp"][1:nx, 1:ny, k], NaN))
    g = Float64.(nomissing(dsh["$(var)_pha"][1:nx, 1:ny, k], NaN))
    Z = A .* cis.(-deg2rad.(g))
    ifelse.(isfinite.(Z) .& (land_mask .> 0), Z, complex(0.0))
end
flux(P, W) = 0.5 .* H .* real.(P .* conj.(W))      # ½ H Re(P W*) [W m-1]

# ============================================================================
# (2) checks: synthetic plane wave, and "all" = Z
# ============================================================================
let
    jj = (1:ny)'
    for (name, Zs, want) in (("grid-north", cis.(-0.2 .* jj) .* ones(nx), "north"), ("grid-south", cis.(0.2 .* jj) .* ones(nx), "south"))
        Zs = complex.(Zs)
        for (sch, secs) in schemes
            e = [sum(abs2, dff_sector_harm(Zs, sec_mask(a1, a2), ones(nx, ny))) for (_, a1, a2) in secs]
            got = secs[argmax(e)][1]
            @printf("check: plane wave travelling %s (g increasing %s), %s scheme -> %s (%.0f%% of the energy)%s\n",
                    name, want == "north" ? "with eta" : "against eta", sch, got, 100maximum(e) / sum(e), got == want ? "" : "  ** WRONG **")
        end
    end
end

# ============================================================================
# (3) DFF of every constituent and mode: flux per sector, cross terms
# ============================================================================
F = Dict{String, Matrix{Float32}}()
deep = (land_mask .> 0) .& (h .>= 1000)
for (k, c) in enumerate(cons), n in modes
    Z = Dict(v => zfield("$v$n", k) for v in ("p", "u", "v"))
    # all: every wavenumber -> Z itself (check)
    Zall = Dict(v => dff_sector_harm(Z[v], mask_all, land_mask) for v in keys(Z))
    err = maximum(abs.(Zall["p"] .- Z["p"])[land_mask .> 0]) / maximum(abs.(Z["p"]))
    err < 1e-8 || @warn "$c mode $n: all-sector p differs from Z by $err (relative)"
    Fx_all, Fy_all = flux(Zall["p"], Zall["u"]), flux(Zall["p"], Zall["v"])
    for (sch, secs) in schemes
        F["Fx_$(c)_$(sch)_all_mode$n"], F["Fy_$(c)_$(sch)_all_mode$n"] = Fx_all, Fy_all
        sx, sy = zeros(nx, ny), zeros(nx, ny)
        zsum = zeros(ComplexF64, nx, ny)
        for (s, a1, a2) in secs
            m = sec_mask(a1, a2)
            Zs = Dict(v => dff_sector_harm(Z[v], m, land_mask) for v in keys(Z))
            fx, fy = flux(Zs["p"], Zs["u"]), flux(Zs["p"], Zs["v"])
            F["Fx_$(c)_$(sch)_$(s)_mode$n"], F["Fy_$(c)_$(sch)_$(s)_mode$n"] = fx, fy
            sx .+= fx; sy .+= fy; zsum .+= Zs["p"]
        end
        F["Fx_$(c)_$(sch)_cross_mode$n"], F["Fy_$(c)_$(sch)_cross_mode$n"] = Fx_all .- sx, Fy_all .- sy
        zerr = maximum(abs.(zsum .- Zall["p"])[land_mask .> 0]) / maximum(abs.(Z["p"]))
        if c == "M2"
            secstr = join([@sprintf("%s %.0f", s, mean(F["Fy_$(c)_$(sch)_$(s)_mode$n"][deep])) for (s, _, _) in secs], ", ")
            @printf("  %s mode %d %s: Σ sectors of p = all (rel. err %.1e); mean Fy over h >= 1000 m: all %.0f, %s, cross %.0f W/m\n",
                    c, n, sch, zerr, mean(Fy_all[deep]), secstr, mean(F["Fy_$(c)_$(sch)_cross_mode$n"][deep]))
        end
    end
end

tmp = out_fname * ".tmp"
isfile(tmp) && rm(tmp)
NCDataset(tmp, "c") do o
    defDim(o, "xi_rho", nx); defDim(o, "eta_rho", ny)
    o.attrib["title"] = "DFF of the harmonic-fit modal flux per constituent, $dff_run"
    o.attrib["method"] = "Z = A e^{-ig} of pn, un, vn (bcef_modes$(NMODE)_harmonic.nc) -> fft2 -> wavenumber sector (θ = direction of travel from grid xi) -> ifft2; F_sec = ½ H Re(P_sec W_sec*); cross = all - Σ sectors"
    o.attrib["schemes"] = "half: north 0-180, south 180-360; quad: north 45-135, west 135-225, south 225-315, east 315-45 (grid coordinates)"
    o.attrib["script"] = "internalwave_julia/DFF/DFF_on_harmonic.jl"
    defVar(o, "lon_rho", lon, ("xi_rho", "eta_rho")); defVar(o, "lat_rho", lat, ("xi_rho", "eta_rho"))
    defVar(o, "H", H, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
    for key in sort(collect(keys(F)))
        defVar(o, key, F[key], ("xi_rho", "eta_rho"); attrib = ["units" => "W m-1"], deflatelevel = 3)
    end
end
mv(tmp, out_fname; force = true)
println("wrote $out_fname ($(length(F)) fields)")

# ============================================================================
# (4) test maps: |F| and arrows of every sector (and the cross term)
# ============================================================================
using CairoMakie
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")
edge = round(Int, 54 / (mean(NCDataset(g -> 1 ./ Float64.(g["pm"][1:nx, 1:ny]), run_grid_fname)) / 1e3))
show = (h .>= 200) .& (land_mask .> 0) .& (edge .< (1:nx) .<= nx - edge) .& (edge .< (1:ny)' .<= ny - edge)
lon180 = ifelse.(lon .> 180, lon .- 360, lon)
to_en(fx, fy) = (fx .* cos.(angle_g) .- fy .* sin.(angle_g), fx .* sin.(angle_g) .+ fy .* cos.(angle_g))
ix_line = round(Int, 0.5 + (nx + 1) / 2)
sb(a) = a[1:2:end, 1:2:end]                         # every 2nd point in the maps (memory: 8 GB on the login node)
lon_s, lat_s = sb(lon180), sb(lat)
for c in String.(split(get(ENV, "TEST_CONS", "M2"), r"[,:; ]+"; keepempty = false)), n in modes, (sch, secs) in schemes
    haskey(F, "Fx_$(c)_$(sch)_all_mode$n") || continue
    panels = [["all"]; first.(secs); ["cross"]]
    fmax = round(quantile(hypot.(F["Fx_$(c)_$(sch)_all_mode$n"], F["Fy_$(c)_$(sch)_all_mode$n"])[show], 0.99) / 1e3; sigdigits = 2)
    np_ = length(panels)
    nrow, ncol = np_ <= 4 ? (1, np_) : (2, cld(np_, 2))       # quad (6 panels): 2 rows x 3 columns
    fig = Figure(size = (430ncol + 80, 470nrow + 90), fontsize = 13)
    for (col, s) in enumerate(panels)
        fx, fy = F["Fx_$(c)_$(sch)_$(s)_mode$n"] ./ 1e3, F["Fy_$(c)_$(sch)_$(s)_mode$n"] ./ 1e3   # [kW m-1]
        fe, fn = to_en(fx, fy)
        rng = s == "all" ? "every direction" : s == "cross" ? "all − Σ sectors" :
              (a = secs[findfirst(==(s), first.(secs))]; "$(a[2])–$(a[3])°")
        r, cc = fldmod1(col, ncol)
        ax = topdown_axis(fig[r, cc]; title = "$s ($rng)", ylabel = cc == 1 ? "Latitude" : "",
                          xlabel = r == nrow ? "Longitude" : "", backgroundcolor = :gray80)
        hm = plot_curvilinear!(ax, lon_s, lat_s, sb(ifelse.(show, hypot.(fx, fy), NaN)); colormap = :plasma, colorrange = (0, fmax))
        quiver_curvilinear!(ax, lon180, lat, ifelse.(show, fe, NaN), ifelse.(show, fn, NaN); skip = 13, lengthscale = 0.8 / fmax,
                            color = (:white, 0.85), shaftwidth = 0.8, tipwidth = 4, tiplength = 4)
        lines!(ax, lon180[ix_line, :], lat[ix_line, :]; color = :white, linestyle = :dash, linewidth = 0.8)
        col == np_ && Colorbar(fig[1:nrow, ncol + 1], hm; label = "|F| [kW m⁻¹]")
    end
    Label(fig[0, :], "$c mode-$n flux by direction, DFF of the harmonic fit ($sch scheme, grid coordinates: θ from grid ξ, η = 90°), $dff_run"; fontsize = 14)
    fname = joinpath(check_path, "harm_dff_$(sch)_$(c)_mode$(n).png")
    save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
    println("saved $fname")
end
close(dsh)
