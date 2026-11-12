using NCDatasets, Dates, Statistics, FFTW

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")   # dff_run, run_dir, run_grid_fname (set by DFF_RUN)
include("/home/hchen54/internalwave_julia/DFF/dff_fun.jl")   # waxis, wdir, dff_sector, dff_spectrum

# Baroclinic energy flux of modes 1..NMODE split into grid-north and grid-south going parts
# (the mode-1-only version, reading bcef_mode1_puv.nc, is DFF_on_bcef_mode1.jl).
# The flux is a product (p'u'), so it is not filtered itself: the mode-n pressure and velocity pn, un, vn (linear in
# the waves, made by bcef_modes_puv.jl) are each split by direction with the DFF, exactly as the ssh in DDF_on_ssh.jl,
# and the flux of each direction is formed from its own parts:
#   F_N,n = H pn_N (un_N, vn_N),   F_S,n = H pn_S (un_S, vn_S)        [W m-1], H = h + mean zeta
# The full mode-n flux H pn (un, vn) = F_N,n + F_S,n + cross terms H (pn_N un_S + pn_S un_N). The cross terms come
# from the north and south waves crossing (a standing pattern): they average out over a record where the two are not
# phase-locked, but not where they are (e.g. near a reflecting slope); they are saved as a check.
# The modes are orthogonal, (1/H) ∫ φm φn dz = δmn, so the depth-integrated flux has no cross-mode terms:
# ∫ p'u' dz = Σn H pn un, and the "modesum" variables (Σ over n of each term) are the flux of modes 1..NMODE together.
# One mode at a time (its 3 spectra + 3 filtered fields: ~40 GB for the 3-month run), so memory does not grow with NMODE.
# Outputs: bcef_dff_flux_modes<N>.nc (time-mean maps per period); hourly along the three columns west / mooring / east
# (bcef_dff_flux_modes<N>_transect_<line>_xi<ix>.nc); hourly mean across the ±STRIP_KM strip of mode_dff_binplot.jl
# (bcef_dff_flux_modes<N>_transect_strip_xi<a>-<b>.nc, STRIP_KM default 200, a distance: its columns follow the grid).
FFTW.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))
NMODE = parse(Int, get(ENV, "NMODE", "3"))
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, "ddf_bcef"))
puv_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_puv.nc")
struct_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc")
out_fname = joinpath(bcef_dir, "bcef_dff_flux_modes$(NMODE).nc")
for f in (puv_fname, struct_fname)
    isfile(f) || error("no $f: run bcef_modes_puv.jl with NMODE = $NMODE first")
end

# ============================================================================
# (1) sizes, depth and grid, trimmed to odd sizes like the ssh
# ============================================================================
ds_puv = NCDataset(puv_fname)
nx, ny, nt = ds_puv.dim["xi_rho"], ds_puv.dim["eta_rho"], ds_puv.dim["time"]
ds_puv.dim["mode"] == NMODE || error("$puv_fname has $(ds_puv.dim["mode"]) modes, NMODE = $NMODE")
nx_f, ny_f = isodd(nx) ? nx : nx - 1, isodd(ny) ? ny : ny - 1
date = ds_puv["ocean_time"][:]   # DateTime
H, cn = NCDataset(struct_fname) do ds
    Float32.(ds["h"][1:nx_f, 1:ny_f] .+ ds["zeta_mean"][1:nx_f, 1:ny_f]), Float32.(ds["c"][1:nx_f, 1:ny_f, :])
end
grid_fname = run_grid_fname   # from dff_run.jl
lon_rho, lat_rho, land_mask, pm = NCDataset(grid_fname) do ds
    ds["lon_rho"][1:nx_f, 1:ny_f], ds["lat_rho"][1:nx_f, 1:ny_f], ds["mask_rho"][1:nx_f, 1:ny_f], Float64.(ds["pm"][1:nx_f, 1:ny_f])
end
lon_rho[lon_rho .> 180] .-= 360
println("$dff_run: $NMODE modes, pn, un, vn $nx x $ny x $nt -> trimmed to $nx_f x $ny_f, $(date[1]) -> $(date[end])")

# ============================================================================
# (2) directions, frequency bands, periods, lines
# ============================================================================
# sectors as for the ssh (θ from the grid xi axis, direction of travel; half-open); "all" keeps every direction, so
# its flux is the full mode-n flux of the band, for the check F_all = F_N + F_S + cross
kx, ky = waxis(nx_f), waxis(ny_f)
θ = wdir(kx, ky)
# The small sectors (60° around grid north / south) are only a check, so they are off by default:
# BCEF_SMALLSEC=true adds them (2 more sectors per mode and band: 5 instead of 3, ~2/3 more run time).
do_smallsec = get(ENV, "BCEF_SMALLSEC", "false") == "true"
sectors = [("all", 0, 360), ("north", 0, 180), ("south", 180, 360)]
do_smallsec && append!(sectors, [("north_smallsec", 60, 120), ("south_smallsec", 240, 300)])
println("sectors: ", join(first.(sectors), ", "), do_smallsec ? "" : " (small sectors off; BCEF_SMALLSEC=true to add them)")
sec_desc = Dict(s => (s == "all" ? "all directions (full flux of the band)" : "$s sector ($a1-$a2 deg from grid xi)")
                for (s, a1, a2) in sectors)
sec_desc["cross"] = "cross terms H (pn_N un_S + pn_S un_N) of the half-plane sectors = all - north - south"
# The time mean (frequency 0) is not in any band, so no plane or mean is removed first.
# bands as the saved flux pieces (energyflux_dum): M2 = periods 9–15 h; hp24 = periods shorter than 24 h; hp9 = periods
# shorter than 9 h (M4 and higher harmonics, nonlinear internal waves). To add a band, add a line. Sharp in
# frequency here (FFT bins), Butterworth in the flux pieces.
nf = cld(nt, 2)   # frequencies kept by dff_spectrum
freq_cph = (0:nf-1) ./ (nt * Dates.value(date[2] - date[1]) / 3.6e6)   # [cycles per hour]
bands = [("M2", (1 / 15 .<= freq_cph .<= 1 / 9), "M2 band, periods 9-15 h"),
         ("hp24", (freq_cph .> 1 / 24), "periods shorter than 24 h"),
         ("hp9", (freq_cph .> 1 / 9), "periods shorter than 9 h")]
for (b, fk, d) in bands
    println("band $b ($d): $(count(fk)) frequencies")
end

# time ranges for the means: the whole record and each month (Aug–Sep together, as in the ssh analysis)
chunk(d1, d2) = (a = findfirst(>=(DateTime(d1)), date); b = findlast(<(DateTime(d2)), date);
                 a === nothing || b === nothing || b <= a ? nothing : a:b)
periods = filter(x -> x[2] !== nothing,
                 [("full", 1:nt), ("augsep", chunk(Date(2022, 8, 1), Date(2022, 10, 1))),
                  ("oct", chunk(Date(2022, 10, 1), Date(2022, 11, 1))), ("nov", chunk(Date(2022, 11, 1), Date(2022, 12, 1)))])
println("periods: ", join(["$p $(date[t[1]]) -> $(date[t[end]])" for (p, t) in periods], ", "))

# the three Hovmöller lines of the ssh analysis (grid columns cutting the domain into 4 along xi): the flux along them
# is kept every hour
tr_lines = [("west", round(Int, 0.5 + nx_f / 4)), ("mooring", round(Int, 0.5 + nx_f / 2)), ("east", round(Int, 0.5 + 3nx_f / 4))]

# the strip of mode_dff_binplot.jl: ±STRIP_KM (default 200 km, a fixed distance: the number of grid columns follows the
# grid spacing) around the mooring line, centred on the same column as mode_analysis.jl / mode_dff_binplot.jl. Its mean
# flux per grid row is kept every hour (H pn un formed at each point, then averaged over the strip's water points
# deeper than strip_hmin, so the shelf is left out); strip_cover = the share of the strip's columns in that mean.
strip_km = parse(Float64, get(ENV, "STRIP_KM", "200"))
strip_hmin = 200.0f0                                               # [m] shelf left out, as mode_dff_binplot.jl
ix_strip_c = round(Int, 0.5 + (nx_f + 1) / 2)
nside = round(Int, strip_km / (mean(1 ./ pm) / 1e3))
strip_cols = max(1, ix_strip_c - nside):min(nx_f, ix_strip_c + nside)
strip_ok = (land_mask[strip_cols, :] .> 0) .& (H[strip_cols, :] .>= strip_hmin)   # (strip column, eta)
strip_n = vec(sum(strip_ok; dims = 1))                             # points averaged per row
strip_cover = Float32.(strip_n ./ length(strip_cols))
println("strip: ±$(strip_km) km = ±$nside columns around xi = $ix_strip_c -> xi = $(first(strip_cols))-$(last(strip_cols)); ",
        "rows with any deep water: $(count(>(0), strip_n)) of $ny_f")
# strip mean over the deep water points of H a b (a, b: (x, y, t)) per row and hour, without forming H a b over the
# whole grid (one (eta, time) slice per column)
# (the globals are passed as arguments, so the loop compiles to typed code)
strip_mean(a, b) = strip_mean_k(a, b, H, strip_cols, strip_ok, strip_n)
function strip_mean_k(a, b, H, cols, ok, n)
    ny_, nt_ = size(a, 2), size(a, 3)
    acc = zeros(Float32, ny_, nt_)
    @inbounds for t in 1:nt_, j in 1:ny_               # inner loop along xi: contiguous in memory
        s = 0f0
        for (k, i) in enumerate(cols)
            ok[k, j] && (s += H[i, j] * a[i, j, t] * b[i, j, t])
        end
        acc[j, t] = s / max(n[j], 1)                   # rows without deep water: 0 (cover = 0)
    end
    acc
end

# ============================================================================
# (3) per mode: spectra of pn, un, vn; per band and sector: the fields of that direction -> flux -> means and lines
# ============================================================================
# The half-plane sectors split every wavenumber exactly once, so pn_all = pn_N + pn_S (same for un, vn) and the cross
# terms are F_all − F_N − F_S; that split is checked at one point.
mtag(n) = "mode$n"
Fmean = Dict{Tuple{String, String, String, String}, Array{Float32, 3}}()      # (mode, band, sector, "Fx"/"Fy") => [xi, eta, period]
Fline = Dict{Tuple{String, String, String, String, String}, Matrix{Float32}}()   # (mode, band, sector, comp, line) => [eta, time]
ixc, iyc = cld(nx_f, 2), cld(ny_f, 2)   # check point (domain centre)

function flux_means(p, u, v)
    Fx = zeros(Float32, nx_f, ny_f, length(periods))
    Fy = zeros(Float32, nx_f, ny_f, length(periods))
    for (ip, (_, tidx)) in enumerate(periods)
        for t in tidx
            @views Fx[:, :, ip] .+= H .* p[:, :, t] .* u[:, :, t]
            @views Fy[:, :, ip] .+= H .* p[:, :, t] .* v[:, :, t]
        end
        Fx[:, :, ip] ./= length(tidx)
        Fy[:, :, ip] ./= length(tidx)
    end
    Fx, Fy
end

for n in 1:NMODE
    m = mtag(n)
    println("\n==== mode $n ====")
    spec = Dict{String, Array{ComplexF32, 3}}()
    for name in ("p$n", "u$n", "v$n")
        a = nomissing(ds_puv[name][1:nx_f, 1:ny_f, :], NaN32)
        any(!isfinite, a) && error("$name has NaN/Inf")
        spec[name] = dff_spectrum(a)
        a = nothing
        GC.gc()
        println("$name: spectrum $(size(spec[name]))")
    end
    size(spec["p$n"], 3) == nf || error("spectrum has $(size(spec["p$n"], 3)) frequencies, expected $nf")

    for (b, fkeep, bdesc) in bands
        pcheck = Dict{String, Vector{Float32}}()   # pn at the check point, per sector
        for (sec, a1, a2) in sectors
            mask = a1 .<= θ .< a2
            @time p, u, v = (dff_sector(spec[f], mask, nt, land_mask; fkeep = fkeep) for f in ("p$n", "u$n", "v$n"))
            Fx, Fy = flux_means(p, u, v)
            Fmean[(m, b, sec, "Fx")], Fmean[(m, b, sec, "Fy")] = Fx, Fy
            for (l, ix) in tr_lines
                Fline[(m, b, sec, "Fx", l)] = H[ix, :] .* p[ix, :, :] .* u[ix, :, :]
                Fline[(m, b, sec, "Fy", l)] = H[ix, :] .* p[ix, :, :] .* v[ix, :, :]
            end
            Fline[(m, b, sec, "Fx", "strip")] = strip_mean(p, u)   # strip mean, every hour (eta, time)
            Fline[(m, b, sec, "Fy", "strip")] = strip_mean(p, v)
            println("mode $n $b $sec: domain-mean flux (full record) Fx $(round(mean(Fx[:, :, 1]), sigdigits = 3)), ",
                    "Fy $(round(mean(Fy[:, :, 1]), sigdigits = 3)) W/m; p$n std $(round(std(p), sigdigits = 3)) Pa")
            pcheck[sec] = p[ixc, iyc, :]
            p = u = v = nothing
            GC.gc()
        end
        # check: north + south = all at the check point; then the cross terms
        err = maximum(abs, pcheck["north"] .+ pcheck["south"] .- pcheck["all"])
        println("mode $n $b check at (xi, eta) = ($ixc, $iyc): max |p$(n)_N + p$(n)_S − p$(n)_all| = $(round(err, sigdigits = 2)) Pa ",
                "(p$(n)_all std $(round(std(pcheck["all"]), sigdigits = 3)) Pa)")
        for c in ("Fx", "Fy")
            Fmean[(m, b, "cross", c)] = Fmean[(m, b, "all", c)] .- Fmean[(m, b, "north", c)] .- Fmean[(m, b, "south", c)]
            println("mode $n $b $c: domain-mean |cross| / |all| = ",
                    "$(round(mean(abs, Fmean[(m, b, "cross", c)][:, :, 1]) / mean(abs, Fmean[(m, b, "all", c)][:, :, 1]), digits = 2)) (full record)")
        end
    end
    spec = nothing
    GC.gc()
end
close(ds_puv)

# modes 1..NMODE together: the sum of each term (no cross-mode terms in the depth-integrated flux)
mode_tags = [mtag.(1:NMODE); "modesum"]
for (b, _, _) in bands, sec in [first.(sectors); "cross"], c in ("Fx", "Fy")
    Fmean[("modesum", b, sec, c)] = sum(Fmean[(mtag(n), b, sec, c)] for n in 1:NMODE)
end
for (b, _, _) in bands, sec in first.(sectors), c in ("Fx", "Fy"), l in [first.(tr_lines); "strip"]
    Fline[("modesum", b, sec, c, l)] = sum(Fline[(mtag(n), b, sec, c, l)] for n in 1:NMODE)
end
println("\nM2 deep-ocean (H > 1000 m) mean |F| of the full record, all directions:")
deep = H .> 1000
for m in mode_tags
    Fx, Fy = Fmean[(m, "M2", "all", "Fx")][:, :, 1], Fmean[(m, "M2", "all", "Fy")][:, :, 1]
    println("  $m: $(round(mean(hypot.(Fx, Fy)[deep]), sigdigits = 3)) W/m")
end

# ============================================================================
# (4) save
# ============================================================================
mode_desc(m) = m == "modesum" ? "modes 1-$NMODE together (sum)" : replace(m, "mode" => "mode-")
isfile(out_fname) && rm(out_fname)
NCDataset(out_fname * ".tmp", "c") do ds
    defDim(ds, "xi_rho", nx_f); defDim(ds, "eta_rho", ny_f); defDim(ds, "period", length(periods)); defDim(ds, "time", nt)
    defDim(ds, "mode", NMODE)
    defVar(ds, "lon_rho", lon_rho, ("xi_rho", "eta_rho"); attrib = ["units" => "degree_east"])
    defVar(ds, "lat_rho", lat_rho, ("xi_rho", "eta_rho"); attrib = ["units" => "degree_north"])
    defVar(ds, "H", H, ("xi_rho", "eta_rho"); attrib = ["long_name" => "h + mean zeta", "units" => "meter"])
    defVar(ds, "c", cn, ("xi_rho", "eta_rho", "mode"); attrib = ["long_name" => "mode eigenspeeds", "units" => "m s-1"])
    defVar(ds, "ocean_time", date, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
    ds.attrib["periods"] = join(["$p = $(date[t[1]]) - $(date[t[end]])" for (p, t) in periods], "; ")
    for m in mode_tags, (b, _, bdesc) in bands, sec in [first.(sectors); "cross"], c in ("Fx", "Fy")
        defVar(ds, "$(c)_$(b)_$(sec)_$(m)", Fmean[(m, b, sec, c)], ("xi_rho", "eta_rho", "period");
               attrib = ["long_name" => "time-mean $(mode_desc(m)) energy flux along grid $(c == "Fx" ? "xi" : "eta"), $bdesc, $(sec_desc[sec])",
                         "units" => "W m-1"])
    end
    ds.attrib["nmode"] = NMODE
    ds.attrib["note"] = "F = H pn (un, vn) from bcef_modes_puv.jl; each direction's pn, un, vn from the DFF (DDF_on_ssh.jl method). " *
                        "Fx along grid xi, Fy along grid eta (not east / north). all = north + south + cross. " *
                        "Variables <Fx|Fy>_<band>_<sector>_<mode1..modeN|modesum>; modesum = sum over the modes (no cross-mode terms)."
end
mv(out_fname * ".tmp", out_fname; force = true)
println("wrote $out_fname")

# flux along the three lines every hour, one file per line (as the ssh transects)
for (l, ix) in tr_lines
    fname = joinpath(bcef_dir, "bcef_dff_flux_modes$(NMODE)_transect_$(l)_xi$(ix).nc")
    isfile(fname) && rm(fname)
    NCDataset(fname, "c") do ds
        defDim(ds, "eta_rho", ny_f); defDim(ds, "time", nt)
        defVar(ds, "ocean_time", date, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
        defVar(ds, "H", H[ix, :], ("eta_rho",); attrib = ["units" => "meter"])
        defVar(ds, "lat_rho", lat_rho[ix, :], ("eta_rho",); attrib = ["units" => "degree_north"])
        for m in mode_tags, (b, _, bdesc) in bands, sec in first.(sectors), c in ("Fx", "Fy")
            defVar(ds, "$(c)_$(b)_$(sec)_$(m)", Fline[(m, b, sec, c, l)], ("eta_rho", "time");
                   attrib = ["long_name" => "$(mode_desc(m)) energy flux along grid $(c == "Fx" ? "xi" : "eta"), $bdesc, $sec", "units" => "W m-1"])
        end
        ds.attrib["line"] = "$l line, grid column xi = $ix (Julia 1-based, trimmed DFF grid)"
    end
    println("wrote $fname")
end

# the strip mean every hour, one file (as the transects): <Fx|Fy>_<band>_<sector>_<mode> (eta, time)
let fname = joinpath(bcef_dir, "bcef_dff_flux_modes$(NMODE)_transect_strip_xi$(first(strip_cols))-$(last(strip_cols)).nc")
    isfile(fname) && rm(fname)
    NCDataset(fname, "c") do ds
        defDim(ds, "eta_rho", ny_f); defDim(ds, "time", nt)
        defVar(ds, "ocean_time", date, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
        defVar(ds, "H", H[ix_strip_c, :], ("eta_rho",); attrib = ["units" => "meter", "long_name" => "h + mean zeta on the centre column"])
        defVar(ds, "lat_rho", lat_rho[ix_strip_c, :], ("eta_rho",); attrib = ["units" => "degree_north", "long_name" => "on the centre column"])
        defVar(ds, "strip_cover", strip_cover, ("eta_rho",);
               attrib = ["long_name" => "share of the strip columns in the mean (water deeper than $(Int(strip_hmin)) m); 0: row not averaged"])
        for m in mode_tags, (b, _, bdesc) in bands, sec in first.(sectors), c in ("Fx", "Fy")
            defVar(ds, "$(c)_$(b)_$(sec)_$(m)", Fline[(m, b, sec, c, "strip")], ("eta_rho", "time");
                   attrib = ["long_name" => "$(mode_desc(m)) energy flux along grid $(c == "Fx" ? "xi" : "eta"), $bdesc, $sec, mean across the strip",
                             "units" => "W m-1"])
        end
        ds.attrib["line"] = "strip: ±$(strip_km) km = ±$nside grid columns around xi = $ix_strip_c, xi = $(first(strip_cols))-$(last(strip_cols)) (Julia 1-based, trimmed DFF grid)"
        ds.attrib["method"] = "H pn (un, vn) at each point of a direction, averaged per row over the strip's water points deeper than $(Int(strip_hmin)) m (centre column as mode_dff_binplot.jl)"
    end
    println("wrote $fname")
end
