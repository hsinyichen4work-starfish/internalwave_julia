#= mode_bcef_harmonic.jl
Hsin-Yi Chen, USM DMS
Harmonic analysis of the modal baroclinic pressure and velocity (pn, un, vn from bcef_modes_puv.jl), and the tidal
energy flux of each constituent from the fit. Same method as model_validation/tidal/ssh_harmonic.jl (function/harm_fit.jl).

Section 1: settings, grid, mean depth H = h + mean zeta (bcef_modes<N>_structure.nc), the time line of the puv file;
  SPINUP days dropped at the start (as ssh_harmonic.jl); STRIDE > 1 fits every STRIDE-th grid point only (quick tests).

Section 2: plain sinusoid least-squares fit at every point, one field at a time (p1, u1, v1, p2, u2, v2):
    y(t) = mean + trend (t - tc)/span + sum_k A'_k cos(w_k (t - tc) - g'_k)
  A' [Pa or m/s], g' [deg] = phase at the record centre tc (no nodal corrections, no inference; see ssh_harmonic.jl).
  pn, un, vn are the mode amplitudes: p'(z, t) = sum_n pn(t) phi_n(z), same for u', v' (phi_n in the structure file);
  un, vn are along the grid xi / eta axes.

Section 3: energy flux of each constituent from the fit. For one sinusoid, the time mean of a product is
    <p u> = ½ A_p A_u cos(g_p - g_u),
  so  Fx_n,k = ½ H A_p A_u cos(g_p - g_u),   Fy_n,k = ½ H A_p A_v cos(g_p - g_v)   [W m-1],
  along grid xi / eta, per mode n and constituent k, and the sum over the modes (orthogonal: no cross-mode terms).
  This is the phase-locked (coherent) tidal flux; the DFF flux of DFF_on_bcef.jl is a band-pass, so it also holds the
  incoherent part of the band. The (pn, un) phase difference shows progressive (0 or 180 deg) vs standing (±90 deg) waves.

Section 4 (when M2 and M4 are both fitted): the M4 / M2 pair of each field, for steepening,
    ratio = A'_M4 / A'_M2,   relative phase = 2 g'_M2 - g'_M4  [0, 360)
  near 0 / 180 deg: crests and troughs sharpened / flattened (symmetric distortion); near 90 / 270 deg: one face
  steeper than the other (asymmetric, steepening of the front or back).

Output: <run>/ddf_bcef/bcef_modes<N>_harmonic[_s<STRIDE>].nc

Usage:
  julia DFF/mode_bcef_harmonic.jl                        # full grid (compute node: sbatch_files/julia_bcef_harmonic.sh)
  STRIDE=10 julia DFF/mode_bcef_harmonic.jl              # every 10th point, test on the login node
  HARM_CONS=M2,S2,N2,K1,O1 HARM_MODES=1,2,3 ...         # other constituents / modes (default adds M4, MS4, MN4, M6)
=#

using NCDatasets, Printf, Statistics, Dates, LinearAlgebra
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")            # dff_run, run_dir, run_grid_fname (DFF_RUN)
include("/home/hchen54/internalwave_julia/function/harm_fit.jl")      # harm_fit, harm_predict, TIDE_FREQ

## --- section 1: settings
NMODE   = parse(Int, get(ENV, "NMODE", "3"))                        # modes in the puv file
MODES   = parse.(Int, split(get(ENV, "HARM_MODES", "1,2"), r"[,:; ]+"; keepempty = false))     # modes to fit
# tides as ssh_harmonic.jl, plus the shallow-water overtides (steepening / nonlinear waves): M4, MS4, MN4, M6.
# All resolved by ~95 days of hourly data (MN4-M4: 3.4 cycles apart; M6 period 4.1 h > the 2 h Nyquist period).
CONS    = String.(split(get(ENV, "HARM_CONS", "M2,S2,N2,K1,O1,M4,MS4,MN4,M6"), r"[,:; ]+"; keepempty = false))
SPINUP  = parse(Float64, get(ENV, "SPINUP", "3.0"))                  # days dropped at the start
STRIDE  = parse(Int, get(ENV, "STRIDE", "1"))
t_ref   = DateTime(1994, 1, 1)                                       # ocean_time reference
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, "ddf_bcef"))
puv_fname    = joinpath(bcef_dir, "bcef_modes$(NMODE)_puv.nc")
struct_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc")
fout = joinpath(bcef_dir, "bcef_modes$(NMODE)_harmonic$(STRIDE > 1 ? "_s$STRIDE" : "").nc")
for f in (puv_fname, struct_fname)
    isfile(f) || error("no $f: run bcef_modes_puv.jl with NMODE = $NMODE first")
end
all(<=(NMODE), MODES) || error("HARM_MODES = $MODES, but the puv file has $NMODE modes")

## --- grid, H, time line
ds = NCDataset(puv_fname)
nx, ny = ds.dim["xi_rho"], ds.dim["eta_rho"]
ix, jy = 1:STRIDE:nx, 1:STRIDE:ny
lon, lat = NCDataset(run_grid_fname) do g
    g["lon_rho"][ix, jy], g["lat_rho"][ix, jy]
end
h, H = NCDataset(struct_fname) do d
    Float64.(d["h"][ix, jy]), Float64.(d["h"][ix, jy] .+ d["zeta_mean"][ix, jy])
end
date = ds["ocean_time"][:]
dt_h = Dates.value.(diff(date)) ./ 3.6e6
all(dt_h .≈ dt_h[1]) || @printf("  warning: records not evenly spaced (%.2f..%.2f h)\n", extrema(dt_h)...)
t0 = findfirst(>=(date[1] + Millisecond(round(Int, SPINUP * 86400e3))), date)
tidx = t0:length(date)                                             # contiguous: one read per field
tdate = date[tidx]
tsec = Dates.value.(tdate .- t_ref) ./ 1e3                          # seconds since t_ref (for the fit)
NX, NY, nt = length(ix), length(jy), length(tidx)
@printf("%s: grid %d x %d (stride %d), %d records %s -> %s (%.1f days spin-up dropped), modes %s, cons %s\n",
        dff_run, NX, NY, STRIDE, nt, tdate[1], tdate[end], SPINUP, join(MODES, ","), join(CONS, ","))

## --- section 2: harmonic fit, one field at a time
# read with a contiguous time range: whole records, then the stride (a strided read is many small reads)
# Blocks of 100 whole records (contiguous), keeping every STRIDE-th point. Lustre timeouts show up as HDF errors:
# the block is then retried with the file reopened, after 2, 4, 8, 16, 32 min.
function read_field(name)
    a = Array{Float32}(undef, NX, NY, nt)
    t = @elapsed for b in Iterators.partition(eachindex(tidx), 100)
        for attempt in 1:6
            try
                NCDataset(puv_fname) do d
                    blk = nomissing(d[name][:, :, tidx[b]], NaN32)
                    a[:, :, b] .= STRIDE == 1 ? blk : blk[ix, jy, :]
                end
                break
            catch e
                attempt == 6 && rethrow()
                @printf("  %s records %d-%d failed (%s); retry %d in %d min\n", name, tidx[first(b)], tidx[last(b)],
                        first(sprint(showerror, e), 80), attempt, 2^attempt)
                flush(stdout)
                sleep(60 * 2^attempt)
            end
        end
    end
    @printf("  read %s: %.2f GB in %.0f s\n", name, sizeof(a) / 1e9, t)
    a
end
fits = Dict{String, Any}()
for n in MODES, v in ("p", "u", "v")
    name = "$v$n"
    println("== $name")
    a = read_field(name)
    fits[name] = harm_fit(a, tsec, CONS)
    a = nothing; GC.gc()
    hf = fits[name]
    ok = isfinite.(hf.std) .& (h .>= 200)
    u = v == "p" ? "Pa" : "m/s"
    for (k, c) in enumerate(CONS)
        A = selectdim(hf.amp, 3, k)[ok]
        @printf("   %-4s median A' %.3g %s (90th pct %.3g), median error %.2g\n", c, median(A), u, quantile(A, 0.9),
                median(selectdim(hf.aerr, 3, k)[ok]))
    end
    @printf("   tides (%s) explain %.1f%% of the variance (median over h >= 200 m)\n", join(CONS, "+"),
            100 * (1 - median(hf.resid[ok] .^ 2 ./ hf.std[ok] .^ 2)))
end
tc = fits["p$(MODES[1])"].tc
tcdate = t_ref + Millisecond(round(Int, 1000tc))

## --- section 3: energy flux of each constituent, Fx = ½ H A_p A_u cos(g_p - g_u)
flux(n, w, k) = 0.5 .* H .* selectdim(fits["p$n"].amp, 3, k) .* selectdim(fits["$w$n"].amp, 3, k) .*
                cosd.(selectdim(fits["p$n"].pha, 3, k) .- selectdim(fits["$w$n"].pha, 3, k))
Fx = Dict((n, c) => flux(n, "u", k) for n in MODES, (k, c) in enumerate(CONS))
Fy = Dict((n, c) => flux(n, "v", k) for n in MODES, (k, c) in enumerate(CONS))
deep = (h .>= 1000) .& isfinite.(H)
println("\nconstituent energy flux, mean over h >= 1000 m [W/m] (grid xi, eta):")
for c in CONS
    @printf("  %-4s", c)
    for n in MODES
        @printf("  mode %d: Fx %8.1f  Fy %8.1f  |F| %8.1f", n, mean(filter(isfinite, Fx[(n, c)][deep])), mean(filter(isfinite, Fy[(n, c)][deep])),
                mean(filter(isfinite, hypot.(Fx[(n, c)], Fy[(n, c)])[deep])))
    end
    println()
end

## --- save
isfile(fout) && rm(fout)
NCDataset(fout * ".tmp", "c") do o
    defDim(o, "xi_rho", NX); defDim(o, "eta_rho", NY); defDim(o, "con", length(CONS))
    o.attrib["title"] = "Plain sinusoid harmonic fit of the modal pn, un, vn and the constituent energy flux, $dff_run, $(Date(tdate[1])) to $(Date(tdate[end]))"
    o.attrib["model"] = "y = mean + trend (t - tc)/span + sum A' cos(w (t - tc) - g'); no nodal corrections, no inference"
    o.attrib["flux"] = "Fx_n = ½ H A_p A_u cos(g_p - g_u), Fy_n = ½ H A_p A_v cos(g_p - g_v), H = h + mean zeta; grid xi / eta"
    o.attrib["to_greenwich"] = "A = A' / f(tc), G = g' + V(tc) + u(tc)"
    o.attrib["tc"] = string(tcdate); o.attrib["tc_seconds_since_1994"] = tc
    o.attrib["records"] = nt; o.attrib["spinup_days"] = SPINUP; o.attrib["stride"] = STRIDE
    o.attrib["modes"] = join(MODES, ","); o.attrib["source"] = puv_fname
    o.attrib["script"] = "internalwave_julia/DFF/mode_bcef_harmonic.jl"
    defVar(o, "con", CONS, ("con",))
    defVar(o, "freq", [TIDE_FREQ[c] for c in CONS], ("con",); attrib = ["units" => "deg/hour"])
    defVar(o, "lon_rho", lon, ("xi_rho", "eta_rho")); defVar(o, "lat_rho", lat, ("xi_rho", "eta_rho"))
    defVar(o, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
    defVar(o, "H", H, ("xi_rho", "eta_rho"); attrib = ["units" => "m", "long_name" => "h + mean zeta"])
    for n in MODES, v in ("p", "u", "v")
        name, hf = "$v$n", fits["$v$n"]
        u = v == "p" ? "Pa" : "m s-1"
        what = v == "p" ? "mode-$n pressure amplitude p$n" : "mode-$n velocity amplitude $name along grid $(v == "u" ? "xi" : "eta")"
        for (nm, x, uu, ln) in (("amp", hf.amp, u, "amplitude A'"), ("pha", hf.pha, "deg", "phase g' relative to tc"),
                                ("amp_err", hf.aerr, u, "white-noise std error of A'"), ("pha_err", hf.perr, "deg", "white-noise std error of g'"))
            defVar(o, "$(name)_$nm", Float32.(x), ("xi_rho", "eta_rho", "con"); attrib = ["units" => uu, "long_name" => "$what: $ln"], deflatelevel = 3)
        end
        for (nm, x, ln) in (("mean", hf.mean, "mean"), ("trend", hf.trend, "linear change over the record"),
                            ("resid_std", hf.resid, "residual std"), ("std", hf.std, "std of the data"))
            defVar(o, "$(name)_$nm", Float32.(x), ("xi_rho", "eta_rho"); attrib = ["units" => u, "long_name" => "$what: $ln"], deflatelevel = 3)
        end
    end
    for (k, c) in enumerate(CONS), (tag, F, ax) in (("Fx", Fx, "xi"), ("Fy", Fy, "eta"))
        for n in MODES
            defVar(o, "$(tag)_$(c)_mode$n", Float32.(F[(n, c)]), ("xi_rho", "eta_rho");
                   attrib = ["units" => "W m-1", "long_name" => "$c mode-$n energy flux along grid $ax, ½ H A_p A_$(tag == "Fx" ? "u" : "v") cos(Δg)"], deflatelevel = 3)
        end
        defVar(o, "$(tag)_$(c)_modesum", Float32.(sum(F[(n, c)] for n in MODES)), ("xi_rho", "eta_rho");
               attrib = ["units" => "W m-1", "long_name" => "$c energy flux along grid $ax, sum of modes $(join(MODES, ","))"], deflatelevel = 3)
    end
    # section 4: M4 / M2 pair of each field
    kM2, kM4 = findfirst(==("M2"), CONS), findfirst(==("M4"), CONS)
    if kM2 !== nothing && kM4 !== nothing
        for n in MODES, v in ("p", "u", "v")
            hf = fits["$v$n"]
            ratio = selectdim(hf.amp, 3, kM4) ./ selectdim(hf.amp, 3, kM2)
            relph = mod.(2 .* selectdim(hf.pha, 3, kM2) .- selectdim(hf.pha, 3, kM4), 360)
            defVar(o, "$(v)$(n)_M4M2_ratio", Float32.(ratio), ("xi_rho", "eta_rho");
                   attrib = ["long_name" => "A'_M4 / A'_M2 of $v$n"], deflatelevel = 3)
            defVar(o, "$(v)$(n)_M4M2_relpha", Float32.(relph), ("xi_rho", "eta_rho");
                   attrib = ["units" => "deg", "long_name" => "2 g'_M2 - g'_M4 of $v$n (0/180: symmetric distortion, 90/270: asymmetric, steepening)"], deflatelevel = 3)
            ok = (h .>= 1000) .& isfinite.(ratio)
            @printf("  %s%d: median A'_M4 / A'_M2 = %.3f (90th pct %.3f) over h >= 1000 m\n", v, n, median(ratio[ok]), quantile(ratio[ok], 0.9))
        end
    end
end
mv(fout * ".tmp", fout; force = true)
close(ds)
println("wrote $fout")
