#= ssh_harmonic.jl
Hsin-Yi Chen, USM DMS
Harmonic analysis of ROMS sea-surface height (zeta), Amazon 900 m run (amazon_900m_3mon_1008_2).
Rewritten section by section from ROMS_TPXO_tidalcomp.jl (step 1).

Section 1: read hourly zeta from roms_his.*.nc (4 hourly records per file) into ssh[xi, eta, time] [m].
  - timeline de-duplicated by hour (a file's last record can repeat the next file's first)
  - SPINUP days dropped at the start
  - STRIDE > 1 reads every STRIDE-th grid point only (quick tests; full grid ~5.5 GB as Float32)

Section 2: plain sinusoid least-squares fit at every point, no tidal atlas, no nodal corrections:
    eta(t) = mean + trend (t - tc)/span + sum_k [ a_k cos(w_k (t - tc)) + b_k sin(w_k (t - tc)) ]
           = ...                        + sum_k A'_k cos(w_k (t - tc) - g'_k)
  A'_k = sqrt(a^2 + b^2) [cm], g'_k = atan(b, a) [deg] = phase relative to the record centre tc.
  One design matrix for all points; normal equations solved with one Cholesky factorisation.
  Constituents must be resolved by the record (Rayleigh: |w_i - w_j| T >= 360 deg); 95 days do not
  separate K2/S2, P1/K1, NU2/N2, so those are left out (S2, K1, N2 absorb them).
  Greenwich phase / nodal-free amplitude later, when comparing with an atlas (V, u, f at tc):
    A = A' / f,   G = g' + V(tc) + u(tc)
  Output: dirout/ROMS_<RUN>_sshfit[_s<STRIDE>].nc

Usage:
  julia -t 8 ssh_harmonic.jl                 # full grid (compute node)
  STRIDE=10 julia ssh_harmonic.jl            # every 10th point, for testing on the login node
=#

using NCDatasets, Printf, Statistics, Dates, LinearAlgebra
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")   # time_dir, records_by_file, figkw, ...

## --- settings
const data_folder = "/expanse/lustre/projects/uso102/hchen54/amazon_900m_3mon_1008_2/after_joint/"
const grid_file   = "/expanse/lustre/projects/uso102/hchen54/input/grid/roms_grd_900m.nc"
const t_ref       = DateTime(1994, 1, 1)                  # ocean_time reference ("Time since 1994/01/01")
const SPINUP      = 3.0                                   # days dropped at the start of the run
const STRIDE      = parse(Int, get(ENV, "STRIDE", "1"))   # read every STRIDE-th grid point (1 = full grid)
const dirout      = "/expanse/lustre/projects/uso102/hchen54/figure/amazon_900m_3mon_1008/tidal_comp/"
const RUN         = "amazon900m"
const fout        = dirout * "ROMS_$(RUN)_sshfit$(STRIDE > 1 ? "_s$STRIDE" : "").nc"

## constituents to fit (frequencies: TIDE_FREQ in function/harm_fit.jl)
const CONS = ["M2", "S2", "N2", "K1", "O1"]

## --- grid (Julia order: xi, eta)
ix = 1:STRIDE:686; jy = 1:STRIDE:856
lon, lat, h, mask = NCDataset(grid_file) do g
    g["lon_rho"][ix, jy], g["lat_rho"][ix, jy], g["h"][ix, jy], g["mask_rho"][ix, jy] .== 1
end
NX, NY = size(lon)
@printf("grid %d x %d (stride %d), lon %.2f..%.2f, lat %.2f..%.2f\n", NX, NY, STRIDE, extrema(lon)..., extrema(lat)...)

## --- timeline: which file / record holds each hour (function/time_dir.jl)
td = time_dir(data_folder; t_ref, spinup = SPINUP)
nt, tdate, tsec = td.nt, td.tdate, td.tsec                     # tsec: seconds since t_ref (for the fit)

## --- read zeta [m] (function/data_load.jl); land -> NaN from the grid mask
ssh = data_load(data_folder, "zeta", td; ix, jy, grid_file)

## --- quick check: SSH at the deepest point
p = argmax(ifelse.(mask, h, -Inf))
@printf("deepest point (%.2f, %.2f), h = %.0f m: SSH mean %.3f m, std %.3f m, range %.2f..%.2f m\n",
        lon[p], lat[p], h[p], mean(ssh[p, :]), std(ssh[p, :]), extrema(ssh[p, :])...)

## --- section 2: plain sinusoid fit (function/harm_fit.jl), SSH m -> cm
hf = harm_fit(ssh, tsec, CONS; scale = 100)
tcdate = t_ref + Millisecond(round(Int, 1000hf.tc))

## --- summary: median over ocean points, and the deepest point
println("record centre tc = $tcdate")
@printf("%-5s %8s %12s  %10s %10s   deepest point: %6s %6s\n", "con", "period h", "median A' cm", "med err cm", "med err deg", "A' cm", "g' deg")
for (k, c) in enumerate(CONS)
    @printf("%-5s %8.2f %12.2f  %10.3f %10.2f   %21.2f %6.1f\n", c, 360 / TIDE_FREQ[c], median(filter(!isnan, hf.amp[:, :, k])),
            median(filter(!isnan, hf.aerr[:, :, k])), median(filter(!isnan, hf.perr[:, :, k])), hf.amp[p, k], hf.pha[p, k])
end
@printf("median residual std %.2f cm of SSH std %.2f cm (tides explain %.1f%% of the variance)\n",
        median(filter(!isnan, hf.resid)), median(filter(!isnan, hf.std)),
        100 * (1 - median(filter(!isnan, hf.resid .^ 2 ./ hf.std .^ 2))))

## --- save
mkpath(dirout); isfile(fout) && rm(fout)
NCDataset(fout, "c") do ds
    defDim(ds, "xi_rho", NX); defDim(ds, "eta_rho", NY); defDim(ds, "con", length(CONS))
    ds.attrib["title"] = "Plain sinusoid harmonic fit of hourly SSH, ROMS $(RUN), $(Date(tdate[1])) to $(Date(tdate[end]))"
    ds.attrib["model"] = "eta = mean + trend (t - tc)/span + sum A' cos(w (t - tc) - g'); no nodal corrections, no inference"
    ds.attrib["to_greenwich"] = "A = A' / f(tc), G = g' + V(tc) + u(tc)"
    ds.attrib["tc"] = string(tcdate); ds.attrib["tc_seconds_since_1994"] = hf.tc
    ds.attrib["records"] = nt; ds.attrib["stride"] = STRIDE
    ds.attrib["script"] = "internalwave_julia/model_validation/ssh_harmonic.jl"
    defVar(ds, "con", CONS, ("con",))
    defVar(ds, "freq", [TIDE_FREQ[c] for c in CONS], ("con",); attrib = ["units" => "deg/hour"])
    defVar(ds, "lon_rho", lon, ("xi_rho", "eta_rho")); defVar(ds, "lat_rho", lat, ("xi_rho", "eta_rho"))
    defVar(ds, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
    for (nm, v, u, ln) in (("amp", hf.amp, "cm", "amplitude A' (no nodal correction)"), ("pha", hf.pha, "deg", "phase g' relative to tc"),
                           ("amp_err", hf.aerr, "cm", "white-noise std error of A'"), ("pha_err", hf.perr, "deg", "white-noise std error of g'"))
        defVar(ds, nm, Float32.(v), ("xi_rho", "eta_rho", "con"); attrib = ["units" => u, "long_name" => ln], deflatelevel = 3)
    end
    for (nm, v, u, ln) in (("mean", hf.mean, "cm", "mean SSH"), ("trend", hf.trend, "cm", "linear SSH change over the record"),
                           ("resid_std", hf.resid, "cm", "residual std"), ("ssh_std", hf.std, "cm", "SSH std"))
        defVar(ds, nm, Float32.(v), ("xi_rho", "eta_rho"); attrib = ["units" => u, "long_name" => ln], deflatelevel = 3)
    end
end
println("wrote $fout")
