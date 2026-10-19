#= ROMS_TPXO_tidalcomp.jl
Hsin-Yi Chen, USM DMS, 2026-10-19
Adapted from M. Buijsman's claudecodes/ROMS/ROMS_tide_harmfit.jl and ROMS_tide_TPXO_compare.jl
(github.com/maartenbuijsman/julia-codes) for the Amazon 900 m run (amazon_900m_3mon_1008_2).

Step 1 (harmonic fit, cached): hourly zeta from roms_his.*.nc, 2022-08-24 .. 2022-11-30, at every
  rho point (686 x 856), least squares with one design matrix shared by all points; normal equations
  accumulated while streaming the files (zeta only, ~2.3 MB per record).
  - timeline de-duplicated by hour; ocean_time = seconds since 1994-01-01; SPINUP days dropped
  - model: mean + linear trend + FIT constituents; f(t), u(t), V(t) every hour (pyTMD, OTIS/TPXO
    conventions, latitude independent, via function/nodal_args.py) -> nodal-free A, G
  - ~95 days do not separate K2 from S2, P1 from K1, NU2 from N2 (Rayleigh 182-206 d): these are
    inferred, Z_K2 = R Z_S2 etc., with R the TPXO10 complex ratio (area mean over the interior),
    and are part of the S2/K1/N2 basis functions (t_tide-like inference)
  - convention (TPXO): eta = f A cos(V + u - G), Z = A exp(-iG) = hRe + i hIm
  Output: dirout/ROMS_<RUN>_harmconst[_test].nc. Delete it, or REFIT=true, to fit again.
Step 2 (comparison with TPXO10 v2a, 1/6 deg OTIS binary, the only atlas available here):
  - TPXO bilinearly interpolated to the rho points (lon_rho is already 0-360); points next to a
    TPXO land cell (z = 0) are masked (Amazon mouth, coast)
  - ROMS barotropic Z_bt = Z_TPXO + LP(Z_ROMS - Z_TPXO); LP a 2-D Gaussian (normalised convolution
    over the comparison area, sigma = cutoff/5.336, half power at the cutoff) in grid-index space
    (the grid is a uniform 0.9 km grid rotated by -34 deg). Z_ROMS - Z_bt is the coherent
    internal-tide SSH. Cutoffs LC_SD / LC_D are set for this ~620 x 770 km domain (mode-1
    wavelength ~120-150 km for M2, ~250-300 km for K1/O1 here); the script prints the RMS
    sensitivity to the cutoff, check it before trusting the numbers.
  - comparison area: SPONGE km removed along each open boundary, TPXO-wet points; cell-area weights
  - per constituent: RMS = sqrt(< |Z1 - Z2|^2 / 2 >), signal rms S, amplitude ratio, amplitude-
    weighted phase difference (+ = ROMS lags); RMS split into shelf (h < HSHELF) and deep water
Output in dirout: ROMS_TPXO_rms.txt, ROMS_TPXO_maps.png, ROMS_TPXO_rms.png, ROMS_M2_ITssh.png,
  ROMS_TPXO_rms_maps.png, ROMS_TPXO_cotidal.png (maps in model coordinates, km along xi / eta,
  with lon/lat lines and the 200 m isobath)

Run (batch: sbatch_files/julia_tidalcomp.sh):
  TEST=true julia -t 8 --project=. ROMS_TPXO_tidalcomp.jl   # first 30 days, M2 S2 N2 K1 O1
  julia -t 8 --project=. ROMS_TPXO_tidalcomp.jl             # full record
=#

using NCDatasets, CairoMakie, Printf, Statistics, LinearAlgebra, Dates

const dirR   = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/after_joint/"
const fgrd   = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
const fT6    = "/expanse/lustre/projects/uso101/hchen54/data_upload/h_tpxo10.v2a"
const dirout = "/expanse/lustre/projects/uso101/hchen54/figure/amazon_900m_3mon_1008/tidal_comp/"
const PY     = "/home/hchen54/.venvs/utide/bin/python"
const NODPY  = "/home/hchen54/internalwave_julia/function/nodal_args.py"
const RUN    = "amazon900m"
const TEST   = get(ENV, "TEST", "false") == "true"
const REFIT  = get(ENV, "REFIT", "false") == "true"
const SUF    = TEST ? "_test" : ""
const TREF   = DateTime(1994, 1, 1)          # ocean_time reference ("Time since 1994/01/01")
const SPINUP = 3.0                           # days dropped at the start of the run
const FIT    = TEST ? ["m2", "s2", "n2", "k1", "o1"] :
               ["m2", "s2", "n2", "2n2", "k1", "o1", "q1", "l2", "j1", "oo1", "m4", "ms4", "mn4"]
const INFER  = TEST ? Tuple{String, String}[] : [("k2", "s2"), ("p1", "k1"), ("nu2", "n2")]   # (inferred, from)
const SPONGE = 30.0                          # km removed along each open boundary (check the run's sponge width)
const LC_SD  = 400.0                         # km, half-power wavelength of the low-pass, semidiurnal and higher
const LC_D   = 800.0                         # km, diurnal
const HSHELF = 200.0                         # m, shelf / deep-water split
const MAJOR  = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
const DIURNAL = ["K1", "O1", "P1", "Q1", "J1", "OO1"]
const MAPC   = ["M2", "S2", "K1", "O1"]
const fhc    = dirout * "ROMS_$(RUN)_harmconst$(SUF).nc"
mkpath(dirout)

## --- grid (Julia order: xi, eta)
lonR, latR, hR, pm, pn = NCDataset(fgrd) do g
    g["lon_rho"][:, :], g["lat_rho"][:, :], g["h"][:, :], g["pm"][:, :], g["pn"][:, :]
end
NX, NY = size(lonR)
dxkm = 1e-3 * mean(1 ./ pm); dykm = 1e-3 * mean(1 ./ pn)
@printf("ROMS grid %d x %d, dx %.3f km, dy %.3f km, lon %.2f..%.2f, lat %.2f..%.2f\n", NX, NY, dxkm, dykm, extrema(lonR)..., extrema(latR)...)

## --- TPXO10 v2a 1/6 deg, OTIS binary: big-endian Fortran records; header n, m, nc, lat limits, lon
##     limits, constituent ids; then per constituent z(2, n, m) float32 (Re, Im) [m] on cell centres.
##     Bilinear interpolation to the rho points [cm]; wet = all four surrounding TPXO cells are ocean.
function otis_h(f, cons)
    out = Dict{String, Matrix{ComplexF64}}(); wet = trues(NX, NY)
    open(f) do io
        r(T) = ntoh(read(io, T))
        r(Int32); n = r(Int32); m = r(Int32); nc = r(Int32)
        th = [r(Float32) for _ in 1:2]; ph = [r(Float32) for _ in 1:2]
        cid = [strip(String(read(io, 4))) for _ in 1:nc]; r(Int32)
        hdr = position(io)
        fx = (lonR .- ph[1]) ./ ((ph[2] - ph[1]) / n) .+ 0.5; fy = (latR .- th[1]) ./ ((th[2] - th[1]) / m) .+ 0.5
        i0 = floor.(Int, fx); j0 = floor.(Int, fy); wx = fx .- i0; wy = fy .- j0
        for c in cons
            ic = findfirst(==(lowercase(c)), cid); ic === nothing && (println("TPXO: no $c"); continue)
            seek(io, hdr + (ic - 1) * (8 + 8n * m) + 4)
            z = reshape(ntoh.(reinterpret(Float32, read(io, 8n * m))), 2, n, m)
            Zc(i, j) = complex(100 * Float64(z[1, i, j]), 100 * Float64(z[2, i, j]))
            out[uppercase(c)] = [(1 - wx[p]) * (1 - wy[p]) * Zc(i0[p], j0[p]) + wx[p] * (1 - wy[p]) * Zc(i0[p] + 1, j0[p]) +
                                 (1 - wx[p]) * wy[p] * Zc(i0[p], j0[p] + 1) + wx[p] * wy[p] * Zc(i0[p] + 1, j0[p] + 1) for p in CartesianIndices(lonR)]
            if lowercase(c) == "m2"
                wet .= [all(!iszero, (Zc(i0[p], j0[p]), Zc(i0[p] + 1, j0[p]), Zc(i0[p], j0[p] + 1), Zc(i0[p] + 1, j0[p] + 1))) for p in CartesianIndices(lonR)]
            end
        end
    end
    return out, wet
end
ALLC = vcat(FIT, first.(INFER))
TP, WET = otis_h(fT6, ALLC)
@printf("TPXO10 v2a: %s; %.2f%% of the rho points next to TPXO land (masked)\n", join(sort(collect(keys(TP))), " "), 100 * (1 - mean(WET)))

## --- comparison area and weights
nsx = round(Int, SPONGE / dxkm); nsy = round(Int, SPONGE / dykm)
M = falses(NX, NY); M[nsx+1:NX-nsx, nsy+1:NY-nsy] .= true; M .&= WET
W = M ./ (pm .* pn)
wmean(X, w = W) = sum(w .* X) / sum(w)
rmsd(Z1, Z2, w = W) = sqrt(wmean(abs2.(Z1 .- Z2) ./ 2, w))
srms(Z, w = W) = sqrt(wmean(abs2.(Z) ./ 2, w))
aratio(Z1, Z2) = wmean(abs.(Z1)) / wmean(abs.(Z2))
dphase(Z1, Z2) = -rad2deg(angle(sum(W .* Z1 .* conj.(Z2))))             # + = Z1 lags Z2
Wsh = W .* (hR .< HSHELF); Wdp = W .* (hR .>= HSHELF)

## --- inference ratios from TPXO (complex, area mean over the comparison area)
RINF = Dict(ci => sum(W .* TP[uppercase(ci)] .* conj.(TP[uppercase(cb)])) / sum(W .* abs2.(TP[uppercase(cb)])) for (ci, cb) in INFER)
for (ci, cb) in INFER
    @printf("inference %s = R %s: |R| %.4f, arg %.2f deg (TPXO10)\n", uppercase(ci), uppercase(cb), abs(RINF[ci]), rad2deg(angle(RINF[ci])))
end

## --- step 1: harmonic fit
function run_fit()
    t0 = time()
    hfiles = sort(filter(x -> startswith(x, "roms_his.") && endswith(x, ".nc"), readdir(dirR)))
    TL = Dict{Int, Tuple{Int, Int, Float64}}()                 # hour -> (file, step, ocean_time)
    for (fi, f) in enumerate(hfiles)
        t = NCDataset(r -> r["ocean_time"].var[:], dirR * f)
        for (k, tk) in enumerate(t); TL[round(Int, tk / 3600)] = (fi, k, tk); end
    end
    hs = sort(collect(keys(TL)))
    hs = filter(h -> h >= hs[1] + round(Int, 24SPINUP), hs)
    TEST && (hs = filter(h -> h < hs[1] + 30 * 24, hs))
    tsec = [TL[h][3] for h in hs]; nt = length(hs)
    tdate(t) = TREF + Second(round(Int, t))
    @printf("timeline: %d records, %s .. %s, %d gaps (largest %d h), dt %.0f-%.0f s\n", nt, tdate(tsec[1]), tdate(tsec[end]),
            count(>(1), diff(hs)), maximum(diff(hs)), extrema(diff(tsec))...); flush(stdout)

    ## nodal arguments (pyTMD, OTIS conventions); times passed as seconds since 2000-01-01
    t2000 = tsec .- Dates.value(Second(DateTime(2000) - TREF))
    ftim = dirout * "times$(SUF).nc"; fnod = dirout * "nodal_args$(SUF).nc"
    isfile(ftim) && rm(ftim)
    NCDataset(ftim, "c"; format = :netcdf3_64bit_offset) do ds
        defDim(ds, "time", nt); defVar(ds, "t", t2000, ("time",))
    end
    run(`$PY $NODPY $ftim $fnod $(join(ALLC, ","))`)
    Gd, pud, pfd, pf_c, pu_c = NCDataset(fnod) do nd
        permutedims(nd["G"][:, :]), permutedims(nd["pu"][:, :]), permutedims(nd["pf"][:, :]), vec(nd["pf_c"][:, 1]), vec(nd["pu_c"][:, 1])
    end
    ic(c) = findfirst(==(c), ALLC)
    K = length(FIT); P = 2 + 2K
    tc = (tsec[1] + tsec[end]) / 2; span = (tsec[end] - tsec[1]) / 2
    Eb(c) = pfd[:, ic(c)] .* cis.(deg2rad.(Gd[:, ic(c)] .+ pud[:, ic(c)]))     # f e^{i(V+u)}: eta = Re{Z Eb}

    ## design matrix; inferred constituents are folded into the basis of their reference
    A = zeros(nt, P); A[:, 1] .= 1.0; A[:, 2] .= (tsec .- tc) ./ span
    for (k, c) in enumerate(FIT)
        E = Eb(c)
        for (ci, cb) in INFER; cb == c && (E .+= RINF[ci] .* Eb(ci)); end
        A[:, 2k + 1] .= real.(E); A[:, 2k + 2] .= imag.(E)
    end
    GA = Symmetric(A' * A)
    @printf("cond(AtA) %.1f, %d parameters\n", cond(GA), P); flush(stdout)

    ## stream SSH
    byfile = Dict{Int, Vector{Tuple{Int, Int}}}()                 # file -> [(step, row)]
    for (n, h) in enumerate(hs); (fi, k, _) = TL[h]; push!(get!(byfile, fi, Tuple{Int, Int}[]), (k, n)); end
    npts = NX * NY
    B = zeros(npts, P); ss = zeros(npts)
    for (q, (fi, v)) in enumerate(sort(collect(byfile)))
        v = sort(v); steps = first.(v); rows = last.(v)
        Z = NCDataset(r -> r["zeta"].var[:, :, steps[1]:steps[end]], dirR * hfiles[fi])
        Y = Float64.(reshape(Z[:, :, steps .- steps[1] .+ 1], npts, :))
        mul!(B, Y, A[rows, :], 1.0, 1.0); ss .+= vec(sum(abs2, Y; dims = 2))
        q % 50 == 0 && (@printf("  %d / %d files, %.1f min\n", q, length(byfile), (time() - t0) / 60); flush(stdout))
    end

    ## solve, residuals, white-noise errors
    C = Matrix((cholesky(GA) \ B')')
    rss = max.(ss .- vec(sum(C .* B; dims = 2)), 0.0); sig = sqrt.(rss ./ (nt - P))
    sstd = sqrt.(max.(ss ./ nt .- (B[:, 1] ./ nt) .^ 2, 0.0))
    Gi = inv(GA)
    rs(x) = reshape(x, NX, NY)
    CON = uppercase.(ALLC); KA = length(CON)
    Zall = zeros(ComplexF64, NX, NY, KA); Aer = zeros(NX, NY, KA)
    for (k, c) in enumerate(FIT)
        Zall[:, :, k] .= 100 .* rs(complex.(C[:, 2k + 1], -C[:, 2k + 2]))
        Aer[:, :, k] .= 100 .* rs(sig) .* sqrt((Gi[2k + 1, 2k + 1] + Gi[2k + 2, 2k + 2]) / 2)
    end
    for (ci, cb) in INFER
        Zall[:, :, ic(ci)] .= RINF[ci] .* Zall[:, :, ic(cb)]; Aer[:, :, ic(ci)] .= abs(RINF[ci]) .* Aer[:, :, ic(cb)]
    end

    isfile(fhc) && rm(fhc)
    NCDataset(fhc, "c") do ds
        defDim(ds, "xi_rho", NX); defDim(ds, "eta_rho", NY); defDim(ds, "con", KA)
        ds.attrib["title"] = "Harmonic constants of hourly SSH, ROMS Amazon 900 m, $(Date(tdate(tsec[1]))) to $(Date(tdate(tsec[end])))"
        ds.attrib["convention"] = "eta = f A cos(V + u - G); Z = A exp(-iG) (TPXO hRe + i hIm); G = Greenwich phase lag [deg], V, f, u from pyTMD (OTIS)"
        ds.attrib["model"] = "mean + linear trend + constituents, ordinary least squares, f(t), u(t), V(t) hourly"
        ds.attrib["inferred"] = join(["$(uppercase(ci)) = R $(uppercase(cb)) (R from TPXO10 v2a)" for (ci, cb) in INFER], "; ")
        ds.attrib["records"] = nt; ds.attrib["t_start"] = string(tdate(tsec[1])); ds.attrib["t_end"] = string(tdate(tsec[end]))
        ds.attrib["script"] = "internalwave_julia/ROMS_TPXO_tidalcomp.jl"
        defVar(ds, "con", CON, ("con",))
        defVar(ds, "inferred", Int8[c in first.(INFER) for c in ALLC], ("con",))
        defVar(ds, "f_c", pf_c, ("con",); attrib = ["long_name" => "nodal f at record centre"])
        defVar(ds, "u_c", pu_c, ("con",); attrib = ["units" => "deg", "long_name" => "nodal u at record centre"])
        defVar(ds, "lon_rho", lonR, ("xi_rho", "eta_rho")); defVar(ds, "lat_rho", latR, ("xi_rho", "eta_rho"))
        defVar(ds, "h", hR, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
        defVar(ds, "A", Float32.(abs.(Zall)), ("xi_rho", "eta_rho", "con"); attrib = ["units" => "cm", "long_name" => "amplitude"], deflatelevel = 3)
        defVar(ds, "G", Float32.(mod.(-rad2deg.(angle.(Zall)), 360)), ("xi_rho", "eta_rho", "con"); attrib = ["units" => "deg", "long_name" => "Greenwich phase lag"], deflatelevel = 3)
        defVar(ds, "A_err", Float32.(Aer), ("xi_rho", "eta_rho", "con"); attrib = ["units" => "cm", "long_name" => "white-noise std error of A"], deflatelevel = 3)
        for (nm, v, u, ln) in (("mean", 100 .* rs(C[:, 1]), "cm", "mean SSH"), ("trend", 100 .* rs(C[:, 2]) ./ (span / (365.25 * 86400)), "cm/yr", "linear SSH trend"),
                               ("resid_std", 100 .* rs(sig), "cm", "residual std"), ("ssh_std", 100 .* rs(sstd), "cm", "SSH std"))
            defVar(ds, nm, Float32.(v), ("xi_rho", "eta_rho"); attrib = ["units" => u, "long_name" => ln], deflatelevel = 3)
        end
    end
    @printf("wrote %s (%.1f min)\n", fhc, (time() - t0) / 60)
    for (k, c) in enumerate(CON)
        @printf("  %-4s%s f_c %.4f  u_c %6.2f   median A %6.2f cm  median A_err %.3f cm\n", c, ALLC[k] in first.(INFER) ? "*" : " ",
                pf_c[k], pu_c[k], median(abs.(Zall[:, :, k])), median(Aer[:, :, k]))
    end
    @printf("  median residual std %.2f cm, median SSH std %.2f cm   (* inferred)\n", median(100 .* sig), median(100 .* sstd)); flush(stdout)
end
(REFIT || !isfile(fhc)) ? run_fit() : println("using existing $fhc (REFIT=true to fit again)")

## --- step 2: comparison
CON, ZR, INF, resid, sshstd, Aerr, tlab = NCDataset(fhc) do ds
    CON = String.(ds["con"][:])
    CON, Float64.(ds["A"][:, :, :]) .* cis.(-deg2rad.(Float64.(ds["G"][:, :, :]))), Bool.(ds["inferred"][:]),
    Float64.(ds["resid_std"][:, :]), Float64.(ds["ssh_std"][:, :]), Float64.(ds["A_err"][:, :, :]),
    "$(ds.attrib["t_start"][1:10])..$(ds.attrib["t_end"][1:10])"
end
kc(c) = findfirst(==(c), CON)

## 2-D Gaussian low-pass by normalised convolution with weights Wf (the filter sees only the comparison
## area; edges and masked points are renormalised): LP(Z) = G*(Wf Z) / G*(Wf). Separable, truncated at
## 4 sigma; sigma = lc / 5.336 gives a response exp(-2 pi^2 sigma^2 / lambda^2) of 1/2 at lambda = lc
function conv1!(out, y, k, hw)
    n = length(y)
    @inbounds for i in 1:n
        acc = zero(eltype(y))
        for j in max(1, i - hw):min(n, i + hw)
            acc += k[j - i + hw + 1] * y[j]
        end
        out[i] = acc
    end
end
function conv2(X, s)
    hw = ceil(Int, 4s); k = exp.(-0.5 .* ((-hw:hw) ./ s) .^ 2)
    T = similar(X); Y = similar(X)
    Threads.@threads for j in axes(X, 2); conv1!(view(T, :, j), X[:, j], k, hw); end
    Threads.@threads for i in axes(X, 1); conv1!(view(Y, i, :), T[i, :], k, hw); end
    return Y
end
const Wf = Float64.(M)
const WfLP = Dict{Float64, Matrix{Float64}}()
function gfilt(Z, lc)
    s = lc / 5.336 / dxkm
    den = get!(() -> conv2(Wf, s), WfLP, lc)
    return conv2(Wf .* Z, s) ./ den
end
lcof(c) = c in DIURNAL ? LC_D : LC_SD

rows = []
LP = Dict{String, Matrix{ComplexF64}}()
for (k, c) in enumerate(CON)
    haskey(TP, c) || continue
    lc = lcof(c); t10 = TP[c]; zr = ZR[:, :, k]
    zl = t10 .+ gfilt(zr .- t10, lc); LP[c] = zl                        # ROMS barotropic
    push!(rows, Dict{Symbol, Any}(:con => c, :inf => INF[k], :A10 => wmean(abs.(t10)), :S => srms(t10),
        :rms => rmsd(zl, t10), :ar => aratio(zl, t10), :dg => dphase(zl, t10),
        :rms_sh => rmsd(zl, t10, Wsh), :S_sh => srms(t10, Wsh), :rms_dp => rmsd(zl, t10, Wdp), :S_dp => srms(t10, Wdp),
        :rms_full => rmsd(zr, t10),                                        # ROMS unfiltered
        :rms_lpR => rmsd(gfilt(zr, lc), t10),                              # ROMS alone low-passed
        :Tlp => rmsd(gfilt(t10, lc), t10),                                 # what the low-pass does to TPXO
        :it => srms(zr .- zl),                                             # coherent IT SSH rms
        :aerr => median(Aerr[:, :, k][M])))
end

io = IOBuffer()
@printf(io, "ROMS Amazon 900 m (%s, hourly zeta) vs TPXO10 v2a 1/6 deg; comparison area: %.0f km sponge removed, TPXO-wet points\n", tlab, SPONGE)
@printf(io, "ROMS_bt = TPXO10 + LP(ROMS - TPXO10), LP Gaussian half power at %.0f km (semidiurnal and higher) / %.0f km (diurnal)\n", LC_SD, LC_D)
@printf(io, "RMS = sqrt(<|dZ|^2/2>) [cm]; S = TPXO10 signal rms; ratio = <A_ROMS>/<A_TPXO>; dG = ROMS - TPXO [deg], + = ROMS lags\n")
@printf(io, "shelf: h < %.0f m, deep: h >= %.0f m; RMSfull: ROMS unfiltered; RMS_lpR: ROMS alone low-passed; Tlp-raw: rms change of TPXO10 by the low-pass;\n", HSHELF, HSHELF)
@printf(io, "IT: rms of coherent internal-tide SSH (ROMS - ROMS_bt); A_err: median white-noise error of the ROMS amplitude; * inferred from TPXO ratio\n\n")
@printf(io, "%-5s %7s %7s | %7s %6s %6s %7s | %7s %6s %7s %6s | %7s %7s %7s | %6s %6s\n", "con", "<A10>", "S", "RMS", "relRMS", "ratio", "dG",
        "RMSshlf", "rel", "RMSdeep", "rel", "RMSfull", "RMS_lpR", "Tlp-raw", "IT", "A_err")
for r in rows
    @printf(io, "%-5s %7.3f %7.3f | %7.3f %5.1f%% %6.3f %+7.2f | %7.3f %5.1f%% %7.3f %5.1f%% | %7.3f %7.3f %7.3f | %6.2f %6.3f\n",
            r[:con] * (r[:inf] ? "*" : ""), r[:A10], r[:S], r[:rms], 100 * r[:rms] / r[:S], r[:ar], r[:dg],
            r[:rms_sh], 100 * r[:rms_sh] / r[:S_sh], r[:rms_dp], 100 * r[:rms_dp] / r[:S_dp], r[:rms_full], r[:rms_lpR], r[:Tlp], r[:it], r[:aerr])
end
mj = [r for r in rows if r[:con] in MAJOR]
rss(key) = sqrt(sum(r[key]^2 for r in mj))
@printf(io, "\nRSS over %s: ROMS_bt %.3f (shelf %.3f, deep %.3f), ROMS unfiltered %.3f, ROMS alone low-passed %.3f cm\n",
        join([r[:con] for r in mj], " "), rss(:rms), rss(:rms_sh), rss(:rms_dp), rss(:rms_full), rss(:rms_lpR))
@printf(io, "median residual std %.2f cm, median SSH std %.2f cm (comparison area)\n", median(resid[M]), median(sshstd[M]))
@printf(io, "\nRMS [cm] of LP(ROMS - TPXO10) vs half-power cutoff [km] (the internal tide drops out as the cutoff grows):\n")
for c in MAPC
    k = kc(c); (k === nothing || !haskey(TP, c)) && continue
    D = ZR[:, :, k] .- TP[c]
    @printf(io, "  %-3s %s\n", c, join([@sprintf("%5.0f: %.3f", L, srms(gfilt(D, L))) for L in (200, 300, 400, 600, 800)], "  "))
end
tab = String(take!(io)); print(tab); write(dirout * "ROMS_TPXO_rms$(SUF).txt", tab)

## --- figures, in model coordinates (km along xi / eta)
cm_to_pt = 72 / 2.54
xk = dxkm .* (0:NX-1); yk = dykm .* (0:NY-1)
xi = nsx+1:NX-nsx; yi = nsy+1:NY-nsy
bx = xk[[xi[1], xi[end], xi[end], xi[1], xi[1]]]; by = yk[[yi[1], yi[1], yi[end], yi[end], yi[1]]]
function deco!(a; box = true)
    contour!(a, xk, yk, lonR; levels = floor(minimum(lonR)):1:ceil(maximum(lonR)), color = (:gray30, 0.5), linewidth = 0.4,
             labels = true, labelsize = 6, labelcolor = :gray30, labelformatter = l -> string.(round.(Int, 360 .- l), "°W"))
    contour!(a, xk, yk, latR; levels = floor(minimum(latR)):1:ceil(maximum(latR)), color = (:gray30, 0.5), linewidth = 0.4,
             labels = true, labelsize = 6, labelcolor = :gray30, labelformatter = l -> string.(round.(Int, l), "°N"))
    contour!(a, xk, yk, hR; levels = [HSHELF], color = :black, linewidth = 0.7)
    box && lines!(a, bx, by; color = :gray40, linewidth = 0.8, linestyle = :dot)
end
nanout(X) = ifelse.(WET, X, NaN)
q(X, p) = quantile(filter(isfinite, vec(X[M])), p)
gm_unw(zr) = (gm = rad2deg(angle(sum(W .* zr))); z -> (G = -rad2deg.(angle.(z)); mod.(G .+ gm .+ 180, 360) .- 180 .- gm))   # phase unwrapped about the area mean
mapsuf = "ROMS Amazon 900 m ($tlab) vs TPXO10 v2a (1/6°)"

## figure 1: ROMS barotropic amplitude + co-phase lines, amplitude and phase differences
mc = [c for c in MAPC if haskey(LP, c)]
fig = Figure(size = (24cm_to_pt, (2 + 8.6length(mc))cm_to_pt), fontsize = 9)
for (row, c) in enumerate(mc)
    zr = LP[c]; zt = TP[c]; unw = gm_unw(zr)
    Ar = abs.(zr); Gr = unw(zr); Gt = unw(zt)
    dA = nanout(Ar .- abs.(zt)); dG = nanout(-rad2deg.(angle.(zr .* conj.(zt))))
    dAm = max(q(abs.(dA), 0.99), 0.05); dGm = max(q(abs.(dG), 0.99), 1.0)
    lev = (floor(minimum(Gr) / 10) * 10):10:(ceil(maximum(Gr) / 10) * 10)
    kw = (; xlabelvisible = row == length(mc), xticklabelsvisible = row == length(mc), aspect = DataAspect())
    a1 = Axis(fig[row, 1]; title = "$c amplitude (ROMS bt) [cm], phase every 10°", ylabel = "y [km]", xlabel = "x [km]", kw...)
    h1 = heatmap!(a1, xk, yk, nanout(Ar); colormap = :Blues, colorrange = (0, q(Ar, 0.995)))
    contour!(a1, xk, yk, Gr; levels = lev, color = :black, linewidth = 0.7)
    contour!(a1, xk, yk, nanout(Gt); levels = lev, color = :darkorange, linewidth = 0.7, linestyle = :dash)
    Colorbar(fig[row, 2], h1; height = Relative(0.8))
    a2 = Axis(fig[row, 3]; title = "$c amplitude ROMS - TPXO10 [cm]", xlabel = "x [km]", yticklabelsvisible = false, kw...)
    h2 = heatmap!(a2, xk, yk, dA; colormap = :balance, colorrange = (-dAm, dAm))
    Colorbar(fig[row, 4], h2; height = Relative(0.8))
    a3 = Axis(fig[row, 5]; title = "$c phase ROMS - TPXO10 [deg]", xlabel = "x [km]", yticklabelsvisible = false, kw...)
    h3 = heatmap!(a3, xk, yk, dG; colormap = :balance, colorrange = (-dGm, dGm))
    Colorbar(fig[row, 6], h3; height = Relative(0.8))
    foreach(deco!, (a1, a2, a3))
end
Label(fig[0, 1:6], "$mapsuf; ROMS barotropic = TPXO10 + low-pass(ROMS - TPXO10), half power at $(Int(LC_SD)) km (semidiurnal), $(Int(LC_D)) km (diurnal).\n" *
      "Left: co-phase lines ROMS black, TPXO10 orange dashed; thick black: $(Int(HSHELF)) m isobath; dotted box: comparison area; grey: lon/lat.", fontsize = 9, tellwidth = false)
colgap!(fig.layout, 6); rowgap!(fig.layout, 4)
save(dirout * "ROMS_TPXO_maps$(SUF).png", fig; px_per_unit = 300 / 72)

## figure 2: RMS per constituent
rows2 = [r for r in rows if r[:con] in MAJOR || r[:con] in ("M4", "MS4", "MN4")]; CB = [r[:con] * (r[:inf] ? "*" : "") for r in rows2]; nb = length(CB)
cols = ["#2a78d6", "#eb6834", "#1baf7a"]
fig2 = Figure(size = (17cm_to_pt, 13cm_to_pt), fontsize = 10)
ax = Axis(fig2[1, 1]; ylabel = "RMS difference [cm]", xticks = (1:nb, CB), title = "$mapsuf, comparison area",
          yscale = log10, yminorticksvisible = true, yminorgridvisible = true)
ser = [("ROMS barotropic vs TPXO10, shelf (h < $(Int(HSHELF)) m)", :rms_sh), ("ROMS barotropic vs TPXO10, deep", :rms_dp), ("ROMS vs TPXO10, unfiltered", :rms_full)]
for (s, (lab, key)) in enumerate(ser)
    barplot!(ax, (1:nb) .+ (s - 2) * 0.27, [max(r[key], 1e-3) for r in rows2]; width = 0.25, color = cols[s], label = lab, fillto = 1e-3, strokewidth = 0)
end
scatter!(ax, 1:nb, [r[:S] for r in rows2]; marker = :hline, markersize = 16, color = :black, label = "TPXO10 signal rms")
ylims!(ax, 1e-2, 2 * maximum(r[:S] for r in rows2))
Legend(fig2[2, 1], ax; orientation = :horizontal, nbanks = 2, framevisible = false, labelsize = 9)
ax2 = Axis(fig2[3, 1]; ylabel = "RMS / signal rms [%]", xticks = (1:nb, CB))
rel = [100 * r[:rms] / r[:S] for r in rows2]
barplot!(ax2, 1:nb, rel; color = cols[1], strokewidth = 0)
text!(ax2, 1:nb, rel; text = [@sprintf("%.0f", v) for v in rel], align = (:center, :bottom), offset = (0, 2), fontsize = 8, color = "#52514e")
ylims!(ax2, 0, 1.25 * maximum(rel))
Label(fig2[4, 1], "* inferred from S2 / K1 / N2 with the TPXO10 ratio (record too short to separate)", fontsize = 8, tellwidth = false)
rowsize!(fig2.layout, 3, Relative(0.25))
save(dirout * "ROMS_TPXO_rms$(SUF).png", fig2; px_per_unit = 300 / 72)

## figure 3: ROMS coherent M2 internal-tide SSH (full - barotropic)
zit = ZR[:, :, kc("M2")] .- LP["M2"]
fig3 = Figure(size = (13cm_to_pt, 14cm_to_pt), fontsize = 10)
a = Axis(fig3[1, 1]; title = "ROMS coherent M2 internal-tide SSH amplitude [cm]\n(full - barotropic, $(Int(LC_SD)) km cutoff), $tlab", xlabel = "x [km]", ylabel = "y [km]", aspect = DataAspect())
h = heatmap!(a, xk, yk, nanout(abs.(zit)); colormap = :Blues, colorrange = (0, q(abs.(zit), 0.995)))
deco!(a)
Colorbar(fig3[1, 2], h)
save(dirout * "ROMS_M2_ITssh$(SUF).png", fig3; px_per_unit = 300 / 72)

## figure 4: 2 x 2 maps of the barotropic rms difference sqrt(|dZ|^2 / 2)
fig4 = Figure(size = (22cm_to_pt, 24cm_to_pt), fontsize = 10)
for (n, c) in enumerate(mc)
    r = rows[findfirst(x -> x[:con] == c, rows)]
    E = nanout(abs.(LP[c] .- TP[c]) ./ sqrt(2))
    row, col = fldmod1(n, 2)
    a4 = Axis(fig4[row, 2col - 1]; title = @sprintf("%s: area rms %.2f cm (%.0f%% of signal rms)", c, r[:rms], 100 * r[:rms] / r[:S]),
              xlabel = "x [km]", ylabel = "y [km]", aspect = DataAspect(), xlabelvisible = row == 2, xticklabelsvisible = row == 2,
              ylabelvisible = col == 1, yticklabelsvisible = col == 1)
    h4 = heatmap!(a4, xk, yk, E; colormap = :Oranges, colorrange = (0, q(E, 0.995)))
    deco!(a4)
    Colorbar(fig4[row, 2col], h4; label = "rms difference [cm]", height = Relative(0.8))
end
Label(fig4[0, 1:4], "Barotropic tide: rms difference sqrt(|Z_ROMS - Z_TPXO10|² / 2), $mapsuf\n" *
      "ROMS barotropic = TPXO10 + low-pass(ROMS - TPXO10), $(Int(LC_SD)) / $(Int(LC_D)) km; dotted box: comparison area ($(Int(SPONGE)) km sponge removed)", fontsize = 9, tellwidth = false)
colgap!(fig4.layout, 6); rowgap!(fig4.layout, 6)
save(dirout * "ROMS_TPXO_rms_maps$(SUF).png", fig4; px_per_unit = 300 / 72)

## figure 5: 2 x 2 co-tidal maps, ROMS barotropic amplitude and co-phase lines of ROMS (solid, labelled)
##           and TPXO10 (dashed); phase unwrapped about the area mean, labels mod 360
fig5 = Figure(size = (22cm_to_pt, 24cm_to_pt), fontsize = 10)
for (n, c) in enumerate(mc)
    zr = LP[c]; zt = TP[c]; unw = gm_unw(zr)
    Gr = unw(zr); Gt = nanout(unw(zt)); Ar = nanout(abs.(zr))
    rng = -(reverse(quantile(vec(Gr[M]), [0.01, 0.99]))...)
    dl = first(d for d in (1, 2, 5, 10, 15, 30) if rng / d <= 12)
    lev = (floor(minimum(Gr) / dl) * dl):dl:(ceil(maximum(Gr) / dl) * dl)
    row, col = fldmod1(n, 2)
    a5 = Axis(fig5[row, 2col - 1]; title = "$c: amplitude [cm], Greenwich phase every $(dl)°", xlabel = "x [km]", ylabel = "y [km]",
              aspect = DataAspect(), xlabelvisible = row == 2, xticklabelsvisible = row == 2, ylabelvisible = col == 1, yticklabelsvisible = col == 1)
    h5 = heatmap!(a5, xk, yk, Ar; colormap = :Blues, colorrange = (0, q(Ar, 0.999)))
    deco!(a5; box = false)
    contour!(a5, xk, yk, Gt; levels = lev, color = :darkorange, linewidth = 0.9, linestyle = :dash)
    contour!(a5, xk, yk, Gr; levels = lev, color = :black, linewidth = 0.9, labels = true, labelsize = 8,
             labelformatter = l -> string.(round.(Int, mod.(l, 360))))
    Colorbar(fig5[row, 2col], h5; label = "amplitude [cm]", height = Relative(0.8))
end
Label(fig5[0, 1:4], "Barotropic co-tidal maps, $mapsuf\nROMS = TPXO10 + low-pass(ROMS - TPXO10) (black, labelled); TPXO10 co-phase lines dashed orange; thick black: $(Int(HSHELF)) m isobath",
      fontsize = 9, tellwidth = false)
colgap!(fig5.layout, 6); rowgap!(fig5.layout, 6)
save(dirout * "ROMS_TPXO_cotidal$(SUF).png", fig5; px_per_unit = 300 / 72)
println("figures written to $dirout")
