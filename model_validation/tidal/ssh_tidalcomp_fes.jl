#= ssh_tidalcomp_fes.jl
Hsin-Yi Chen, USM DMS
Compare the ROMS SSH harmonic constants from ssh_harmonic.jl with FES2022b.
Rewritten section by section from ROMS_TPXO_tidalcomp.jl (step 2).

Section 1: read the plain sinusoid fit (ROMS_<RUN>_sshfit[_s<STRIDE>].nc):
  A' [cm], g' [deg, phase at the record centre tc], their errors, mean, residual std, SSH std,
  lon/lat/h of the fitted points; complex Z' = A' exp(-i g').
Section 2: FES2022b Z = A exp(-iG) for the fitted constituents, bilinear to the fitted points (ZF), and the
  complex FES ratios R = Z_K2 / Z_S2, Z_NU2 / Z_N2, Z_P1 / Z_K1 (area mean).
Section 3: ROMS to the atlas convention, A = A' / f(tc), G = g' + V(tc) + u(tc) (ZR); V, u, f at tc from
  pyTMD (nodal_args.py, FES convention). 95 days do not separate K2, NU2, P1 from S2, N2, K1, so those three
  are corrected after the conversion with the FES ratios R (inference, as t_tide / UTide).
Section 4: comparison area (sponge removed, FES-wet points), cell-area weights W = 1 / (pm pn), and the
  barotropic / internal-tide split: ZB = ZF + LP(ZR - ZF), ZIT = ZR - ZB. LP = 2-D Gaussian low-pass (half power
  at LC_SD / LC_D), so the sharp shelf-break tide, which ZR and ZF share, is not smeared by the filter.
Section 5: area statistics of ZB vs ZF: RMS = sqrt(< |dZ|^2 / 2 >), signal rms S, amplitude ratio, phase
  difference, shelf and deep RMS; RMSfull for ZR (unfiltered), the rms of ZIT; RSS, RMS vs cutoff, phase gradient.
  All printed output is collected in io and written to data_folder/ROMS_FES_sshfit_rms[_s<STRIDE>].txt at the end.
Section 6: save everything the figures need to data_folder/ROMS_FES_sshfit_comp[_s<STRIDE>].nc (no plotting here).

Usage:
  julia ssh_tidalcomp_fes.jl              # full-grid fit
  STRIDE=20 julia ssh_tidalcomp_fes.jl    # the every-20th-point test fit (_s20)
=#

using NCDatasets, Printf, Statistics, Dates, LinearAlgebra
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")   # time_dir, data_load, harm_fit, figkw, ...

## --- settings
const data_folder = "/expanse/lustre/projects/uso102/hchen54/figure/amazon_900m_3mon_1008/tidal_comp/"
const tidal_path  = "/expanse/lustre/projects/uso102/hchen54/input_cmems/fes2022/ocean_tide_extrapolated/"   # FES2022b: <con>_fes2022.nc, mask_fes2022B.nc
const RUN         = "amazon900m"
const STRIDE      = parse(Int, get(ENV, "STRIDE", "1"))   # which fit to read (1 = full grid)
const fname       = data_folder * "ROMS_$(RUN)_sshfit$(STRIDE > 1 ? "_s$STRIDE" : "").nc"
const t_ref       = DateTime(1994, 1, 1)                  # ocean_time reference of the fit's tc
const PY          = "/home/hchen54/.venvs/utide/bin/python"                     # python with pyTMD
const NODPY       = "/home/hchen54/internalwave_julia/function/nodal_args.py"  # V, u, f from pyTMD
const NODAL       = "FES"                                 # nodal convention of the atlas (FES2022)
const INFER       = [("K2", "S2"), ("NU2", "N2"), ("P1", "K1")]   # (absorbed, fitted): not separated by the record
const grid_file   = "/expanse/lustre/projects/uso102/hchen54/input/grid/roms_grd_900m.nc"
const SPONGE      = 27.0                                  # km removed along each open boundary
const HSHELF      = 200.0                                 # m, shelf / deep-water split
const LC_SD       = 400.0                                 # km, half-power wavelength of the low-pass, semidiurnal
const LC_D        = 800.0                                 # km, diurnal
const DIURNAL     = ["K1", "O1"]                          # fitted constituents using LC_D
const LCP_SD      = 300.0                                 # km, low-pass of ROMS alone for the deep co-phase lines, semidiurnal
const LCP_D       = 600.0                                 # km, diurnal
const HDEEP       = 2000.0                                # m, deep water for the co-phase lines
const ftab        = data_folder * "ROMS_FES_sshfit_rms$(STRIDE > 1 ? "_s$STRIDE" : "").txt"
const fcomp       = data_folder * "ROMS_FES_sshfit_comp$(STRIDE > 1 ? "_s$STRIDE" : "").nc"
const io          = IOBuffer()                            # everything printed, written to ftab at the end
const fold        = data_folder * "ROMS_$(RUN)_harmconst_fes.nc"   # old fit (ROMS_TPXO_tidalcomp.jl), for the check

## --- section 1: read the ROMS fit (Julia order: xi, eta, con)
isfile(fname) || error("no fit file $fname (run ssh_harmonic.jl first)")
hfit_data = NCDataset(fname) do ds
    (; cons    = String.(ds["con"][:]),
       freq    = Float64.(ds["freq"][:]),                         # deg / hour
       lon     = Float64.(ds["lon_rho"][:, :]), lat = Float64.(ds["lat_rho"][:, :]), h = Float64.(ds["h"][:, :]),
       amp     = Float64.(ds["amp"][:, :, :]),     pha  = Float64.(ds["pha"][:, :, :]),      # A' [cm], g' [deg]
       aerr    = Float64.(ds["amp_err"][:, :, :]), perr = Float64.(ds["pha_err"][:, :, :]),
       mean    = Float64.(ds["mean"][:, :]),  resid = Float64.(ds["resid_std"][:, :]), sshstd = Float64.(ds["ssh_std"][:, :]),
       tc      = Float64(ds.attrib["tc_seconds_since_1994"]),     # record centre [s since t_ref]
       tcdate  = DateTime(ds.attrib["tc"]),
       records = Int(ds.attrib["records"]), title = ds.attrib["title"])
end
NX, NY = size(hfit_data.lon); K = length(hfit_data.cons)
kc(c) = findfirst(==(c), hfit_data.cons)                                # index of a constituent
Zp = hfit_data.amp .* cis.(-deg2rad.(hfit_data.pha))                          # Z' = A' exp(-i g') [cm], (xi, eta, con)
# cis(x) = exp(i·x) = cos(x) + i·sin(x) 
# --> Zp = hfit_data.amp .* exp.(-im .* deg2rad.(hfit_data.pha))

println(io, hfit_data.title)
@printf(io, "fit file %s\n  grid %d x %d, %d records, tc = %s, lon %.2f..%.2f, lat %.2f..%.2f\n",
        fname, NX, NY, hfit_data.records, hfit_data.tcdate, extrema(hfit_data.lon)..., extrema(hfit_data.lat)...)
for (k, c) in enumerate(hfit_data.cons)
    @printf(io, "  %-3s %6.2f h   median A' %6.2f cm (err %.3f)   median g' err %.2f deg\n", c, 360 / hfit_data.freq[k],
            median(filter(!isnan, hfit_data.amp[:, :, k])), median(filter(!isnan, hfit_data.aerr[:, :, k])), median(filter(!isnan, hfit_data.perr[:, :, k])))
end
@printf(io, "  median residual std %.2f cm, SSH std %.2f cm\n", median(filter(!isnan, hfit_data.resid)), median(filter(!isnan, hfit_data.sshstd)))

## --- section 2: FES2022b for the fitted constituents, at the fitted points
## <con>_fes2022.nc: amplitude [cm], phase [deg] (Greenwich phase lag G), Julia order (lon, lat), uniform 1/30 deg,
## lon 0..360; land / no data = fill ~1.8e19. Only the window around the ROMS points is read.
## Z = A exp(-iG) [cm], bilinear interpolation to (lon, lat); wet = all four surrounding nodes native ocean
## (mask_fes2022B.nc: 0 = ocean, 1 = extrapolated, 2 = land, 3 = lake); not wet -> NaN.
function fes_read(dir, cons, lon, lat)
    fn(c) = dir * "$(lowercase(c))_fes2022.nc"
    for c in cons; isfile(fn(c)) || error("no FES file for $c: $(fn(c))"); end
    lonf, latf = NCDataset(ds -> (ds["lon"][:], ds["lat"][:]), fn(cons[1]))
    ia = searchsortedlast(lonf, minimum(lon)) - 1; ib = searchsortedfirst(lonf, maximum(lon)) + 1   # window, 1-node margin
    ja = searchsortedlast(latf, minimum(lat)) - 1; jb = searchsortedfirst(latf, maximum(lat)) + 1
    dl = (lonf[ib] - lonf[ia]) / (ib - ia); dt = (latf[jb] - latf[ja]) / (jb - ja)
    fx = (lon .- lonf[ia]) ./ dl .+ 1; fy = (lat .- latf[ja]) ./ dt .+ 1   # fractional window index of each point
    i0 = floor.(Int, fx); j0 = floor.(Int, fy); wx = fx .- i0; wy = fy .- j0
    # bil: helper for the interpolation, bilinear value of window field z at ROMS point p (4 surrounding nodes)
    bil(z, p) = (1 - wx[p]) * (1 - wy[p]) * z[i0[p], j0[p]] + wx[p] * (1 - wy[p]) * z[i0[p] + 1, j0[p]] +
                (1 - wx[p]) * wy[p] * z[i0[p], j0[p] + 1] + wx[p] * wy[p] * z[i0[p] + 1, j0[p] + 1]

    # land masking: oc = FES node is native ocean (mask 0; extrapolated, land, lake excluded),
    # wet = ROMS point whose 4 surrounding nodes are all ocean (only those are interpolated, others NaN)
    oc = NCDataset(ds -> ds["mask"].var[ia:ib, ja:jb] .== 0, dir * "mask_fes2022B.nc")
    wet = [oc[i0[p], j0[p]] && oc[i0[p] + 1, j0[p]] && oc[i0[p], j0[p] + 1] && oc[i0[p] + 1, j0[p] + 1] for p in CartesianIndices(lon)]
    Z = fill(complex(NaN, NaN), size(lon)..., length(cons))
    for (k, c) in enumerate(cons)
        z = NCDataset(fn(c)) do ds
            a = Float64.(ds["amplitude"].var[ia:ib, ja:jb]); g = Float64.(ds["phase"].var[ia:ib, ja:jb])
            ifelse.(a .< 1e10, a .* cis.(-deg2rad.(g)), 0.0im)
        end
        for p in CartesianIndices(lon); wet[p] && (Z[p, k] = bil(z, p)); end
    end
    return Z, wet
end

allcons = vcat(hfit_data.cons, first.(INFER))                            # fitted, then absorbed constituents
ZFall, wetF = fes_read(tidal_path, allcons, hfit_data.lon, hfit_data.lat)   # (xi, eta, con) [cm], (xi, eta)
ZF = ZFall[:, :, 1:K]                                                    # the fitted constituents, compared with ROMS
ka(c) = findfirst(==(c), allcons)                                        # index in allcons
@printf(io, "FES2022b: %d constituents at %d points, %d wet (%.1f%%)\n", K, NX * NY, count(wetF), 100 * count(wetF) / (NX * NY))
for (k, c) in enumerate(hfit_data.cons)
    @printf(io, "  %-3s median A %6.2f cm\n", c, median(abs.(filter(!isnan, ZF[:, :, k]))))
end
## FES ratios for the inference: complex area mean over the wet points, R = <Z_a conj(Z_b)> / <|Z_b|^2>
RINF = Dict(ci => sum(ZFall[:, :, ka(ci)][wetF] .* conj.(ZFall[:, :, ka(cb)][wetF])) / sum(abs2.(ZFall[:, :, ka(cb)][wetF])) for (ci, cb) in INFER)
for (ci, cb) in INFER
    @printf(io, "  inference %-3s = R %-2s: |R| %.4f, arg %.2f deg\n", ci, cb, abs(RINF[ci]), rad2deg(angle(RINF[ci])))
end

## --- section 3: ROMS to the atlas convention (Greenwich phase, nodal-free amplitude)
## The fit gives A' cos(w (t - tc) - g'); the atlas form is f A cos(V(t) + u - G), with V(t) = V(tc) + w (t - tc).
## Equal at all t:  A = A' / f(tc),  G = g' + V(tc) + u(tc),  i.e.  Z = Z' exp(-i (V + u)) / f.
## V, u, f at the single time tc from pyTMD (nodal_args.py, FES convention to match FES2022).
## S2, N2, K1 include K2, NU2, P1 (not separated by 95 days): converted the same way, then corrected (below).
tc2000 = hfit_data.tc - Dates.value(Second(DateTime(2000) - t_ref))      # tc in seconds since 2000-01-01
ftim = tempname() * "_times.nc"; fnod = tempname() * "_nodal.nc"         # small scratch files
NCDataset(ftim, "c"; format = :netcdf3_64bit_offset) do ds
    defDim(ds, "time", 1); defVar(ds, "t", [tc2000], ("time",))
end
run(`$PY $NODPY $ftim $fnod $(join(lowercase.(allcons), ",")) $NODAL`)
Vc, uc, fc = NCDataset(fnod) do nd
    vec(nd["G"][:, :]), vec(nd["pu"][:, :]), vec(nd["pf"][:, :])        # V [deg], u [deg], f [-], one per constituent in allcons
end
rm(ftim); rm(fnod)

# fit  : η(t) = A′ · cos( ω(t − tc) − g′ )
# atlas: η(t) = f · A · cos( V(t) + u − G )
# V grows at exactly ω, V(t) = V(tc) + ω(t − tc), so
#        η(t) = f · A · cos( ω(t − tc) + V(tc) + u − G )
# both forms equal at every t:  A′ = f · A  and  −g′ = V + u − G, so
#        A = A′ / f,   G = g′ + V + u
# complex, Z = A exp(−iG):
#        Z = (A′ / f) exp(−i (g′ + V + u)) = A′ exp(−i g′) · exp(−i (V + u)) / f = Z′ · exp(−i (V + u)) / f
ZR = similar(Zp)                                                         # ROMS, atlas convention [cm]
for k in 1:K
    ZR[:, :, k] .= Zp[:, :, k] .* cis(-deg2rad(Vc[k] + uc[k])) ./ fc[k]  # phase rotation (V + u), amplitude scaling (1 / f)
end
AerrR = hfit_data.aerr ./ reshape(fc[1:K], 1, 1, :)                      # amplitude error scales with 1/f; phase error unchanged

# inference: the fitted wave b (S2) also holds the unresolved wave a (K2), Z_a = R Z_b. Both in the record:
#        η(t) = Re{ f_b Z_b e^{i(V_b + u_b)} e^{i ω_b (t - tc)} + f_a Z_a e^{i(V_a + u_a)} e^{i ω_a (t - tc)} }
# the fit at ω_b sees wave a averaged over the record, the mean of e^{i Δω (t - tc)} = sin(x) / x, x = Δω T / 2, Δω = ω_a - ω_b:
#        Z′_b = f_b Z_b e^{i(V_b + u_b)} · B,   B = 1 + R (f_a / f_b) e^{i(ΔV + Δu)} sin(x) / x
# so the converted Z_b above is B times too large (amplitude |B|, phase -arg B): divide by B
Trec = hfit_data.records - 1                                             # record length [hours]
println(io, "inference correction (B: what the fitted wave holds relative to itself alone):")
for (ci, cb) in INFER
    ia, ib = ka(ci), ka(cb)
    x = deg2rad(TIDE_FREQ[ci] - TIDE_FREQ[cb]) * Trec / 2
    B = 1 + RINF[ci] * (fc[ia] / fc[ib]) * cis(deg2rad((Vc[ia] + uc[ia]) - (Vc[ib] + uc[ib]))) * sin(x) / x
    ZR[:, :, ib] ./= B; AerrR[:, :, ib] ./= abs(B)
    @printf(io, "  %-2s - %-3s: sin(x)/x %.3f, |B| %.4f, arg B %6.2f deg\n", cb, ci, sin(x) / x, abs(B), rad2deg(angle(B)))
end
AR = abs.(ZR); GR = mod.(-rad2deg.(angle.(ZR)), 360)                     # amplitude [cm], Greenwich phase [deg]

@printf(io, "nodal arguments at tc = %s (%s convention):\n", hfit_data.tcdate, NODAL)
for (k, c) in enumerate(hfit_data.cons)
    dA = abs.(ZR[:, :, k]) ./ abs.(ZF[:, :, k]); dG = -rad2deg.(angle.(ZR[:, :, k] .* conj.(ZF[:, :, k])))   # + = ROMS lags
    @printf(io, "  %-3s V %7.2f  u %6.2f  f %.4f   ROMS/FES: median A ratio %.3f, median dG %6.2f deg%s\n", c, Vc[k], uc[k], fc[k],
            median(filter(!isnan, dA)), median(filter(!isnan, dG)), c in last.(INFER) ? "  (corrected, FES ratio)" : "")
end

## check against the old fit (atlas inference + hourly nodal factors inside the fit), same points
if isfile(fold)
    ZO = NCDataset(fold) do ds
        oc = String.(ds["con"][:]); ii = 1:STRIDE:ds.dim["xi_rho"]; jj = 1:STRIDE:ds.dim["eta_rho"]
        cat([Float64.(ds["A"][ii, jj, findfirst(==(c), oc)]) .* cis.(-deg2rad.(Float64.(ds["G"][ii, jj, findfirst(==(c), oc)])))
             for c in hfit_data.cons]...; dims = 3)
    end
    println(io, "check against the old fit ($(basename(fold))):")
    for (k, c) in enumerate(hfit_data.cons)
        @printf(io, "  %-3s new/old A %.4f, new - old G %6.2f deg\n", c, median(abs.(ZR[:, :, k]) ./ abs.(ZO[:, :, k])),
                median(-rad2deg.(angle.(ZR[:, :, k] .* conj.(ZO[:, :, k])))))
    end
end

## --- section 4: comparison area, weights and the barotropic / internal-tide split
## pm, pn [1/m] from the grid file, thinned like the fit. The weight of a point is its cell area 1 / (pm pn),
## so area means do not depend on how the cells vary; points outside the comparison area have weight 0.
ii = 1:STRIDE:686; jj = 1:STRIDE:856
pm, pn = NCDataset(g -> (g["pm"][ii, jj], g["pn"][ii, jj]), grid_file)
dxkm = 1e-3 * mean(1 ./ pm); dykm = 1e-3 * mean(1 ./ pn)                 # km, cell size
nsx = round(Int, SPONGE / (STRIDE * dxkm)); nsy = round(Int, SPONGE / (STRIDE * dykm))   # sponge width [points]
M = falses(NX, NY); M[nsx+1:NX-nsx, nsy+1:NY-nsy] .= true; M .&= wetF    # comparison area
W = M ./ (pm .* pn)                                                      # cell-area weights [m^2], 0 outside M
shelf = hfit_data.h .< HSHELF

## separable Gaussian convolution (sigma s in grid cells, truncated at 4 sigma)
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

## 2-D Gaussian low-pass by normalised convolution with weights Wf (the filter sees only the comparison
## area; edges and masked points are renormalised): LP(Z) = G*(Wf Z) / G*(Wf);
## sigma = lc / 5.336 gives a response exp(-2 pi^2 sigma^2 / lambda^2) of 1/2 at lambda = lc
const Wf = Float64.(M)
const WfLP = Dict{Float64, Matrix{Float64}}()                            # G*(Wf) for each cutoff
function gfilt(Z, lc)
    s = lc / 5.336 / (STRIDE * dxkm)                                     # sigma in grid points
    den = get!(() -> conv2(Wf, s), WfLP, lc)
    return conv2(ifelse.(M, Z, zero(eltype(Z))), s) ./ den              # Wf Z, without NaN x 0 at masked points
end
lcof(c) = c in DIURNAL ? LC_D : LC_SD                                    # cutoff of a constituent [km]
lcp(c)  = c in DIURNAL ? LCP_D : LCP_SD                                  # same, for the co-phase lines

## barotropic tide: FES + LP(ROMS - FES); the difference is filtered, not ROMS alone, because the tide's sharp
## shelf-break gradients are in both and cancel. What is left in ZR - ZB is the coherent internal-tide SSH.
ZB = similar(ZR)                                                         # ROMS barotropic [cm]
for (k, c) in enumerate(hfit_data.cons)
    D = ifelse.(M, ZR[:, :, k] .- ZF[:, :, k], 0.0im)                    # outside M: no data for the filter
    ZB[:, :, k] .= ZF[:, :, k] .+ gfilt(D, lcof(c))
end
ZIT = ZR .- ZB                                                           # coherent internal-tide SSH [cm]

## --- section 5: statistics over the comparison area
wmean(X, w = W) = sum(w[w .> 0] .* X[w .> 0]) / sum(w)                   # area mean over the points with weight
rmsd(Z1, Z2, w = W) = sqrt(wmean(abs2.(Z1 .- Z2) ./ 2, w))               # rms difference, RMS = sqrt(< |dZ|^2 / 2 >)
srms(Z, w = W) = sqrt(wmean(abs2.(Z) ./ 2, w))                           # signal rms
aratio(Z1, Z2, w = W) = wmean(abs.(Z1), w) / wmean(abs.(Z2), w)          # <A1> / <A2>
dphase(Z1, Z2, w = W) = -rad2deg(angle(wmean(Z1 .* conj.(Z2), w)))       # + = Z1 lags Z2

@printf(io, "\ncomparison area: %d of %d points, %.0f km sponge removed (%d x, %d y points), dx %.3f km, dy %.3f km, shelf %d points\n",
        count(M), NX * NY, SPONGE, nsx, nsy, dxkm, dykm, count(M .& shelf))

## ROMS-only barotropic for the co-phase lines: low-pass over deep water only (h >= HDEEP), so neither FES
## nor the shelf (different, fast-changing phase) enters; only meaningful where h >= HDEEP
const Md = M .& (hfit_data.h .>= HDEEP)
const Wdeep = Float64.(Md)
const WdLP = Dict{Float64, Matrix{Float64}}()
function zplot(c)
    s = lcp(c) / 5.336 / (STRIDE * dxkm)
    den = get!(() -> conv2(Wdeep, s), WdLP, lcp(c))
    return conv2(Wdeep .* ZR[:, :, kc(c)], s) ./ den
end

rows = []
for (k, c) in enumerate(hfit_data.cons)
    lc = lcof(c); zr, zb, zf = ZR[:, :, k], ZB[:, :, k], ZF[:, :, k]
    wsh = W .* shelf; wdp = W .* .!shelf
    push!(rows, Dict{Symbol, Any}(:con => c, :inf => c in last.(INFER), :A => wmean(abs.(zf)), :S => srms(zf),
        :rms => rmsd(zb, zf), :ar => aratio(zb, zf), :dg => dphase(zb, zf),
        :rms_sh => rmsd(zb, zf, wsh), :S_sh => srms(zf, wsh), :rms_dp => rmsd(zb, zf, wdp), :S_dp => srms(zf, wdp),
        :rms_full => rmsd(zr, zf),                                       # ROMS unfiltered
        :rms_lpR => rmsd(gfilt(zr, lc), zf),                             # ROMS alone low-passed
        :Tlp => rmsd(gfilt(zf, lc), zf),                                 # what the low-pass does to FES
        :it => srms(ZIT[:, :, k]),                                       # coherent IT SSH rms
        :aerr => median(AerrR[:, :, k][M])))
end

@printf(io, "\nROMS Amazon 900 m (%s, hourly zeta) vs FES2022b (1/30 deg); comparison area: %.0f km sponge removed, FES-wet points (%d)\n", hfit_data.title[end-23:end], SPONGE, count(M))
@printf(io, "ROMS_bt = FES + LP(ROMS - FES), LP Gaussian half power at %.0f km (semidiurnal) / %.0f km (diurnal)\n", LC_SD, LC_D)
@printf(io, "RMS = sqrt(<|dZ|^2/2>) [cm]; S = FES signal rms; ratio = <A_ROMS>/<A_FES>; dG = ROMS - FES [deg], + = ROMS lags\n")
@printf(io, "shelf: h < %.0f m, deep: h >= %.0f m; RMSfull: ROMS unfiltered; RMS_lpR: ROMS alone low-passed; Tlp-raw: rms change of FES by the low-pass\n", HSHELF, HSHELF)
@printf(io, "IT: rms of coherent internal-tide SSH (ROMS - ROMS_bt); A_err: median white-noise error of the ROMS amplitude; * corrected with the FES ratios (K2, NU2, P1)\n")
@printf(io, "%-5s %7s %7s | %7s %6s %6s %7s | %7s %6s %7s %6s | %7s %7s %7s | %6s %6s\n", "con", "<A>", "S", "RMS", "rel", "ratio", "dG",
        "RMSshlf", "rel", "RMSdeep", "rel", "RMSfull", "RMS_lpR", "Tlp-raw", "IT", "A_err")
for r in rows
    @printf(io, "%-5s %7.3f %7.3f | %7.3f %5.1f%% %6.3f %+7.2f | %7.3f %5.1f%% %7.3f %5.1f%% | %7.3f %7.3f %7.3f | %6.2f %6.3f\n",
            r[:con] * (r[:inf] ? "*" : ""), r[:A], r[:S], r[:rms], 100 * r[:rms] / r[:S], r[:ar], r[:dg],
            r[:rms_sh], 100 * r[:rms_sh] / r[:S_sh], r[:rms_dp], 100 * r[:rms_dp] / r[:S_dp], r[:rms_full], r[:rms_lpR], r[:Tlp], r[:it], r[:aerr])
end
rss(key) = sqrt(sum(r[key]^2 for r in rows))
@printf(io, "\nRSS over %s: ROMS_bt %.3f (shelf %.3f, deep %.3f), ROMS unfiltered %.3f, ROMS alone low-passed %.3f cm\n",
        join(hfit_data.cons, " "), rss(:rms), rss(:rms_sh), rss(:rms_dp), rss(:rms_full), rss(:rms_lpR))
@printf(io, "median residual std %.2f cm, median SSH std %.2f cm (comparison area)\n", median(hfit_data.resid[M]), median(hfit_data.sshstd[M]))

@printf(io, "\nRMS [cm] of LP(ROMS - FES) vs half-power cutoff [km] (the internal tide drops out as the cutoff grows):\n")
for c in ("M2", "S2", "K1", "O1")
    k = kc(c); k === nothing && continue
    D = ZR[:, :, k] .- ZF[:, :, k]
    @printf(io, "  %-3s %s\n", c, join([@sprintf("%5.0f: %.3f", L, srms(gfilt(D, L))) for L in (200, 300, 400, 600, 800)], "  "))
end

## deep-water phase gradient, plane fit G = a + b x + c y over h >= HDEEP: magnitude [deg / 100 km] and
## direction [deg, ccw from xi]; checks whether the plot filter turns the co-phase lines
xk = STRIDE * dxkm .* repeat(1:NX, 1, NY); yk = STRIDE * dykm .* repeat((1:NY)', NX, 1)
function pgrad(Z)
    zm = sum(Z[Md]); G = -rad2deg.(angle.(Z[Md] .* conj(zm)))
    b = hcat(ones(count(Md)), xk[Md], yk[Md]) \ G
    return 100hypot(b[2], b[3]), rad2deg(atan(b[3], b[2]))
end
@printf(io, "\nDeep-water (h >= %.0f m) phase gradient, plane fit: deg/100 km @ direction [deg ccw from xi]\n", HDEEP)
for c in ("M2", "S2", "K1", "O1")
    k = kc(c); k === nothing && continue
    @printf(io, "  %-3s FES2022: %.2f @ %6.1f   ROMS raw: %.2f @ %6.1f   ROMS plot-LP (%.0f km): %.2f @ %6.1f\n", c,
            pgrad(ZF[:, :, k])..., pgrad(ZR[:, :, k])..., lcp(c), pgrad(zplot(c))...)
end

tab = String(take!(io)); print(tab); write(ftab, tab)
println("wrote $ftab")

## --- section 6: save for the plotting code
## NetCDF has no complex type: each Z = A exp(-iG) is stored as amplitude [cm] and phase [deg, 0-360].
ampof(Z) = Float32.(abs.(Z)); phaof(Z) = Float32.(mod.(-rad2deg.(angle.(Z)), 360))
ZCP = cat([ifelse.(Md, zplot(c), complex(NaN, NaN)) for c in hfit_data.cons]...; dims = 3)   # deep-water co-phase field
isfile(fcomp) && rm(fcomp)
NCDataset(fcomp, "c") do ds
    defDim(ds, "xi_rho", NX); defDim(ds, "eta_rho", NY); defDim(ds, "con", K)
    ds.attrib["title"] = "ROMS SSH tide vs FES2022b, for plotting: " * hfit_data.title
    ds.attrib["convention"] = "Z = A exp(-iG): Greenwich phase lag G, nodal-free A; ROMS corrected for K2, NU2, P1 with FES ratios"
    ds.attrib["split"] = @sprintf("ZB = ZF + LP(ZR - ZF), ZIT = ZR - ZB; LP half power at %.0f km (semidiurnal) / %.0f km (diurnal)", LC_SD, LC_D)
    ds.attrib["tc"] = string(hfit_data.tcdate); ds.attrib["stride"] = STRIDE; ds.attrib["sponge_km"] = SPONGE
    ds.attrib["hshelf"] = HSHELF; ds.attrib["hdeep"] = HDEEP
    ds.attrib["script"] = "internalwave_julia/model_validation/ssh_tidalcomp_fes.jl"
    defVar(ds, "con", hfit_data.cons, ("con",))
    defVar(ds, "inferred", Int8[c in last.(INFER) for c in hfit_data.cons], ("con",); attrib = ["long_name" => "corrected with a FES ratio"])
    defVar(ds, "lon_rho", hfit_data.lon, ("xi_rho", "eta_rho")); defVar(ds, "lat_rho", hfit_data.lat, ("xi_rho", "eta_rho"))
    defVar(ds, "h", hfit_data.h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
    defVar(ds, "M", Int8.(M), ("xi_rho", "eta_rho"); attrib = ["long_name" => "comparison area (1 = in)"])
    for (nm, v, u, ln) in (("resid_std", hfit_data.resid, "cm", "residual std of the fit"), ("ssh_std", hfit_data.sshstd, "cm", "SSH std"))
        defVar(ds, nm, Float32.(v), ("xi_rho", "eta_rho"); attrib = ["units" => u, "long_name" => ln], deflatelevel = 3)
    end
    for (nm, Z, ln) in (("R", ZR, "ROMS tide"), ("F", ZF, "FES2022b"), ("B", ZB, "ROMS barotropic"), ("IT", ZIT, "coherent internal-tide SSH"),
                        ("CP", ZCP, "ROMS low-passed over deep water, for co-phase lines"))
        defVar(ds, "A_$nm", ampof(Z), ("xi_rho", "eta_rho", "con"); attrib = ["units" => "cm", "long_name" => "amplitude: $ln"], deflatelevel = 3)
        defVar(ds, "G_$nm", phaof(Z), ("xi_rho", "eta_rho", "con"); attrib = ["units" => "deg", "long_name" => "phase: $ln"], deflatelevel = 3)
    end
    for (nm, key, u, ln) in (("A_mean", :A, "cm", "area mean FES amplitude"), ("S", :S, "cm", "FES signal rms"), ("rms", :rms, "cm", "ROMS_bt - FES rms"),
                             ("ratio", :ar, "1", "amplitude ratio ROMS_bt / FES"), ("dG", :dg, "deg", "phase ROMS_bt - FES, + = ROMS lags"),
                             ("rms_shelf", :rms_sh, "cm", "rms, shelf"), ("rms_deep", :rms_dp, "cm", "rms, deep"),
                             ("rms_full", :rms_full, "cm", "ROMS unfiltered - FES rms"), ("it_rms", :it, "cm", "internal-tide SSH rms"))
        defVar(ds, nm, Float64[r[key] for r in rows], ("con",); attrib = ["units" => u, "long_name" => ln])
    end
end
println("wrote $fcomp")
