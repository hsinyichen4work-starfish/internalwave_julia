#= ssh_tidalcomp_fes_models.jl
Hsin-Yi Chen, USM DMS
Compare the SSH harmonic constants of several models (ssh_harmonic_models.jl) with FES2022b. Same method as
ssh_tidalcomp_fes.jl (ROMS 900 m), but every model is first put on ONE target grid, the 2700 m ROMS rho grid
(new_design_fix/roms_grd_2700m_1104.nc), so the comparison area, cell weights, sponge, bathymetry (shelf / deep
split) and low-pass filter are the same for all models.
Settings, machine and sources: tidal_models_setup.jl (TC_HOST, TC_SRC, TC_STRIDE).

Section 1: read the fit <SRC>_sshfit.nc (looked for in P.fitdirs, so expanse also finds fits made on aurelia),
  Z' = A' exp(-i g') [cm], and interpolate Z', A' error, residual std, SSH std bilinearly onto the target grid
  (cell search works on regular and curvilinear grids); a target point is wet only when its 4 model corners
  are wet and the target mask_rho is 1.
Section 2: FES2022b at the target points (as ssh_tidalcomp_fes.jl) and the FES ratios for the inference.
Section 3: atlas convention A = A' / f(tc), G = g' + V(tc) + u(tc) (pyTMD, FES convention); K2, NU2, P1 inferred
  with the FES ratios (record length: span_hours of the fit, any sampling interval).
Section 4: comparison area = target interior (SPONGE km removed along the 3 open boundaries W, N, E and the
  closed south edge) & FES wet & model wet; weights 1 / (pm pn); ZB = ZF + LP(ZR - ZF), ZIT = ZR - ZB.
Section 5: statistics (same table as ssh_tidalcomp_fes.jl) -> P.out/<SRC>_FES_sshfit_rms[_s<n>].txt
Section 6: P.out/<SRC>_FES_sshfit_comp[_s<n>].nc for tidal_comp_models_plot.jl

Usage:
  expanse:  TC_SRC=ROMS_HYCOMparent,ROMS_GLORYSparent,HYCOM,NCOM2700 julia --project=. -t 4 model_validation/tidal/ssh_tidalcomp_fes_models.jl
            (HYCOM / NCOM fits made on aurelia are read over ~/aurelia; only small files)
  aurelia:  TC_SRC=HYCOM,NCOM2700 julia --project=. -t 8 model_validation/tidal/ssh_tidalcomp_fes_models.jl   (needs pyTMD, see TC_PY)
=#

using NCDatasets, Printf, Statistics, Dates, LinearAlgebra
include(joinpath(@__DIR__, "tidal_models_setup.jl"))
include(P.func * "harm_fit.jl")                          # TIDE_FREQ

## --- settings (as ssh_tidalcomp_fes.jl, sponge and cutoffs for the 2700 m domain)
const NODPY   = P.func * "nodal_args.py"                 # V, u, f from pyTMD
const NODAL   = "FES"
const INFER   = [("K2", "S2"), ("NU2", "N2"), ("P1", "K1")]   # (absorbed, fitted)
const SPONGE  = parse(Float64, get(ENV, "TC_SPONGE", "100"))   # km removed along each edge (ROMS 2700 m sponge: 37 points = 100 km)
const HSHELF  = 200.0                                    # m, shelf / deep split
const LC_SD   = 400.0                                    # km, half-power wavelength of the low-pass, semidiurnal
const LC_D    = 800.0                                    # km, diurnal
const DIURNAL = ["K1", "O1"]
const LCP_SD  = 300.0                                    # km, deep-water co-phase low-pass, semidiurnal
const LCP_D   = 600.0                                    # km, diurnal
const HDEEP   = 2000.0                                   # m, deep water for the co-phase lines
isfile(P.py) || error("no python at $(P.py): it needs pyTMD (set TC_PY, see tidal_models_setup.jl / the usage notes)")
println("host $HOST, sources $(join(SRC_RUN, ", ")), stride $STRIDE, sponge $SPONGE km")

## --- target grid (Julia order: xi, eta)
const TG = NCDataset(P.grid) do g
    nx, ny = size(g["lon_rho"]); ii = 1:STRIDE:nx; jj = 1:STRIDE:ny
    (lon = Float64.(g["lon_rho"][ii, jj]), lat = Float64.(g["lat_rho"][ii, jj]), h = Float64.(g["h"][ii, jj]),
     mask = g["mask_rho"][ii, jj] .== 1, pm = Float64.(g["pm"][ii, jj]), pn = Float64.(g["pn"][ii, jj]))
end
const NX, NY = size(TG.lon)

## --- bilinear cell search on any (regular or curvilinear) source grid
function invbilin(X, Y, x, y)                            # corners 00, 10, 01, 11 -> (a, b), may be outside [0, 1]
    a, b = 0.5, 0.5
    for _ in 1:30
        fx = (1 - a) * (1 - b) * X[1] + a * (1 - b) * X[2] + (1 - a) * b * X[3] + a * b * X[4] - x
        fy = (1 - a) * (1 - b) * Y[1] + a * (1 - b) * Y[2] + (1 - a) * b * Y[3] + a * b * Y[4] - y
        xa = (1 - b) * (X[2] - X[1]) + b * (X[4] - X[3]); xb = (1 - a) * (X[3] - X[1]) + a * (X[4] - X[2])
        ya = (1 - b) * (Y[2] - Y[1]) + b * (Y[4] - Y[3]); yb = (1 - a) * (Y[3] - Y[1]) + a * (Y[4] - Y[2])
        det = xa * yb - xb * ya; abs(det) < 1e-20 && return NaN, NaN
        da = (yb * fx - xb * fy) / det; db = (-ya * fx + xa * fy) / det
        a -= da; b -= db
        abs(da) + abs(db) < 1e-12 && break
    end
    return a, b
end
# for each target point: lower-left source corner (i, j) and weights a, b (0 = outside the source grid).
# Walk from the previous point's cell (the target points are visited in grid order); global search if it fails.
function locator(slon, slat, tlon, tlat)
    nx, ny = size(slon)
    I0 = zeros(Int, size(tlon)); J0 = zeros(Int, size(tlon)); A = zeros(size(tlon)); Bw = zeros(size(tlon))
    ci, cj = 1, 1
    function trycell(i, j, x, y, c)
        X = c .* (slon[i, j], slon[i+1, j], slon[i, j+1], slon[i+1, j+1]); Y = (slat[i, j], slat[i+1, j], slat[i, j+1], slat[i+1, j+1])
        return invbilin(X, Y, c * x, y)
    end
    for p in CartesianIndices(tlon)
        x, y = tlon[p], tlat[p]; c = cosd(y); found = false
        for _ in 1:200                                    # walk
            a, b = trycell(ci, cj, x, y, c)
            isfinite(a) || break
            if -1e-9 <= a <= 1 + 1e-9 && -1e-9 <= b <= 1 + 1e-9
                I0[p], J0[p], A[p], Bw[p] = ci, cj, clamp(a, 0, 1), clamp(b, 0, 1); found = true; break
            end
            ni = clamp(ci + floor(Int, a), 1, nx - 1); nj = clamp(cj + floor(Int, b), 1, ny - 1)
            (ni, nj) == (ci, cj) && break                 # stuck at the edge: outside
            ci, cj = ni, nj
        end
        if !found                                         # global nearest node, then its 4 cells
            d = ((slon .- x) .* c) .^ 2 .+ (slat .- y) .^ 2; q = argmin(d)
            for i in (q[1] - 1, q[1]), j in (q[2] - 1, q[2])
                (1 <= i < nx && 1 <= j < ny) || continue
                a, b = trycell(i, j, x, y, c)
                if -1e-9 <= a <= 1 + 1e-9 && -1e-9 <= b <= 1 + 1e-9
                    I0[p], J0[p], A[p], Bw[p] = i, j, clamp(a, 0, 1), clamp(b, 0, 1); ci, cj = i, j; break
                end
            end
        end
    end
    return (; I0, J0, A, B = Bw)
end
# bilinear value; NaN when the point is outside or any corner is NaN
function bilin(L, F)
    out = fill(convert(eltype(F), NaN), size(L.I0))
    for p in CartesianIndices(L.I0)
        i, j = L.I0[p], L.J0[p]; i == 0 && continue
        a, b = L.A[p], L.B[p]
        out[p] = (1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i+1, j] + (1 - a) * b * F[i, j+1] + a * b * F[i+1, j+1]
    end
    return out
end

## --- FES2022b (same as ssh_tidalcomp_fes.jl)
function fes_read(dir, cons, lon, lat)
    fn(c) = dir * "$(lowercase(c))_fes2022.nc"
    for c in cons; isfile(fn(c)) || error("no FES file for $c: $(fn(c))"); end
    lonf, latf = NCDataset(ds -> (ds["lon"][:], ds["lat"][:]), fn(cons[1]))
    ia = searchsortedlast(lonf, minimum(lon)) - 1; ib = searchsortedfirst(lonf, maximum(lon)) + 1
    ja = searchsortedlast(latf, minimum(lat)) - 1; jb = searchsortedfirst(latf, maximum(lat)) + 1
    dl = (lonf[ib] - lonf[ia]) / (ib - ia); dt = (latf[jb] - latf[ja]) / (jb - ja)
    fx = (lon .- lonf[ia]) ./ dl .+ 1; fy = (lat .- latf[ja]) ./ dt .+ 1
    i0 = floor.(Int, fx); j0 = floor.(Int, fy); wx = fx .- i0; wy = fy .- j0
    bil(z, p) = (1 - wx[p]) * (1 - wy[p]) * z[i0[p], j0[p]] + wx[p] * (1 - wy[p]) * z[i0[p] + 1, j0[p]] +
                (1 - wx[p]) * wy[p] * z[i0[p], j0[p] + 1] + wx[p] * wy[p] * z[i0[p] + 1, j0[p] + 1]
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

## FES at the target points, once for all models (FES lon 0..360; the 2700 m grid lon is 0..360 too)
const ALLCONS = vcat(CONS, first.(INFER))
const ZFALL, WETF = fes_read(P.fes, ALLCONS, mod.(TG.lon, 360), TG.lat)
const K = length(CONS)
ka(c) = findfirst(==(c), ALLCONS)
kc(c) = findfirst(==(c), CONS)

## --- separable Gaussian low-pass (as ssh_tidalcomp_fes.jl)
function conv1!(out, y, k, hw)
    n = length(y)
    @inbounds for i in 1:n
        acc = zero(eltype(y))
        for j in max(1, i - hw):min(n, i + hw); acc += k[j - i + hw + 1] * y[j]; end
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
lcof(c) = c in DIURNAL ? LC_D : LC_SD
lcp(c)  = c in DIURNAL ? LCP_D : LCP_SD

## --- one source
function run_comp(sname)
    t_start = time(); io = IOBuffer()
    fname = findfile(fitfile(sname))
    fname === nothing && error("no fit $(fitfile(sname)) in $(join(P.fitdirs, ", ")) (run ssh_harmonic_models.jl first)")
    println("===== $sname: $fname")

    ## section 1: read the fit, interpolate to the target grid
    hd = NCDataset(fname) do ds
        String.(ds["con"][:]) == CONS || error("$fname: constituents $(ds["con"][:]) differ from $CONS")
        (lon = Float64.(ds["lon"][:, :]), lat = Float64.(ds["lat"][:, :]),
         amp = Float64.(ds["amp"][:, :, :]), pha = Float64.(ds["pha"][:, :, :]), aerr = Float64.(ds["amp_err"][:, :, :]),
         resid = Float64.(ds["resid_std"][:, :]), sshstd = Float64.(ds["ssh_std"][:, :]),
         tc = Float64(ds.attrib["tc_seconds_since_1994"]), tcdate = DateTime(ds.attrib["tc"]),
         span = Float64(ds.attrib["span_hours"]), dth = Float64(ds.attrib["dt_hours"]), records = Int(ds.attrib["records"]),
         title = ds.attrib["title"], label = ds.attrib["label"])
    end
    # model and target on the same lon convention (-180..180); the ROMS 2700 m runs ARE the target grid
    same = size(hd.lon) == (NX, NY) && maximum(abs.(hd.lon .- lon180.(TG.lon))) < 1e-6
    Zn = hd.amp .* cis.(-deg2rad.(hd.pha))                                  # Z' native [cm]
    if same
        Zp = Zn; aerr = hd.aerr; resid = hd.resid; sshstd = hd.sshstd
    else
        L = locator(hd.lon, hd.lat, lon180.(TG.lon), TG.lat)
        Zp = cat([bilin(L, Zn[:, :, k]) for k in 1:K]...; dims = 3)
        aerr = cat([bilin(L, hd.aerr[:, :, k]) for k in 1:K]...; dims = 3)
        resid = bilin(L, hd.resid); sshstd = bilin(L, hd.sshstd)
    end
    wetM = TG.mask .& vec_all_finite(Zp)
    println(io, hd.title)
    @printf(io, "fit file %s: native %d x %d, %d records (dt %.2f h, span %.1f d), tc = %s\n", fname, size(hd.lon)..., hd.records, hd.dth, hd.span / 24, hd.tcdate)
    @printf(io, "  on the 2700 m target grid %d x %d%s: %d wet points (target ocean %d)\n", NX, NY,
            same ? " (same grid, no interpolation)" : " (bilinear)", count(wetM), count(TG.mask))

    ## section 2: FES
    ZF = ZFALL[:, :, 1:K]
    RINF = Dict(ci => sum(ZFALL[:, :, ka(ci)][WETF] .* conj.(ZFALL[:, :, ka(cb)][WETF])) / sum(abs2.(ZFALL[:, :, ka(cb)][WETF])) for (ci, cb) in INFER)

    ## section 3: atlas convention, nodal corrections at tc
    tc2000 = hd.tc - Dates.value(Second(DateTime(2000) - T_REF))
    ftim = tempname() * "_times.nc"; fnod = tempname() * "_nodal.nc"
    NCDataset(ftim, "c"; format = :netcdf3_64bit_offset) do ds
        defDim(ds, "time", 1); defVar(ds, "t", [tc2000], ("time",))
    end
    run(`$(P.py) $NODPY $ftim $fnod $(join(lowercase.(ALLCONS), ",")) $NODAL`)
    Vc, uc, fc = NCDataset(nd -> (vec(nd["G"][:, :]), vec(nd["pu"][:, :]), vec(nd["pf"][:, :])), fnod)
    rm(ftim); rm(fnod)
    ZR = similar(Zp)
    for k in 1:K; ZR[:, :, k] .= Zp[:, :, k] .* cis(-deg2rad(Vc[k] + uc[k])) ./ fc[k]; end
    AerrR = aerr ./ reshape(fc[1:K], 1, 1, :)
    println(io, "inference correction (B: what the fitted wave holds relative to itself alone):")
    for (ci, cb) in INFER
        ia, ib = ka(ci), ka(cb)
        x = deg2rad(TIDE_FREQ[ci] - TIDE_FREQ[cb]) * hd.span / 2
        B = 1 + RINF[ci] * (fc[ia] / fc[ib]) * cis(deg2rad((Vc[ia] + uc[ia]) - (Vc[ib] + uc[ib]))) * sin(x) / x
        ZR[:, :, kc(cb)] ./= B; AerrR[:, :, kc(cb)] ./= abs(B)
        @printf(io, "  %-2s - %-3s: |R| %.4f, sin(x)/x %.3f, |B| %.4f, arg B %6.2f deg\n", cb, ci, abs(RINF[ci]), sin(x) / x, abs(B), rad2deg(angle(B)))
    end

    ## section 4: comparison area, weights, barotropic / internal-tide split
    dxkm = 1e-3 * mean(1 ./ TG.pm); dykm = 1e-3 * mean(1 ./ TG.pn)
    nsx = round(Int, SPONGE / (STRIDE * dxkm)); nsy = round(Int, SPONGE / (STRIDE * dykm))
    M = falses(NX, NY); M[nsx+1:NX-nsx, nsy+1:NY-nsy] .= true; M .&= WETF .& wetM
    W = M ./ (TG.pm .* TG.pn); shelf = TG.h .< HSHELF
    Wf = Float64.(M); WfLP = Dict{Float64, Matrix{Float64}}()
    function gfilt(Z, lc)
        s = lc / 5.336 / (STRIDE * dxkm)
        den = get!(() -> conv2(Wf, s), WfLP, lc)
        return conv2(ifelse.(M, Z, zero(eltype(Z))), s) ./ den
    end
    ZB = similar(ZR)
    for (k, c) in enumerate(CONS)
        ZB[:, :, k] .= ZF[:, :, k] .+ gfilt(ifelse.(M, ZR[:, :, k] .- ZF[:, :, k], 0.0im), lcof(c))
    end
    ZIT = ZR .- ZB
    Md = M .& (TG.h .>= HDEEP); Wdeep = Float64.(Md); WdLP = Dict{Float64, Matrix{Float64}}()
    function zplot(c)
        s = lcp(c) / 5.336 / (STRIDE * dxkm)
        den = get!(() -> conv2(Wdeep, s), WdLP, lcp(c))
        return conv2(Wdeep .* ifelse.(Md, ZR[:, :, kc(c)], 0.0im), s) ./ den
    end

    ## section 5: statistics
    wmean(X, w = W) = sum(w[w .> 0] .* X[w .> 0]) / sum(w)
    rmsd(Z1, Z2, w = W) = sqrt(wmean(abs2.(Z1 .- Z2) ./ 2, w))
    srms(Z, w = W) = sqrt(wmean(abs2.(Z) ./ 2, w))
    aratio(Z1, Z2, w = W) = wmean(abs.(Z1), w) / wmean(abs.(Z2), w)
    dphase(Z1, Z2, w = W) = -rad2deg(angle(wmean(Z1 .* conj.(Z2), w)))
    @printf(io, "\ncomparison area: %d of %d points, %.0f km removed along the edges (%d x, %d y points), dx %.2f km, dy %.2f km, shelf %d points\n",
            count(M), NX * NY, SPONGE, nsx, nsy, dxkm, dykm, count(M .& shelf))
    rows = map(enumerate(CONS)) do (k, c)
        lc = lcof(c); zr, zb, zf = ZR[:, :, k], ZB[:, :, k], ZF[:, :, k]
        wsh = W .* shelf; wdp = W .* .!shelf
        Dict{Symbol, Any}(:con => c, :inf => c in last.(INFER), :A => wmean(abs.(zf)), :S => srms(zf),
            :rms => rmsd(zb, zf), :ar => aratio(zb, zf), :dg => dphase(zb, zf),
            :rms_sh => rmsd(zb, zf, wsh), :S_sh => srms(zf, wsh), :rms_dp => rmsd(zb, zf, wdp), :S_dp => srms(zf, wdp),
            :rms_full => rmsd(zr, zf), :rms_lpR => rmsd(gfilt(zr, lc), zf), :Tlp => rmsd(gfilt(zf, lc), zf),
            :it => srms(ZIT[:, :, k]), :aerr => median(filter(isfinite, AerrR[:, :, k][M])))
    end
    @printf(io, "\n%s vs FES2022b (1/30 deg), on the ROMS 2700 m grid; comparison area: %.0f km edge removed, FES & model wet (%d points)\n", hd.label, SPONGE, count(M))
    @printf(io, "bt = FES + LP(model - FES), LP Gaussian half power at %.0f km (semidiurnal) / %.0f km (diurnal)\n", LC_SD, LC_D)
    @printf(io, "RMS = sqrt(<|dZ|^2/2>) [cm]; S = FES signal rms; ratio = <A_bt>/<A_FES>; dG = model - FES [deg], + = model lags\n")
    @printf(io, "shelf: h < %.0f m (2700 m ROMS bathymetry for every model); RMSfull: model unfiltered; RMS_lpR: model alone low-passed\n", HSHELF)
    @printf(io, "%-5s %7s %7s | %7s %6s %6s %7s | %7s %6s %7s %6s | %7s %7s %7s | %6s %6s\n", "con", "<A>", "S", "RMS", "rel", "ratio", "dG",
            "RMSshlf", "rel", "RMSdeep", "rel", "RMSfull", "RMS_lpR", "Tlp-raw", "IT", "A_err")
    for r in rows
        @printf(io, "%-5s %7.3f %7.3f | %7.3f %5.1f%% %6.3f %+7.2f | %7.3f %5.1f%% %7.3f %5.1f%% | %7.3f %7.3f %7.3f | %6.2f %6.3f\n",
                r[:con] * (r[:inf] ? "*" : ""), r[:A], r[:S], r[:rms], 100 * r[:rms] / r[:S], r[:ar], r[:dg],
                r[:rms_sh], 100 * r[:rms_sh] / r[:S_sh], r[:rms_dp], 100 * r[:rms_dp] / r[:S_dp], r[:rms_full], r[:rms_lpR], r[:Tlp], r[:it], r[:aerr])
    end
    rss(key) = sqrt(sum(r[key]^2 for r in rows))
    @printf(io, "\nRSS over %s: bt %.3f (shelf %.3f, deep %.3f), unfiltered %.3f, model alone low-passed %.3f cm\n",
            join(CONS, " "), rss(:rms), rss(:rms_sh), rss(:rms_dp), rss(:rms_full), rss(:rms_lpR))
    @printf(io, "median residual std %.2f cm, median SSH std %.2f cm (comparison area)\n", median(resid[M]), median(sshstd[M]))
    tab = String(take!(io)); print(tab)
    mkpath(P.out); ftab = P.out * "$(sname)_FES_sshfit_rms$(SFX).txt"; write(ftab, tab); println("wrote $ftab")

    ## section 6: save for plotting
    ampof(Z) = Float32.(abs.(Z)); phaof(Z) = Float32.(mod.(-rad2deg.(angle.(Z)), 360))
    ZCP = cat([ifelse.(Md, zplot(c), complex(NaN, NaN)) for c in CONS]...; dims = 3)
    fcomp = P.out * compfile(sname); isfile(fcomp) && rm(fcomp)
    NCDataset(fcomp, "c") do ds
        defDim(ds, "xi_rho", NX); defDim(ds, "eta_rho", NY); defDim(ds, "con", K)
        ds.attrib["title"] = "SSH tide vs FES2022b on the ROMS 2700 m grid, for plotting: " * hd.title
        ds.attrib["source"] = sname; ds.attrib["label"] = hd.label
        ds.attrib["convention"] = "Z = A exp(-iG): Greenwich phase lag G, nodal-free A; corrected for K2, NU2, P1 with FES ratios"
        ds.attrib["split"] = @sprintf("ZB = ZF + LP(ZR - ZF), ZIT = ZR - ZB; LP half power at %.0f km (semidiurnal) / %.0f km (diurnal)", LC_SD, LC_D)
        ds.attrib["tc"] = string(hd.tcdate); ds.attrib["stride"] = STRIDE; ds.attrib["sponge_km"] = SPONGE
        ds.attrib["hshelf"] = HSHELF; ds.attrib["hdeep"] = HDEEP; ds.attrib["grid"] = P.grid
        ds.attrib["script"] = "internalwave_julia/model_validation/tidal/ssh_tidalcomp_fes_models.jl"
        defVar(ds, "con", CONS, ("con",))
        defVar(ds, "inferred", Int8[c in last.(INFER) for c in CONS], ("con",); attrib = ["long_name" => "corrected with a FES ratio"])
        defVar(ds, "lon_rho", TG.lon, ("xi_rho", "eta_rho")); defVar(ds, "lat_rho", TG.lat, ("xi_rho", "eta_rho"))
        defVar(ds, "h", TG.h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
        defVar(ds, "area", 1 ./ (TG.pm .* TG.pn), ("xi_rho", "eta_rho"); attrib = ["units" => "m2", "long_name" => "cell area"])
        defVar(ds, "M", Int8.(M), ("xi_rho", "eta_rho"); attrib = ["long_name" => "comparison area (1 = in)"])
        for (nm, v, ln) in (("resid_std", resid, "residual std of the fit"), ("ssh_std", sshstd, "SSH std"))
            defVar(ds, nm, Float32.(v), ("xi_rho", "eta_rho"); attrib = ["units" => "cm", "long_name" => ln], deflatelevel = 3)
        end
        for (nm, Z, ln) in (("R", ZR, "model tide"), ("F", ZF, "FES2022b"), ("B", ZB, "model barotropic"), ("IT", ZIT, "coherent internal-tide SSH"),
                            ("CP", ZCP, "model low-passed over deep water, for co-phase lines"))
            defVar(ds, "A_$nm", ampof(Z), ("xi_rho", "eta_rho", "con"); attrib = ["units" => "cm", "long_name" => "amplitude: $ln"], deflatelevel = 3)
            defVar(ds, "G_$nm", phaof(Z), ("xi_rho", "eta_rho", "con"); attrib = ["units" => "deg", "long_name" => "phase: $ln"], deflatelevel = 3)
        end
        for (nm, key, u, ln) in (("A_mean", :A, "cm", "area mean FES amplitude"), ("S", :S, "cm", "FES signal rms"), ("rms", :rms, "cm", "bt - FES rms"),
                                 ("ratio", :ar, "1", "amplitude ratio bt / FES"), ("dG", :dg, "deg", "phase bt - FES, + = model lags"),
                                 ("rms_shelf", :rms_sh, "cm", "rms, shelf"), ("rms_deep", :rms_dp, "cm", "rms, deep"),
                                 ("rms_full", :rms_full, "cm", "unfiltered - FES rms"), ("it_rms", :it, "cm", "internal-tide SSH rms"))
            defVar(ds, nm, Float64[r[key] for r in rows], ("con",); attrib = ["units" => u, "long_name" => ln])
        end
    end
    @printf("wrote %s (%.1f min)\n", fcomp, (time() - t_start) / 60)
end
vec_all_finite(Z) = dropdims(all(isfinite, Z; dims = 3); dims = 3)

for s in SRC_RUN; run_comp(s); end
