# Harmonic analysis by plain sinusoid least squares, for any variable with time as the last dimension.
# No tidal atlas, no nodal corrections, no inference: at every point
#     y(t) = mean + trend (t - tc)/span + sum_k [ a_k cos(w_k (t - tc)) + b_k sin(w_k (t - tc)) ]
#          = ...                        + sum_k A'_k cos(w_k (t - tc) - g'_k)
#   A'_k = sqrt(a^2 + b^2), g'_k = atan(b, a) = phase at the record centre tc.
# One design matrix for all points; normal equations solved with one Cholesky factorisation, points in chunks.
# Constituents must be resolved by the record (Rayleigh: |w_i - w_j| T >= 360 deg, warned otherwise);
# 95 days do not separate K2/S2, P1/K1, NU2/N2 (S2, K1, N2 absorb them).
# Greenwich phase / nodal-free amplitude, when comparing with an atlas (V, u, f at tc from nodal_args.py):
#     A = A' / f,   G = g' + V(tc) + u(tc)
# Needs `using LinearAlgebra, Printf` in the calling script.
#   hf = harm_fit(ssh, tsec, ["M2", "S2", "N2", "K1", "O1"]; scale = 100)   # ssh (xi, eta, time) [m] -> cm
#   hf = harm_fit(u, tsec, ["M2", "K1"])                                     # u (xi, eta, s, time) [m/s]
#   hf = harm_fit(p[300, 400, :], tsec, ["M2"])                              # one time series -> (1, K)
#   hf.amp[:, :, k], hf.pha[:, :, k]                                         # A', g' of cons[k]
#   ym = harm_predict(hf, tsec; cons = ["M2"])                               # M2 time series again
# Returns a NamedTuple; maps have the spatial shape of Y (a vector Y gives shape (1,)), then con:
#   amp, pha [deg], aerr, perr [deg] (..., con); mean, trend (change over the record), resid (residual std),
#   std (std of the data) (...); cons, freq [deg/h], tc [s, same clock as tsec], span [s], scale
# amp, aerr, mean, trend, resid, std are in the units of Y times scale.

## constituent frequencies [deg/hour] (Doodson / Schureman; same as t_tide, UTide, pyTMD); period = 360 / freq
const TIDE_FREQ = Dict(
    "M2"  => 28.9841042, "S2"  => 30.0000000, "N2"  => 28.4397295, "K2"  => 30.0821373,   # semidiurnal
    "2N2" => 27.8953548, "NU2" => 28.5125831, "MU2" => 27.9682084, "L2"  => 29.5284789,
    "K1"  => 15.0410686, "O1"  => 13.9430356, "P1"  => 14.9589314, "Q1"  => 13.3986609,   # diurnal
    "J1"  => 15.5854433, "OO1" => 16.1391017,
    "M4"  => 57.9682084, "MS4" => 58.9841042, "MN4" => 57.4238337, "M6"  => 86.9523127,   # overtides
    "MF"  => 1.0980331,  "MM"  => 0.5443747)                                              # long period

function harm_fit(Y, tsec, cons; freq = TIDE_FREQ, scale = 1.0, chunk = 20_000, verbose = true)
    clock0 = time()
    Y  = ndims(Y) == 1 ? reshape(Y, 1, :) : Y
    sp = size(Y)[1:end-1]                                       # spatial shape
    nt = length(tsec); K = length(cons); P = 2 + 2K
    size(Y, ndims(Y)) == nt || error("last dimension of Y ($(size(Y, ndims(Y)))) must match tsec ($nt)")
    tc = (tsec[1] + tsec[end]) / 2; span = (tsec[end] - tsec[1]) / 2
    th = (tsec .- tc) ./ 3600                                   # hours from the record centre

    # Rayleigh check: each pair must be separated by at least one cycle over the record
    T = (tsec[end] - tsec[1]) / 3600
    for i in 1:K, j in i+1:K
        dw = abs(freq[cons[i]] - freq[cons[j]]) * T / 360
        dw < 1 && @printf("  warning: %s and %s not resolved (%.2f cycles apart over the record)\n", cons[i], cons[j], dw)
    end

    # design matrix: mean, trend, cos / sin pair per constituent
    A = zeros(nt, P); A[:, 1] .= 1.0; A[:, 2] .= (tsec .- tc) ./ span
    for (k, c) in enumerate(cons)
        ph = deg2rad(freq[c]) .* th
        A[:, 2k + 1] .= cos.(ph); A[:, 2k + 2] .= sin.(ph)
    end
    GA = Symmetric(A' * A); F = cholesky(GA)
    verbose && @printf("harm_fit: %d points, %d records, %d constituents, %d parameters, cond(AtA) %.1f\n",
                       prod(sp), nt, K, P, cond(GA))

    # normal equations for all points: B = Y A (npts x P), solved in chunks of points (Float64)
    npts = prod(sp)
    Y2 = reshape(Y, npts, nt)
    C = fill(NaN, npts, P); rss = fill(NaN, npts); sstd = fill(NaN, npts)
    for r in Iterators.partition(1:npts, chunk)
        y = Float64.(Y2[r, :])
        B = y * A                                               # A'y for each point
        c = (F \ B')'
        ss = vec(sum(abs2, y; dims = 2))
        C[r, :] .= c
        rss[r]  .= max.(ss .- vec(sum(c .* B; dims = 2)), 0.0)  # residual sum of squares: y'y - c'A'y
        sstd[r] .= sqrt.(max.(ss ./ nt .- (B[:, 1] ./ nt) .^ 2, 0.0))
    end
    sig = sqrt.(rss ./ (nt - P))                                # residual std

    # amplitudes, phases and their errors (white noise)
    Gi = inv(GA); rs(x) = reshape(x, sp)
    amp = zeros(sp..., K); pha = similar(amp); aerr = similar(amp); perr = similar(amp)
    for k in 1:K
        a, b = C[:, 2k + 1], C[:, 2k + 2]
        selectdim(amp, ndims(amp), k)   .= scale .* rs(hypot.(a, b))
        selectdim(pha, ndims(pha), k)   .= rs(mod.(rad2deg.(atan.(b, a)), 360))
        selectdim(aerr, ndims(aerr), k) .= scale .* rs(sig) .* sqrt((Gi[2k + 1, 2k + 1] + Gi[2k + 2, 2k + 2]) / 2)
        selectdim(perr, ndims(perr), k) .= rad2deg.(selectdim(aerr, ndims(aerr), k) ./ selectdim(amp, ndims(amp), k))
    end
    verbose && @printf("harm_fit done: %.1f min\n", (time() - clock0) / 60)
    return (; amp, pha, aerr, perr, mean = scale .* rs(C[:, 1]), trend = 2 .* scale .* rs(C[:, 2]),
              resid = scale .* rs(sig), std = scale .* rs(sstd),
              cons = collect(cons), freq = [freq[c] for c in cons], tc, span, scale)
end

# Tidal time series rebuilt from a harm_fit result, at times tsec (same clock as the fit, any times):
#   sum over cons of A' cos(w (t - tc) - g'), in the units of the fit (Y times scale);
#   mean = true adds the mean and trend. Shape (spatial..., time).
function harm_predict(hf, tsec; cons = hf.cons, mean = false)
    sp = size(hf.mean); th = (tsec .- hf.tc) ./ 3600
    out = mean ? hf.mean .+ (hf.trend ./ 2) .* reshape((tsec .- hf.tc) ./ hf.span, ntuple(_ -> 1, length(sp))..., :) :
                 zeros(sp..., length(tsec))
    for c in cons
        k = findfirst(==(c), hf.cons); k === nothing && error("$c was not fitted")
        A = selectdim(hf.amp, ndims(hf.amp), k); g = deg2rad.(selectdim(hf.pha, ndims(hf.pha), k))
        ph = reshape(deg2rad(hf.freq[k]) .* th, ntuple(_ -> 1, length(sp))..., :)
        out .+= A .* cos.(ph .- g)
    end
    return out
end
