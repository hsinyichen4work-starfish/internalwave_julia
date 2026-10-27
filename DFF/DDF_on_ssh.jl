using NCDatasets, Dates, Statistics, FFTW

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")   # dff_run, run_dir, ddf_dir, fig_dir (set by DFF_RUN)
datadir = joinpath(run_dir, "after_joint")   # HPC output dir — contains his/dia/station files mixed together

# ============================================================================
# (1) read ssh (zeta) from every his file into one [xi, eta, time] array
# ============================================================================
mask_rho, lon_rho, lat_rho, h, pm, pn = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:, :], ds["pm"][:, :], ds["pn"][:, :]
end
lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)

# his files are roms_his.<yyyymmddHHMMSS>.nc, 4 hourly records each; sorting by name = sorting by time
his_files = sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir)))
his_files = joinpath.(datadir, his_files)
println("found $(length(his_files)) his files: ", basename(his_files[1]), " -> ", basename(his_files[end]))

# record count per file, so zeta can be allocated once instead of growing with cat
nrec = [NCDataset(f -> length(f["ocean_time"]), fname) for fname in his_files]
ntime = sum(nrec)
nx, ny = size(mask_rho)

zeta = Array{Float32}(undef, nx, ny, ntime)   # free-surface elevation [m]
ocean_time = Vector{Float64}(undef, ntime)    # seconds since 1994-01-01
i0 = 0
for (k, fname) in enumerate(his_files)
    NCDataset(fname) do ds
        idx = i0+1:i0+nrec[k]
        zeta[:, :, idx] = nomissing(ds["zeta"][:, :, :], NaN32)
        ocean_time[idx] = ds["ocean_time"][:]
    end
    global i0 += nrec[k]
    k % 50 == 0 && println("  read $k / $(length(his_files)) files")
end
date = DateTime(1994, 1, 1) .+ Second.(round.(Int, ocean_time))

# sanity checks: time must be strictly increasing with a constant step, and no fill-valued records
dt = diff(ocean_time)
all(dt .> 0) || @warn "ocean_time is not strictly increasing — duplicate/overlapping records between files?"
length(unique(dt)) == 1 || @warn "uneven time step: dt values (s) = $(unique(dt))"
bad = [t for t in 1:ntime if maximum(abs, @view zeta[:, :, t]) > 1e30]
isempty(bad) || @warn "$(length(bad)) record(s) look fill-valued: $(date[bad])"

zeta .= ifelse.(mask_rho .== 0, NaN32, zeta)   # land -> NaN (in place; mask broadcasts over time)

println("zeta: size $(size(zeta)), dt = $(dt[1] / 3600) h, ", date[1], " -> ", date[end])

# ============================================================================
# (1b) save the raw ssh with lon / lat / time to its own NetCDF file (for sharing; independent of the DDF below)
# ============================================================================
# Full 686 x 856 grid as in the ROMS output (no trimming), land = NaN. Set save_ssh = false to skip on re-runs.
save_ssh = !isfile(joinpath(ddf_dir, "roms_ssh_hourly.nc"))   # made once per run; DFF_ploting.jl needs it
ssh_fname = joinpath(ddf_dir, "roms_ssh_hourly.nc")

if save_ssh
    mkpath(dirname(ssh_fname))
    isfile(ssh_fname) && rm(ssh_fname)
    NCDataset(ssh_fname, "c") do ds
        defDim(ds, "xi_rho", nx)
        defDim(ds, "eta_rho", ny)
        defDim(ds, "time", ntime)
        defVar(ds, "ocean_time", date, ("time",);
               attrib = ["long_name" => "time", "units" => "seconds since 1994-01-01 00:00:00", "calendar" => "standard"])
        defVar(ds, "lon_rho", lon_rho, ("xi_rho", "eta_rho"); attrib = ["long_name" => "longitude", "units" => "degree_east"])
        defVar(ds, "lat_rho", lat_rho, ("xi_rho", "eta_rho"); attrib = ["long_name" => "latitude", "units" => "degree_north"])
        defVar(ds, "mask_rho", mask_rho, ("xi_rho", "eta_rho"); attrib = ["long_name" => "1 = ocean, 0 = land"])
        defVar(ds, "h", h, ("xi_rho", "eta_rho"); attrib = ["long_name" => "bathymetry (positive down)", "units" => "meter"])
        defVar(ds, "zeta", zeta, ("xi_rho", "eta_rho", "time");
               chunksizes = [nx, ny, 1], deflatelevel = 1, shuffle = true, fillvalue = NaN32,   # one snapshot per chunk, light compression
               attrib = ["long_name" => "free-surface elevation (sea surface height)", "units" => "meter"])
        ds.attrib["title"] = "Hourly ROMS sea surface height, Amazon 900 m run"
        ds.attrib["source_dir"] = datadir
        ds.attrib["grid"] = grid_fname
        ds.attrib["note"] = "zeta as saved in roms_his.*.nc, concatenated in time; longitude converted to -180..180"
    end
    println("wrote $ssh_fname ($(round(filesize(ssh_fname) / 1e9, digits = 2)) GB)")
end

# ============================================================================
# (2) build the DFF input var[x, y, t] and make sure it has no NaN
# ============================================================================
# The FFT mixes every point with every other point: one NaN anywhere turns the whole spectrum into NaN.
# So, like the MATLAB `var(isnan(var)) = 0`, land is set to 0 here and masked back to NaN after filtering.
#
# The MATLAB wavenumber axis 2π/n .* [0:floor(n/2), -floor(n/2):-1] has n points only when n is odd
# (for even n it has n+1 and no longer matches the fft2 output), hence "ensure nx and ny to be odd".
# The grid is 686 x 856 (both even), so the last xi and eta row are dropped — they sit in the boundary sponge anyway.
nx_f = isodd(nx) ? nx : nx - 1
ny_f = isodd(ny) ? ny : ny - 1

# where the NaNs are before filling: land is expected, NaN over the ocean would mean bad model output
zeta_f = @view zeta[1:nx_f, 1:ny_f, :]   # view, not a copy — zeta is ~5.5 GB for the full run
nan_land = count(isnan, @view zeta_f[mask_rho[1:nx_f, 1:ny_f] .== 0, :])
nan_ocean = count(isnan, @view zeta_f[mask_rho[1:nx_f, 1:ny_f] .== 1, :])
println("NaN before filling: $nan_land on land, $nan_ocean over ocean")
nan_ocean == 0 || @warn "NaN over the ocean — check those records before filtering"

var = zeta[1:nx_f, 1:ny_f, :]   # a copy (trimmed to odd sizes); zeta is freed below
replace!(var, NaN32 => 0f0)
land_mask = mask_rho[1:nx_f, 1:ny_f]

any(!isfinite, var) && error("var still has NaN/Inf after filling land with 0")
println("var: size $(size(var)) (nx, ny odd: $(isodd(nx_f)), $(isodd(ny_f))), no NaN/Inf, ",
        "range $(extrema(var)) m")

# ============================================================================
# (2b) remove a fitted plane a + b·i + c·j from every snapshot (ocean points) before the FFT
# ============================================================================
# The FFT treats the domain as periodic, but the barotropic tide is not: its ssh on one edge does not match the
# opposite edge, and that jump leaks barotropic energy into every wavenumber, including the internal-tide ones
# (tens of cm to metres of barotropic ssh vs a few cm of internal-tide ssh). The best-fit plane takes out the
# spatial mean and the large-scale tilt, most of that mismatch. A plane has no travel direction, so it is kept
# aside and saved, not split north/south:   var (trimmed zeta) = ssh_north + ssh_south + plane
# (apart from the Nyquist term, see step 3). With the mean removed, k = 0 is ~0 and no longer lands in "north".
ocean = land_mask .== 1
A = Float64[ones(count(ocean)) [i for i in 1:nx_f, j in 1:ny_f][ocean] [j for i in 1:nx_f, j in 1:ny_f][ocean]]
AtA_inv = inv(A' * A)   # 3 x 3, the same for every snapshot
plane_coef = zeros(3, size(var, 3))   # a [m], b [m per xi point], c [m per eta point] for each time
for t in 1:size(var, 3)
    z = @view var[:, :, t]
    a, b, c = AtA_inv * (A' * z[ocean])
    plane_coef[:, t] .= (a, b, c)
    z .-= ifelse.(ocean, Float32.(a .+ b .* (1:nx_f) .+ c .* (1:ny_f)'), 0f0)   # land stays 0
end
plane_range = extrema(plane_coef[1, :])
println("plane removed: mean ssh $(round.(plane_range, digits = 3)) m, tilt across the domain up to ",
        "$(round(maximum(abs.(plane_coef[2, :]) .* nx_f .+ abs.(plane_coef[3, :]) .* ny_f), digits = 3)) m; ",
        "var range now $(extrema(var)) m")

# zeta is not used after this (it was saved in 1b); free it — ~5.5 GB for the full run.
# zeta_f is a view into zeta, so it has to go too or zeta stays in memory.
zeta = nothing; zeta_f = nothing
GC.gc()


## DO TIME FFT
# ============================================================================
# (3) FFT in time, keep the positive frequencies only, and double them
# ============================================================================
# MATLAB: var_t = fft(var,[],3); var_t = var_t(:,:,1:ceil(nt/2)); var_t(:,:,2:end) = 2.*var_t(:,:,2:end);
#
# Why: for a real signal the negative frequencies are just complex conjugates of the positive ones, so they
# carry no new information. Keeping only ω > 0 means a wave cos(ωt - k·x) shows up once, with the sign of
# its wavenumber k set by which way it travels. With both ±ω kept, every wave would also appear at -ω, -k
# and the direction filter in step 4 could not tell east-going from west-going.
# Doubling makes up for the dropped half, so real(ifft(...)) later gives back the original amplitude.
# The time mean (frequency 0) has no negative partner, so it is not doubled.
FFTW.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))

nt = size(var, 3)
nf = cld(nt, 2)   # = MATLAB ceil(nt/2): frequencies 0, 1, ..., nf-1 cycles per record length

# rfft along dim 3 returns only frequencies 0..div(nt,2) — the half we want — at half the memory of fft.
# For even nt it also returns the Nyquist frequency, which MATLAB drops, so cut to nf.
var_t = rfft(var, 3)[:, :, 1:nf]   # ComplexF32 [x, y, frequency]
var_t[:, :, 2:end] .*= 2

dt_s = ocean_time[2] - ocean_time[1]
freq_cph = (0:nf-1) ./ (nt * dt_s / 3600)   # frequency axis [cycles per hour]
println("var_t: size $(size(var_t)), frequency 0 -> $(round(freq_cph[end], digits = 3)) cph, ",
        "resolution $(round(freq_cph[2], sigdigits = 3)) cph")

# check at one point: putting the zeros back for the negative half and taking real(ifft) must return var.
# For even nt, ceil(nt/2) drops the Nyquist frequency (period = 2 records), so that one term is lost on purpose —
# it is its own negative frequency and has no direction anyway. The check removes it before comparing.
ix, iy = cld(nx_f, 2), cld(ny_f, 2)
back = real(ifft([var_t[ix, iy, :]; zeros(ComplexF32, nt - nf)]))
nyq = iseven(nt) ? real(fft(var[ix, iy, :])[nt÷2+1]) / nt .* (-1) .^ (0:nt-1) : zeros(nt)
var_pt = var[ix, iy, :]   # keep this one time series for the north + south check in step 6
println("round-trip check at (xi, eta) = ($ix, $iy): max |error| = $(maximum(abs, back .- (var_pt .- nyq))) m, ",
        "dropped Nyquist part = $(maximum(abs, nyq)) m")

# ============================================================================
# (4) 2-D FFT in space at each frequency, then fftshift so k = 0 sits in the middle
# ============================================================================
# MATLAB: for i = 1:ceil(nt/2); var_xy(:,:,i) = fftshift(fft2(var_t(:,:,i))); end
#
# fft (not rfft): var_t is complex now, and after keeping only ω > 0 the +k and -k halves are different waves
# (travelling opposite ways), so both halves are needed — rfft would throw half the directions away.
# fft over dims (1, 2) does every frequency slice at once, no loop; fft! works in place so no second 5.5 GB copy.
# fftshift moves k = 0 from index 1 to the centre, (cld(nx_f, 2), cld(ny_f, 2)) for odd sizes, so the
# wavenumber axes run from negative to positive like a normal plot axis.
var_t_check = var_t[:, :, 2]   # keep one slice to check the transform below
fft!(var_t, (1, 2))
var_xy = fftshift(var_t, (1, 2))   # ComplexF32 [kx, ky, frequency]
var_t = nothing   # var_xy is the shifted copy; free the unshifted one
var = nothing     # var is not needed after the time FFT (var_pt keeps the one point step 6 checks); ~5.5 GB
GC.gc()

println("var_xy: size $(size(var_xy)), k = 0 at index ($(cld(nx_f, 2)), $(cld(ny_f, 2)))")

# check on one frequency: undoing the shift and the 2-D FFT must give back var_t
back = ifft(ifftshift(var_xy[:, :, 2]), (1, 2))
println("round-trip check at frequency 2: max |error| = $(maximum(abs, back .- var_t_check)), ",
        "max |var_t| = $(maximum(abs, var_t_check))")

# ============================================================================
# (5) wavenumber axes, the direction θ of every (kx, ky) point, and the angle mask
# ============================================================================
# MATLAB: kx = fliplr(fftshift(2*pi./nx.*[0:floor(nx/2), -floor(nx/2):-1])); (same for ky)
#         [KY, KX] = meshgrid(ky,kx); [theta, radii] = cart2pol(KX,KY); theta in 0..360; mask = theta1 <= theta <= theta2
#
# Axes are in rad per grid point along xi (kx) and eta (ky), so θ is measured from the grid's xi axis:
#   0° = +xi (grid east), 90° = +eta (grid north), 180° = -xi, 270° = -eta (grid south).
# This is a fair physical angle only because dx ≈ dy (899–900 m on this grid); divide by 900 m for rad/m.
#
# The reverse (MATLAB fliplr) is a sign flip: keeping only ω > 0 keeps e^{i(ωt - k·x)}, whose space part e^{-ik·x}
# the spatial FFT puts at index wavenumber -k. Reversing the axis labels gives each index its true k, so θ is the
# direction the wave travels. Without it every direction would be off by 180° (north and south swapped).
θ1, θ2 = 0, 180   # sector to keep [deg]: 0–180 = grid-north-going, 180–360 = grid-south-going
# Half-open sectors θ1 <= θ < θ2 (MATLAB uses <= at both ends): every wavenumber belongs to exactly one sector,
# so north + south = total. Waves along exactly +xi (θ = 0) go north, along -xi (θ = 180) go south,
# and k = 0 (the spatial mean, θ = atan(0, 0) = 0) goes north.

waxis(n) = reverse(2π .* collect(fftshift(fftfreq(n))))   # 2π/n .* [0:n÷2; -(n÷2):-1], shifted, then flipped
wdir(kx, ky) = mod.(rad2deg.(atan.(ky', kx)), 360)         # θ[ix, iy] in 0..360, same layout as var_xy[:, :, f]

kx, ky = waxis(nx_f), waxis(ny_f)
θ = wdir(kx, ky)
mask = θ1 .<= θ .< θ2

# check: the north and south sectors together cover every wavenumber exactly once
n_north, n_south = count(0 .<= θ .< 180), count(180 .<= θ .< 360)
n_north + n_south == length(θ) || error("north + south sectors do not cover each wavenumber once")
println("kx: $(length(kx)) points, $(round(kx[1], digits = 3)) -> $(round(kx[end], digits = 3)) rad/grid point; ",
        "ky: $(length(ky)) points; mask keeps $(count(mask)) of $(length(mask)) wavenumbers ($(θ1)°–$(θ2)°)")

# check with synthetic waves of known direction: run them through the same steps 3–4 and see where the peak lands.
# Integer wave counts (m cycles across the box) keep each wave on one FFT bin, so the peak angle is exact.
let n = 61, ntt = 48, kk = waxis(n), θt = wdir(kk, kk)
    for (name, mx, my, expect) in [("grid east", 5, 0, 0), ("grid north", 0, 5, 90), ("grid NW", -5, 5, 135), ("grid south", 0, -5, 270)]
        a = [cos(2π * (4t / ntt - mx * i / n - my * j / n)) for i in 1:n, j in 1:n, t in 0:ntt-1]
        a_t = rfft(a, 3)[:, :, 1:cld(ntt, 2)]
        a_t[:, :, 2:end] .*= 2
        a_xy = fftshift(fft(a_t, (1, 2)), (1, 2))
        peak = argmax(abs.(a_xy[:, :, 5]))   # frequency index 5 = 4 cycles per record
        println("  synthetic $name wave: peak at θ = $(θt[peak])°, expected $(expect)°")
    end
end

# ============================================================================
# (6) apply the angle mask and transform back to (x, y, t)
# ============================================================================
# MATLAB: var_theta(:,:,i) = ifft2(ifftshift(var_xy(:,:,i).*mask));   inverse 2-D FFT at each frequency
#         var_theta = cat(3,var_theta,zeros(nx,ny,floor(nt/2)));       put zeros back for the negative frequencies
#         var_theta = real(ifft(var_theta,[],3));                      inverse time FFT, real part
#         var1(land_mask == 0) = NaN;                                  land back to NaN
#
# The zero padding in MATLAB builds a full-length complex array (≈ 11 GB here). irfft gives the same answer from the
# half spectrum: it assumes the missing half is the conjugate mirror, so it computes (1/nt)[Re X0 + 2 Re Σ Y_k e^{iωt}].
# With Y_k = X_k / 2 for k ≥ 1 that is (1/nt)[Re X0 + Re Σ X_k e^{iωt}] = real(ifft(zero-padded X)) exactly.
# irfft also wants the Nyquist slot for even nt; MATLAB dropped it, so it is filled with 0.
#
# fkeep (optional, not in MATLAB) is a true/false per frequency slice: only those frequencies are kept, the rest are
# zeroed, which band-passes in time on top of the direction filter in space. Default keeps all frequencies.
function dff_sector(var_xy, mask, nt, land_mask; fkeep = trues(size(var_xy, 3)))
    a = ifftshift(var_xy .* mask .* reshape(fkeep, 1, 1, :), (1, 2))   # var_xy itself is left untouched
    ifft!(a, (1, 2))                          # back to (x, y), still one-sided in frequency
    a[:, :, 2:end] ./= 2                      # MATLAB's doubled X_k -> irfft's half-spectrum convention
    nhalf = div(nt, 2) + 1
    size(a, 3) < nhalf && (a = cat(a, zeros(eltype(a), size(a, 1), size(a, 2), nhalf - size(a, 3)); dims = 3))
    out = irfft(a, nt, 3)                     # Float32 [x, y, t]
    out .= ifelse.(land_mask .== 0, NaN32, out)
    out
end

# check on a small synthetic case (like the example in the MATLAB header): a grid-north wave plus a twice-as-big
# grid-south wave at the same frequency. The north sector must return only the north wave, the south only the south.
# Also check irfft against the MATLAB zero-padding way.
let n = 61, ntt = 48, kk = waxis(n), θt = wdir(kk, kk)
    wave(amp, my) = [amp * cos(2π * (4t / ntt - my * j / n)) for i in 1:n, j in 1:n, t in 0:ntt-1]
    a_north, a_south = wave(1.0, 5), wave(2.0, -5)
    a_t = rfft(a_north .+ a_south, 3)[:, :, 1:cld(ntt, 2)]
    a_t[:, :, 2:end] .*= 2
    a_xy = fftshift(fft(a_t, (1, 2)), (1, 2))
    lm = ones(n, n)
    got_n = dff_sector(a_xy, 0 .<= θt .< 180, ntt, lm)
    got_s = dff_sector(a_xy, 180 .<= θt .< 360, ntt, lm)
    println("  synthetic: north sector error $(maximum(abs, got_n .- a_north)), south sector error $(maximum(abs, got_s .- a_south)) ",
            "(wave amplitudes 1 and 2)")
    mat = ifft(ifftshift(a_xy .* (0 .<= θt .< 180), (1, 2)), (1, 2))
    mat = real(ifft(cat(mat, zeros(ComplexF64, n, n, ntt - size(mat, 3)); dims = 3), 3))
    println("  irfft vs MATLAB zero-padding: max difference $(maximum(abs, got_n .- mat))")

    # frequency band: add a north wave at 7 cycles per record; keeping only frequency index 5 (4 cycles) must drop it
    a_t2 = rfft(a_north .+ a_south .+ [cos(2π * (7t / ntt - 5j / n)) for i in 1:n, j in 1:n, t in 0:ntt-1], 3)[:, :, 1:cld(ntt, 2)]
    a_t2[:, :, 2:end] .*= 2
    a_xy2 = fftshift(fft(a_t2, (1, 2)), (1, 2))
    got_band = dff_sector(a_xy2, 0 .<= θt .< 180, ntt, lm; fkeep = (1:cld(ntt, 2)) .== 5)
    println("  synthetic band filter: north sector at 4 cycles only, error $(maximum(abs, got_band .- a_north))")
end

# M2 band: periods 9–15 h. A wide semidiurnal band, so it also holds N2, S2 and K2 (M2 dominates);
# the inertial period here (~5°N) is ~140 h, well outside. Any record longer than ~2 days has frequencies in it.
T_M2_band = (9, 15)   # [hours]
fband_M2 = (1 / T_M2_band[2], 1 / T_M2_band[1])   # [cycles per hour]
fkeep_M2 = fband_M2[1] .<= freq_cph .<= fband_M2[2]
println("M2 band $(T_M2_band[1])–$(T_M2_band[2]) h ($(round.(fband_M2, digits = 4)) cph) keeps $(count(fkeep_M2)) frequencies")
count(fkeep_M2) == 0 && @warn "no frequency falls in the M2 band — the record is too short; ssh_*_M2 will be all zero"

# M2 outputs only: also drop wavelengths longer than 3λ1 (|k| < kcut). The barotropic tide is hundreds to thousands
# of km long, so what the plane in 2b left of it sits at the smallest |k|; mode-1 internal tides (λ1) are kept.
# The domain is only ~5λ1 across, so this removes just k = 0 and the lowest few harmonics.
# The all-frequency outputs keep every |k|, so ssh_north + ssh_south + plane still add back to var.
# Caveat: on the shallow inner shelf the barotropic tide itself is short (√(g h)·12.42 h ≈ 440 km at 10 m depth,
# ≈ 310 km at 5 m), so some of it can pass the cut there.
dx_m = mean(1 ./ pm)   # grid spacing [m] (≈ 900; dx ≈ dy on this grid)
λ1 = 120e3             # mode-1 M2 wavelength [m] — placeholder; check against the |k| spectrum in DFF_ploting.jl
kcut = 2π / (3λ1 / dx_m)   # [rad per grid point]: keep wavelengths shorter than 3λ1
kmag = sqrt.(kx .^ 2 .+ (ky') .^ 2)   # |k| [rad per grid point], same layout as θ
println("M2 wavenumber cut: λ1 = $(λ1 / 1e3) km, keep wavelengths < $(3λ1 / 1e3) km; ",
        "removes $(count(kmag .<= kcut)) of $(length(kmag)) wavenumbers")

# M2-band ssh variance against |k| (before the cut), north and south sectors, for choosing λ1 and kcut.
# Parseval with the one-sided doubled spectrum: mean variance over (x, y, t) = Σ |X|² / (2 (nx ny nt)²),
# so each bin holds the M2 ssh variance [m²] from that |k| ring; the bins add up to the total M2 variance.
dk = 2π / max(nx_f, ny_f)   # bin width = the coarser FFT wavenumber step
kbin = floor.(Int, kmag ./ dk) .+ 1
nkbin = maximum(kbin)
P = zeros(nx_f, ny_f)
for f in findall(fkeep_M2)
    P .+= abs2.(@view var_xy[:, :, f])
end
P ./= 2 * (nx_f * ny_f * nt)^2
spec_M2_north, spec_M2_south = zeros(nkbin), zeros(nkbin)
for I in eachindex(P)
    (0 <= θ[I] < 180 ? spec_M2_north : spec_M2_south)[kbin[I]] += P[I]
end
k_center = ((1:nkbin) .- 0.5) .* dk   # [rad per grid point]
println("M2 |k| spectrum: $nkbin bins, width 2π/($(round(2π / dk * dx_m / 1e3, digits = 1)) km), ",
        "total M2 variance north $(round(sum(spec_M2_north), sigdigits = 3)) m², south $(round(sum(spec_M2_south), sigdigits = 3)) m²")

# the fields to make: four direction sectors, each as all frequencies, M2 and bc
#   north / south                    the two half planes (0–180°, 180–360°): north + south = total
#   north_smallsec / south_smallsec  narrow sectors ±30° about grid north (60–120°) and grid south (240–300°): waves
#                                    travelling close to along the eta axis only, so the oblique waves that go into the
#                                    half-plane sectors are left out (smallsec north + south is not the total)
#   ssh_<sector>      all frequencies, all wavelengths
#   ssh_<sector>_M2   M2 band (9–15 h), wavelengths < 3λ1
#   ssh_<sector>_bc   all frequencies, wavelengths < 3λ1: an approximate baroclinic (bc) ssh. Not a true
#                     barotropic/baroclinic split — on the shallow shelf the barotropic tide is short enough to pass
#                     the cut, and anything else longer than 3λ1 (e.g. large eddies, basin-scale subtidal signal) is
#                     removed too. Time-mean and very low frequency features have no clear travel direction, so they
#                     split between the sectors.
sectors = [("north", 0, 180, "grid-north-going"), ("south", 180, 360, "grid-south-going"),
           ("north_smallsec", 60, 120, "grid-north-going, small sector"), ("south_smallsec", 240, 300, "grid-south-going, small sector")]
λstr = "$(3λ1 / 1e3) km"
ssh_out = NamedTuple[]
for (sec, a1, a2, desc) in sectors
    dmask = a1 .<= θ .< a2   # half-open, as in step 5
    push!(ssh_out, (name = "ssh_$sec", mask = dmask, fkeep = trues(nf),
                    long_name = "$desc ssh (wavenumber direction $a1-$a2 deg from xi axis), all frequencies"))
    push!(ssh_out, (name = "ssh_$(sec)_M2", mask = dmask .& (kmag .> kcut), fkeep = fkeep_M2,
                    long_name = "$desc ssh ($a1-$a2 deg), M2 band (periods 9-15 h), wavelengths < $λstr only"))
    push!(ssh_out, (name = "ssh_$(sec)_bc", mask = dmask .& (kmag .> kcut), fkeep = trues(nf),
                    long_name = "$desc ssh ($a1-$a2 deg), all frequencies, wavelengths < $λstr only (approximate baroclinic)"))
end

# ============================================================================
# (7) make each field, check land is masked back, and write it to NetCDF — one field at a time
# ============================================================================
# Each field is ~5.5 GB for the full run (12 fields ≈ 66 GB), and dff_sector needs ~16 GB of temporaries, so the fields
# are not all kept in memory: each is computed, checked, written and freed before the next.
# rebuild = false: if the file already exists, keep it and only add the fields it does not have yet (e.g. the small
# sectors added later) — the existing ones are not recomputed. rebuild = true: delete the file and make everything.
# Land was put back to NaN inside dff_sector (MATLAB's last loop: var1(land_mask == 0) = NaN). This grid has no land,
# so the check is trivially true here, but it guards a grid that does.
out_dir = ddf_dir
out_fname = joinpath(out_dir, "ssh_ddf_north_south.nc")
rebuild = false
mkpath(out_dir)
land = land_mask .== 0

rebuild && isfile(out_fname) && rm(out_fname)
new_file = !isfile(out_fname)
ds = NCDataset(out_fname, new_file ? "c" : "a")
if new_file
    defDim(ds, "xi_rho", nx_f)
    defDim(ds, "eta_rho", ny_f)
    defDim(ds, "time", nt)
    defDim(ds, "kbin", nkbin)

    defVar(ds, "ocean_time", date, ("time",);
           attrib = ["long_name" => "time", "units" => "seconds since 1994-01-01 00:00:00", "calendar" => "standard"])
    defVar(ds, "lon_rho", lon_rho[1:nx_f, 1:ny_f], ("xi_rho", "eta_rho"); attrib = ["units" => "degree_east"])
    defVar(ds, "lat_rho", lat_rho[1:nx_f, 1:ny_f], ("xi_rho", "eta_rho"); attrib = ["units" => "degree_north"])
    defVar(ds, "mask_rho", land_mask, ("xi_rho", "eta_rho"); attrib = ["long_name" => "1 = ocean, 0 = land"])

    # plane removed in step 2b: plane(i, j, t) = plane_a + plane_b * i + plane_c * j, with i, j the 1-based xi, eta index
    defVar(ds, "plane_a", plane_coef[1, :], ("time",); attrib = ["long_name" => "plane removed before DFF: constant", "units" => "meter"])
    defVar(ds, "plane_b", plane_coef[2, :], ("time",); attrib = ["long_name" => "plane removed before DFF: slope along xi", "units" => "meter per xi point"])
    defVar(ds, "plane_c", plane_coef[3, :], ("time",); attrib = ["long_name" => "plane removed before DFF: slope along eta", "units" => "meter per eta point"])

    defVar(ds, "k_center", k_center ./ dx_m, ("kbin",); attrib = ["long_name" => "|k| bin centre", "units" => "rad per meter"])
    defVar(ds, "spec_M2_north", spec_M2_north, ("kbin",);
           attrib = ["long_name" => "M2-band ssh variance per |k| bin, grid-north sector, before the wavenumber cut", "units" => "m2"])
    defVar(ds, "spec_M2_south", spec_M2_south, ("kbin",);
           attrib = ["long_name" => "M2-band ssh variance per |k| bin, grid-south sector, before the wavenumber cut", "units" => "m2"])
else
    # appending: the existing file must be on the same grid and hours
    (ds.dim["xi_rho"], ds.dim["eta_rho"], ds.dim["time"]) == (nx_f, ny_f, nt) && ds["ocean_time"][:] == date ||
        error("$out_fname has a different grid or time axis — set rebuild = true")
end

pt = Dict{String, Vector{Float32}}()   # time series at (ix, iy) of each field made in this run, for the checks below
first_map = Dict{String, Matrix{Float32}}()   # first snapshot of each field made, for the read-back check
for o in ssh_out
    if haskey(ds, o.name)
        println("$(o.name): already in the file, skipped")
        continue
    end
    a = dff_sector(var_xy, o.mask, nt, land_mask; fkeep = o.fkeep)
    n_land_ok = count(isnan, @view a[land, :])
    n_ocean_nan = count(isnan, @view a[.!land, :])
    (n_land_ok == count(land) * nt && n_ocean_nan == 0) ||
        error("$(o.name): land not all NaN ($n_land_ok of $(count(land) * nt)) or NaN over ocean ($n_ocean_nan)")
    # one chunk per snapshot: reading a map at one time is fast; a time series at one point reads all chunks
    defVar(ds, o.name, a, ("xi_rho", "eta_rho", "time");
           chunksizes = [nx_f, ny_f, 1], fillvalue = NaN32,
           attrib = ["long_name" => o.long_name, "units" => "meter"])
    pt[o.name] = a[ix, iy, :]
    first_map[o.name] = a[:, :, 1]
    println("$(o.name): written, land check passed, std at (xi, eta) = ($ix, $iy) $(round(std(pt[o.name]), sigdigits = 3)) m")
    a = nothing
    GC.gc()   # free this field and dff_sector's temporaries (~16 GB for the full run) before the next one
end

ds.attrib["title"] = "Directional Fourier filter (DFF) of ROMS ssh (zeta) into grid-north and grid-south going parts"
ds.attrib["source_dir"] = datadir
ds.attrib["method"] = "rfft in time, keep f >= 0, double f > 0; fft2 in space; angle mask theta1 <= theta < theta2 " *
                      "(theta of wavenumber from grid xi axis, wave travel direction); inverse via irfft. " *
                      "Based on DFF_space.m (O. Siyanbola 2025, after Gong et al. 2020)."
ds.attrib["grid"] = "$grid_fname, last xi and eta row dropped so nx, ny are odd"
ds.attrib["grid_north_sector_deg"] = "0 <= theta < 180"
ds.attrib["grid_south_sector_deg"] = "180 <= theta < 360"
ds.attrib["grid_north_smallsec_deg"] = "60 <= theta < 120"
ds.attrib["grid_south_smallsec_deg"] = "240 <= theta < 300"
ds.attrib["M2_band_period_hours"] = collect(T_M2_band)
ds.attrib["M2_band_cph"] = collect(fband_M2)
ds.attrib["M2_lambda1_m"] = λ1
ds.attrib["M2_kcut_wavelength_m"] = 3λ1
ds.attrib["note"] = "A best-fit plane (plane_a/b/c) is removed from each snapshot before the DFF: " *
                    "zeta = ssh_north + ssh_south + plane. Nyquist frequency dropped for even nt (as in DFF_space.m). " *
                    "M2 and bc fields also drop wavelengths >= M2_kcut_wavelength_m, so ssh_north_M2 + ssh_south_M2 is not the full M2 signal; " *
                    "the bc fields are an approximate baroclinic ssh (wavelength cut only, not a barotropic/baroclinic decomposition). " *
                    "The smallsec fields keep only the narrow sectors (60-120, 240-300 deg), so they do not add up to the total."
close(ds)
println("wrote $out_fname")

# north + south must give back var (plane removed), apart from the Nyquist term dropped in step 3 (removed here as before)
if haskey(pt, "ssh_north") && haskey(pt, "ssh_south")
    resid = pt["ssh_north"] .+ pt["ssh_south"] .- (var_pt .- nyq)
    println("at (xi, eta) = ($ix, $iy) north + south vs var: max |error| = $(maximum(abs, resid)) m")
end
# small sector vs its half plane at one point, for comparison: the small sector has less energy over the domain, but
# at a single point it can be larger, where the oblique waves left out partly cancel it in the half plane
for dir in ("north", "south"), sfx in ("", "_M2", "_bc")
    big, small = "ssh_$(dir)$sfx", "ssh_$(dir)_smallsec$sfx"
    haskey(pt, small) && haskey(pt, big) &&
        println("$small vs $big at (xi, eta) = ($ix, $iy): std $(round(std(pt[small]), sigdigits = 3)) vs $(round(std(pt[big]), sigdigits = 3)) m")
end

# read back the fields written in this run (first snapshot) and the time axis to make sure the file is right
NCDataset(out_fname) do ds
    println("read back: ocean_time $(ds["ocean_time"][1]) -> $(ds["ocean_time"][end])")
    for (name, m) in first_map
        println("  $name[:, :, 1] max |difference| = $(maximum(abs, ds[name][:, :, 1] .- m))")
    end
end
