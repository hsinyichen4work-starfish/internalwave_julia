using NCDatasets, Dates, Statistics, FFTW

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/after_joint"   # HPC output dir — contains his/dia/station files mixed together

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

var = zeta[1:nx_f, 1:ny_f, :]   # a copy, so zeta keeps its NaN land for plotting
replace!(var, NaN32 => 0f0)
land_mask = mask_rho[1:nx_f, 1:ny_f]

any(!isfinite, var) && error("var still has NaN/Inf after filling land with 0")
println("var: size $(size(var)) (nx, ny odd: $(isodd(nx_f)), $(isodd(ny_f))), no NaN/Inf, ",
        "range $(extrema(var)) m")


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
println("round-trip check at (xi, eta) = ($ix, $iy): max |error| = $(maximum(abs, back .- (var[ix, iy, :] .- nyq))) m, ",
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

# the real data: grid-north-going (0–180°) and grid-south-going (180–360°) ssh, all frequencies
ssh_north = dff_sector(var_xy, 0 .<= θ .< 180, nt, land_mask)
ssh_south = dff_sector(var_xy, 180 .<= θ .< 360, nt, land_mask)

# M2 band: periods 9–15 h. A wide semidiurnal band, so it also holds N2, S2 and K2 (M2 dominates);
# the inertial period here (~5°N) is ~140 h, well outside. Any record longer than ~2 days has frequencies in it.
T_M2_band = (9, 15)   # [hours]
fband_M2 = (1 / T_M2_band[2], 1 / T_M2_band[1])   # [cycles per hour]
fkeep_M2 = fband_M2[1] .<= freq_cph .<= fband_M2[2]
println("M2 band $(T_M2_band[1])–$(T_M2_band[2]) h ($(round.(fband_M2, digits = 4)) cph) keeps $(count(fkeep_M2)) frequencies")
count(fkeep_M2) == 0 && @warn "no frequency falls in the M2 band — the record is too short; ssh_*_M2 will be all zero"

ssh_north_M2 = dff_sector(var_xy, 0 .<= θ .< 180, nt, land_mask; fkeep = fkeep_M2)
ssh_south_M2 = dff_sector(var_xy, 180 .<= θ .< 360, nt, land_mask; fkeep = fkeep_M2)

# north + south must give back var, apart from the Nyquist term dropped in step 3 (removed here as before)
resid = ssh_north[ix, iy, :] .+ ssh_south[ix, iy, :] .- (var[ix, iy, :] .- nyq)
println("ssh_north, ssh_south: size $(size(ssh_north)); at (xi, eta) = ($ix, $iy) north + south vs var: ",
        "max |error| = $(maximum(abs, resid)) m; std north $(std(ssh_north[ix, iy, :])) m, south $(std(ssh_south[ix, iy, :])) m")
println("M2 at (xi, eta) = ($ix, $iy): std north $(std(ssh_north_M2[ix, iy, :])) m, south $(std(ssh_south_M2[ix, iy, :])) m")

# ============================================================================
# (7) check land is masked back, then save the four fields and the time axis to NetCDF
# ============================================================================
# Land was put back to NaN inside dff_sector (MATLAB's last loop: var1(land_mask == 0) = NaN).
# This grid has no land, so the check is trivially true here, but it guards a grid that does.
out_dir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/ddf_ssh"
out_fname = joinpath(out_dir, "ssh_ddf_north_south.nc")
mkpath(out_dir)

ssh_out = [("ssh_north", ssh_north, "grid-north-going ssh (wavenumber direction 0-180 deg from xi axis), all frequencies"),
           ("ssh_south", ssh_south, "grid-south-going ssh (wavenumber direction 180-360 deg from xi axis), all frequencies"),
           ("ssh_north_M2", ssh_north_M2, "grid-north-going ssh, M2 band (periods 9-15 h) only"),
           ("ssh_south_M2", ssh_south_M2, "grid-south-going ssh, M2 band (periods 9-15 h) only")]

land = land_mask .== 0
for (name, a, _) in ssh_out
    n_land_ok = count(isnan, @view a[land, :])
    n_ocean_nan = count(isnan, @view a[.!land, :])
    (n_land_ok == count(land) * nt && n_ocean_nan == 0) ||
        error("$name: land not all NaN ($n_land_ok of $(count(land) * nt)) or NaN over ocean ($n_ocean_nan)")
end
println("land mask check passed: land is NaN and the ocean has no NaN in all four fields ($(count(land)) land points)")

# one chunk per snapshot: reading a map at one time is fast; a time series at one point reads all chunks
# (~5.5 GB per field for the full run, ~22 GB in total, uncompressed)
isfile(out_fname) && rm(out_fname)
NCDataset(out_fname, "c") do ds
    defDim(ds, "xi_rho", nx_f)
    defDim(ds, "eta_rho", ny_f)
    defDim(ds, "time", nt)

    defVar(ds, "ocean_time", date, ("time",);
           attrib = ["long_name" => "time", "units" => "seconds since 1994-01-01 00:00:00", "calendar" => "standard"])
    defVar(ds, "lon_rho", lon_rho[1:nx_f, 1:ny_f], ("xi_rho", "eta_rho"); attrib = ["units" => "degree_east"])
    defVar(ds, "lat_rho", lat_rho[1:nx_f, 1:ny_f], ("xi_rho", "eta_rho"); attrib = ["units" => "degree_north"])
    defVar(ds, "mask_rho", land_mask, ("xi_rho", "eta_rho"); attrib = ["long_name" => "1 = ocean, 0 = land"])

    for (name, a, long_name) in ssh_out
        defVar(ds, name, a, ("xi_rho", "eta_rho", "time");
               chunksizes = [nx_f, ny_f, 1], fillvalue = NaN32,
               attrib = ["long_name" => long_name, "units" => "meter"])
    end

    ds.attrib["title"] = "Directional Fourier filter (DFF) of ROMS ssh (zeta) into grid-north and grid-south going parts"
    ds.attrib["source_dir"] = datadir
    ds.attrib["method"] = "rfft in time, keep f >= 0, double f > 0; fft2 in space; angle mask theta1 <= theta < theta2 " *
                          "(theta of wavenumber from grid xi axis, wave travel direction); inverse via irfft. " *
                          "Based on DFF_space.m (O. Siyanbola 2025, after Gong et al. 2020)."
    ds.attrib["grid"] = "$grid_fname, last xi and eta row dropped so nx, ny are odd"
    ds.attrib["grid_north_sector_deg"] = "0 <= theta < 180"
    ds.attrib["grid_south_sector_deg"] = "180 <= theta < 360"
    ds.attrib["M2_band_period_hours"] = collect(T_M2_band)
    ds.attrib["M2_band_cph"] = collect(fband_M2)
    ds.attrib["note"] = "Nyquist frequency dropped for even nt (as in DFF_space.m); k = 0 (spatial mean) goes to the north sector"
end
println("wrote $out_fname")

# read back one field and the time axis to make sure the file is right
NCDataset(out_fname) do ds
    err = maximum(abs, ds["ssh_north"][:, :, 1] .- ssh_north[:, :, 1])
    println("read back: ocean_time $(ds["ocean_time"][1]) -> $(ds["ocean_time"][end]), ssh_north[:, :, 1] max |difference| = $err")
end
