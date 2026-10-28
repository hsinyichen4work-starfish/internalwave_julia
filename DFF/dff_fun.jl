# Directional Fourier filter (DFF) helpers, the same as in DDF_on_ssh.jl (steps 5 and 6), where each step is explained
# and checked on synthetic waves. Used by DFF_on_bcef.jl. Needs FFTW.
#
# The input of dff_sector is var_xy[kx, ky, f]: a field rfft-ed in time (frequencies ≥ 0 only, f > 0 doubled, like
# MATLAB's DFF_space), then fft-ed over (x, y) and fftshift-ed. nx, ny must be odd.

# wavenumber axis [rad per grid point], fftshift-ed and reversed (MATLAB fliplr), so θ is the direction of travel
waxis(n) = reverse(2π .* collect(fftshift(fftfreq(n))))
# θ[ix, iy] in 0..360 [deg] from the grid xi axis, same layout as var_xy[:, :, f]
wdir(kx, ky) = mod.(rad2deg.(atan.(ky', kx)), 360)

# keep the wavenumbers where mask is true (and, optionally, the frequency slices where fkeep is true) and transform back
# to (x, y, t); land (land_mask == 0) is set to NaN. irfft from the one-sided spectrum = MATLAB's zero padding + real(ifft).
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

# the DFF input from a real field var[x, y, t] (nx, ny odd): rfft in time, keep f ≥ 0 (cld(nt, 2) of them), double
# f > 0, then fft over (x, y) and fftshift. var is not changed.
function dff_spectrum(var)
    nf = cld(size(var, 3), 2)
    a = rfft(var, 3)[:, :, 1:nf]
    a[:, :, 2:end] .*= 2
    fft!(a, (1, 2))
    fftshift(a, (1, 2))
end
