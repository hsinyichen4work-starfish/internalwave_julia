# compute_rx1.jl
#
# Ported from compute_rx1.m — offline estimate of the ROMS grid stiffness
# ratio (Haney number, rx1), assuming a flat sea surface (zeta = 0).
#
#   rx1 = | dz_k + dz_{k-1} | / | sz_k - sz_{k-1} |
#
# with, for a pair of neighboring columns,
#   dz_k = z(i+1,k) - z(i,k),   sz_k = z(i+1,k) + z(i,k)
#
# h       : 2D bathymetry at rho-points [m], positive down
# returns : (rx1_max, rx1_field, loc) — loc = (i, j, k) of the maximum,
#           with j indexing h's first dimension and i its second (same
#           convention as the MATLAB version, whatever orientation h has)

function compute_rx1(h::AbstractMatrix, theta_s, theta_b, hc, N)
    Mp, Lp = size(h)

    # ---- build z at rho-points for every layer, assuming zeta = 0 ----
    z_r = zeros(Mp, Lp, N)
    for k in 1:N
        sigma = (k - N - 0.5) / N   # rho-point sigma, matches ROMS convention

        Csur = theta_s > 0 ? (1 - cosh(theta_s * sigma)) / (cosh(theta_s) - 1) : -sigma^2
        C = theta_b > 0 ? (exp(theta_b * Csur) - 1) / (1 - exp(-theta_b)) : Csur

        S = (hc * sigma .+ h .* C) ./ (hc .+ h)
        z_r[:, :, k] = h .* S       # zeta = 0  =>  z = h * S (negative down)
    end

    # ---- rx1 in xi-direction (i, i+1 neighbors) ----
    dz_xi = diff(z_r; dims = 2)                              # [Mp, Lp-1, N]
    sz_xi = z_r[:, 1:end-1, :] .+ z_r[:, 2:end, :]
    rx1_xi = abs.(dz_xi[:, :, 2:end] .+ dz_xi[:, :, 1:end-1]) ./
             abs.(sz_xi[:, :, 2:end] .- sz_xi[:, :, 1:end-1])  # [Mp, Lp-1, N-1]

    # ---- rx1 in eta-direction (j, j+1 neighbors) ----
    dz_eta = diff(z_r; dims = 1)                             # [Mp-1, Lp, N]
    sz_eta = z_r[1:end-1, :, :] .+ z_r[2:end, :, :]
    rx1_eta = abs.(dz_eta[:, :, 2:end] .+ dz_eta[:, :, 1:end-1]) ./
              abs.(sz_eta[:, :, 2:end] .- sz_eta[:, :, 1:end-1]) # [Mp-1, Lp, N-1]

    # ---- combine: max of xi/eta at each common (i,j,k) footprint ----
    Mc = min(size(rx1_xi, 1), size(rx1_eta, 1))
    Lc = min(size(rx1_xi, 2), size(rx1_eta, 2))
    # MATLAB's max() ignores NaN
    nmax(a, b) = isnan(a) ? b : (isnan(b) ? a : max(a, b))
    rx1_field = nmax.(rx1_xi[1:Mc, 1:Lc, :], rx1_eta[1:Mc, 1:Lc, :])

    rx1_max = -Inf
    idx = CartesianIndex(1, 1, 1)
    for I in CartesianIndices(rx1_field)
        v = rx1_field[I]
        if !isnan(v) && v > rx1_max
            rx1_max = v
            idx = I
        end
    end
    jm, im, km = Tuple(idx)
    loc = (i = im, j = jm, k = km)

    @printf("Estimated max rx1 = %.4f  at (i=%d, j=%d, k=%d), h there ~= %.1f m\n",
        rx1_max, im, jm, km, h[jm, im])
    return rx1_max, rx1_field, loc
end
